import os
import sys

sys.path.append(os.getcwd())

import argparse
import ast
import time

import numpy as np
import pytorch_lightning as pl
import torch
import vedo
from omegaconf import OmegaConf
from pygco import cut_from_graph

from model.LitModule import LitModule
from postprocessing import *
from utils import visualize_mesh


def infer(cfg_path: str, ckpt_path: str, mesh_file: str, refine: bool, device='cuda'):
    cfg = OmegaConf.load(cfg_path)
    if len(cfg.infer.devices) == 1 and cfg.infer.accelerator == "gpu":
        device = f"cuda:{cfg.infer.devices[0]}"
    elif len(cfg.infer.devices) > 1 and cfg.infer.accelerator == "gpu":
        device = "cuda:0"
    module = LitModule(cfg).load_from_checkpoint(ckpt_path)
    model = module.model.to(device)
    model.eval()
    
    mesh = vedo.load(mesh_file)
    start_time = time.time()
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:,1:]
    cells = points[ids].reshape(N, 9).astype(dtype='float32')
    normals = vedo.vedo2trimesh(mesh).face_normals
    barycenters = mesh.cell_centers()

    mesh_d = mesh.clone()
    predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
    input_data = gen_metadata_inf(cfg, mesh, device)

    infer_start_time = time.time()
    with torch.no_grad():
        tensor_prob_output = model(input_data["cells"], input_data["KG_12"],input_data["KG_6"])
    print("Inference time: ", time.time()-infer_start_time)
    patch_prob_output = tensor_prob_output.cpu().numpy()

    for i_label in range(cfg.model.num_classes):
        predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

    # output downsampled predicted labels
    mesh2 = mesh_d.clone()
    mesh2.celldata['labels'] = predicted_labels_d
    
    if not refine:
        print("Total time: ", time.time()-start_time)
        return mesh2
    else:
        # refinement
        print('\tRefining by pygco...')
        round_factor = 100
        patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, cfg.model.num_classes)

        # parawise
        pairwise = (1 - np.eye(cfg.model.num_classes, dtype=np.int32))

        #edges
        cell_ids = np.asarray(mesh_d.faces())

        lambda_c = 30
        edges = np.empty([1, 3], order='C')
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei==2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi/2.0:
                        edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                        edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c*round_factor
        edges = edges.astype(np.int32)

        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        # output refined result
        mesh3 = mesh_d.clone()
        mesh3.celldata['labels'] = refine_labels

        print("Total time: ", time.time()-start_time)
        return mesh3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")
    parser.add_argument("-gco", "--pygco", type=ast.literal_eval, metavar="", help="pygco", default=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", default='./checkpoints/iMeshSegNet_17_Classes_32_f_best_loss.ckpt')
    parser.add_argument("-mesh", "--mesh_file", type=str, metavar="", help="mesh file", default='./dataset/3D_scans_per_patient_obj_files_b1/01A6GW4A/01A6GW4A_lower.vtk')

    args = parser.parse_args()

    output = infer(args.config_file, args.ckpt_path, args.mesh_file, args.pygco)
    
    visualize_mesh(output)
    
    outuput_name = args.mesh_file.split('/')[-1]
    vedo.write(output, f'./inf_output/predicted_{outuput_name}')