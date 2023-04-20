import os
import sys
from joblib import Parallel, delayed

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
from utils import *
import csv
from os import cpu_count
from tqdm import tqdm


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
    # print("Inference time: ", time.time()-infer_start_time)
    patch_prob_output = tensor_prob_output.cpu().numpy()

    for i_label in range(cfg.model.num_classes):
        predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

    # output downsampled predicted labels
    mesh2 = mesh_d.clone()
    mesh2.celldata['labels'] = predicted_labels_d
    
    if not refine:
        # print("Total time: ", time.time()-start_time)
        return mesh2
    else:
        # refinement
        # print('\tRefining by pygco...')
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

        # print("Total time: ", time.time()-start_time)
        return mesh3

def cal_precision(path, args):
    mesh_trth = vedo.load(path)
    mesh_pred = infer(args.config_file, args.ckpt_path, path, args.pygco)
    orig_labels = mesh_trth.celldata['labels']
    pred_labels = mesh_pred.celldata['labels']
    acc = float(np.mean(orig_labels == pred_labels))
    # print(f"acc: {acc*100}%")
    with open('./pred&truth.csv', 'a') as f:
        f.write('\t'.join([get_sample_name(path), ','.join(map(str, orig_labels)), ','.join(map(str, pred_labels))]) + '\n')
    with open('./precision.csv', 'a') as f:
        f.write('\t'.join([get_sample_name(path), str(acc)]) + '\n')
    return acc

def cal_precision_all(args, filelist):
    # clear the file contents
    with open('./pred&truth.csv', 'w') as f:
        pass
    with open('./precision.csv', 'w') as f:
        pass
    Parallel(n_jobs=4)(delayed(cal_precision)(path, args) for path in tqdm(filelist, desc='calculating precision'))
    # cal_precision(filelist, args)
    total_acc, AUG_acc = 0, 0
    total_count, AUG_count = 0, 0
    with open('precision.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = list(reader)
    for item, i in lines:
        if 'AUG' in item:
            AUG_count += 1
            AUG_acc += float(i)
        total_acc += float(i)
        total_count += 1
    return f'total accuracy is {total_acc / total_count * 100}% \n \
        augmented item accuracy is {AUG_acc / AUG_count * 100}% \n \
            origin item accuracy is {(total_acc-AUG_acc)/(total_count-AUG_count)*100}%'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")
    parser.add_argument("-gco", "--pygco", type=ast.literal_eval, metavar="", help="pygco", default=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", default='./checkpoints/iMeshSegNet_17_Classes_32_f_best_loss-v1.ckpt')
    parser.add_argument("-mesh", "--mesh_file", type=str, metavar="", help="mesh file", default='./dataset/3D_scans_ds/Z8HJR6YS_upper_FLP_AUG03_.vtk')

    args = parser.parse_args()

    # output = infer(args.config_file, args.ckpt_path, args.mesh_file, args.pygco)
    # mesh_ds = vedo.load(args.mesh_file)
    # paral_visualize_mesh(mesh_ds, output)
    
    # cal_precision(mesh_ds, output)
    val_list = []
    with open('./dataset/FileLists/val_list.csv', 'r') as f:
        reader = csv.reader(f)
        val_list = list(reader)
    val_list = [item[0] for item in val_list]
    print(cal_precision_all(args, val_list))
    # outuput_name = args.mesh_file.split('/')[-1]
    # vedo.write(output, f'./inf_output/predicted_{outuput_name}')