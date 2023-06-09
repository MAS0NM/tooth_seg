import os
import sys
from joblib import Parallel, delayed
import random
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

from torchviz import make_dot

def infer(cfg, model, mesh_file, cfg_path=None, ckpt_path=None, refine=True, device='cuda', print_time=False, with_raw_output=False):
    if not cfg and cfg_path:
        cfg = OmegaConf.load(cfg_path)
    if len(cfg.infer.devices) == 1 and cfg.infer.accelerator == "gpu":
        device = f"cuda:{cfg.infer.devices[0]}"
    elif len(cfg.infer.devices) > 1 and cfg.infer.accelerator == "gpu":
        device = "cuda:0"
    if not model and ckpt_path:
        module = LitModule(cfg).load_from_checkpoint(ckpt_path)
        model = module.model.to(device)
        model.eval()
    
    start_time = time.time()
    if type(mesh_file) == str:
        mesh = vedo.load(mesh_file)
    else:
        mesh = mesh_file
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:,1:]
    cells = points[ids].reshape(N, 9).astype(dtype='float32')
    normals = vedo.vedo2trimesh(mesh).face_normals
    barycenters = mesh.cell_centers()

    mesh_d = mesh.clone()
    predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
    input_data = gen_metadata_inf(cfg, mesh, device)
    load_time = time.time() - start_time

    infer_start_time = time.time()
    with torch.no_grad():
        tensor_prob_output = model(input_data["cells"], input_data["KG_12"],input_data["KG_6"])
    inf_time = time.time() - infer_start_time
    patch_prob_output = tensor_prob_output.cpu().numpy()

    for i_label in range(cfg.model.num_classes):
        predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

    # output downsampled predicted labels
    mesh2 = mesh_d.clone()
    mesh2.celldata['labels'] = predicted_labels_d
    
    if not refine:
        if print_time:
            print(f"mesh loading time:{load_time}\ninference time:{inf_time}\ntotal time:{total_time}")
        if with_raw_output:
            return mesh2, tensor_prob_output
        return mesh2
    else:
        post_start_time = time.time()
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
        post_time = time.time()-post_start_time
        total_time = time.time()-start_time
        if print_time:
            print(f"mesh loading time:{load_time}\ninference time:{inf_time}\npost processing time:{post_time}\ntotal time:{total_time}")
        if with_raw_output:
            return mesh3, tensor_prob_output
        return mesh3

def cal_precision(cfg, model, path, args):
    mesh_trth = vedo.load(path)
    mesh_pred = infer(cfg, model, path, args.config_file, args.ckpt_path, args.pygco)
    orig_labels = mesh_trth.celldata['labels']
    pred_labels = mesh_pred.celldata['labels']
    acc = float(np.mean(orig_labels == pred_labels))
    # print(f"acc: {acc*100}%")
    with open('./pred&truth.csv', 'a') as f:
        f.write('\t'.join([get_sample_name(path), ','.join(map(str, orig_labels)), ','.join(map(str, pred_labels))]) + '\n')
    with open('./precision.csv', 'a') as f:
        f.write('\t'.join([get_sample_name(path), str(acc)]) + '\n')
    return acc

def cal_precision_all(cfg, model, args, filelist):
    # clear the file contents
    with open('./pred&truth.csv', 'w') as f:
        pass
    with open('./precision.csv', 'w') as f:
        pass
    Parallel(n_jobs=4)(delayed(cal_precision)(cfg, model, path, args) for path in tqdm(filelist, desc='calculating precision'))
    # cal_precision(filelist, args)
    total_acc, R_acc, T_acc, S_acc = 0, 0, 0, 0
    total_count, R_count, T_count, S_count = 0, 0, 0, 0
    with open('precision.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = list(reader)
    for item, i in lines:
        if '_R_' in item:
            R_count += 1
            R_acc += float(i)
        if '_T_' in item:
            T_count += 1
            T_acc += float(i)
        if '_S_' in item:
            S_count += 1
            S_acc += float(i)
        total_acc += float(i)
        total_count += 1
    return f'total accuracy is {total_acc / total_count * 100}% \n '
# rotated item accuracy is {R_acc / R_count * 100}% \n \
# translated item accuracy is {T_acc / T_count * 100}% \n \
# rescaled item accuracy is {S_acc / S_count * 100}% \n \
# origin item accuracy is {(total_acc-R_acc-T_acc-S_acc)/(total_count-R_count-T_count-S_count)*100}%'
            
            
def visualize_network(cfg_path: str, device='cuda'):
    cfg = OmegaConf.load(cfg_path)
    if len(cfg.infer.devices) == 1 and cfg.infer.accelerator == "gpu":
        device = f"cuda:{cfg.infer.devices[0]}"
    elif len(cfg.infer.devices) > 1 and cfg.infer.accelerator == "gpu":
        device = "cuda:0"
    module = LitModule(cfg)
    model = module.model.to(device)
    model.eval()
    cells, kg6, kg12 = torch.randn(1,15,10000).cuda(), torch.randn(1,6,10000, 12).cuda(), torch.randn(1,6,10000,6).cuda(), 
    y = model(cells, kg6, kg12)
    g = make_dot(y, params=dict(model.named_parameters()))
    g.render('dcg', view=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")
    parser.add_argument("-gco", "--pygco", type=ast.literal_eval, metavar="", help="pygco", default=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", default='./checkpoints/iMeshSegNet_17_Classes_32_f_rotate_enhance-v1.ckpt')
    parser.add_argument("-mesh", "--mesh_file", type=str, metavar="", help="mesh file", default='./dataset/3D_scans_ds/Z8HJR6YS_upper_FLP_AUG03_.vtk')

    args = parser.parse_args()
    
    '''
    load configure file and the model
    '''
    cfg = OmegaConf.load(args.config_file)
    if len(cfg.infer.devices) == 1 and cfg.infer.accelerator == "gpu":
        device = f"cuda:{cfg.infer.devices[0]}"
    elif len(cfg.infer.devices) > 1 and cfg.infer.accelerator == "gpu":
        device = "cuda:0"
    module = LitModule(cfg).load_from_checkpoint(args.ckpt_path)
    model = module.model.to(device)
    model.eval()
    
    
    '''
    calculate precision
    '''
    val_list = []
    with open('./dataset/FileLists/filelist_final.csv', 'r') as f:
        reader = csv.reader(f)
        val_list = list(reader)
    val_list = [item[0] for item in val_list]
    random.shuffle(val_list)
    print(cal_precision_all(cfg, model, args, val_list[:100]))
    
    
    '''
    visualize network
    '''
    # visualize_network(args.config_file)
    # outuput_name = args.mesh_file.split('/')[-1]W
    # vedo.write(output, f'./inf_output/predicted_{outuput_name}')
    
    
    '''
    do inference on single mesh
    '''
    # mesh_with_refine = infer(cfg=cfg, model=model, mesh_file='./dataset/3D_scans_ds/01328DDN_upper.vtk', refine=True)
    # mesh_wiou_refine = infer(cfg=cfg, model=model, mesh_file='./dataset/3D_scans_ds/Z8HJR6YS_upper_FLP.vtk', refine=False)
    # visualize_mesh(mesh_with_refine)
    # visualize_mesh(mesh_wiou_refine)
    
    
    '''
    generate mesh predictions
    '''
    # pred_dir = './dataset/3D_pred/'
    # mesh_ls = read_dir('./dataset/3D_scans_ds/', extension='vtk', constrain='')
    # mesh_ls = [i for i in mesh_ls if 'AUG' not in i]
    # for mesh_path in tqdm([mesh_ls[0]]):
    #     mesh_with_refine = infer(cfg=cfg, model=model, mesh_file=mesh_path, refine=True)
    #     mesh_name = get_sample_name(mesh_path)
    #     des_file = os.path.join(pred_dir, mesh_name + '.vtk')
    #     mesh_with_refine.write(des_file)
    #     visualize_mesh(mesh_with_refine)