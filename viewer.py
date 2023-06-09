import os
import numpy as np
import vedo
import pyvista as pv
import tkinter as tk
from utils import visualize_mesh
from infer import infer
from omegaconf import OmegaConf
from model.LitModule import LitModule
import argparse
import ast

def recursively_get_file(dir_path, ext):
    ls = []
    if os.path.isfile(dir_path):
        return [dir_path]
    files = os.listdir(dir_path)
    for file in files:
        if os.path.isfile(f'{dir_path}/{file}'):
            if f'.{ext}' in file:
                ls.append(f'{dir_path}/{file}')
            else:
                return []
        else:
            ls += recursively_get_file(f'{dir_path}/{file}', ext)
    return ls
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")
    parser.add_argument("-gco", "--pygco", type=ast.literal_eval, metavar="", help="pygco", default=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", default='./checkpoints/iMeshSegNet_17_Classes_32_f_best_DSC-v2.ckpt')
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
    
    dataset_dir = './dataset/3D_scans_ds'
    # stls = recursively_get_file(dataset_dir, ext='vtk')
    # # vis_single_mesh(stls[0], pps[0])
    # window = tk.Tk()
    # window.title("mesh annotation viewer")
    
    # window.geometry("400x600")
    
    # listbox = tk.Listbox(window)
    # listbox.pack(fill=tk.BOTH, expand=1)
    
    # for name in stls:
    #     listbox.insert(tk.END, name.split('/')[-1])
    
    # listbox.bind("<Double-Button-1>", lambda x:\
    #     (visualize_mesh(stls[listbox.curselection()[0]], mode='cells')))
        #  visualize_mesh(infer(cfg=cfg, model=model, mesh_file=stls[listbox.curselection()[0]], refine=True))))
    visualize_mesh(infer(cfg=cfg, model=model, mesh_file='./dataset/3D_scans_ds/0OF8OOCX_upper_R_AUG00_.vtk', refine=True))
    # listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    # window.mainloop()