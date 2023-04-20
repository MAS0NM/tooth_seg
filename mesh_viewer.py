from utils import visualize_mesh
from utils import read_dir
import vedo
from random import shuffle
import threading
from utils import visualize_mesh
from infer import infer
import argparse
import ast
def view_mesh(constrain='FLP'):
    files = read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain=constrain)
    shuffle(files)
    for file in files:
        mesh = vedo.load(file)
        visualize_mesh(mesh)

def view_mesh(path, args):
    pred_mesh = infer(args.config_file, args.ckpt_path, path, args.pygco)
    trth_mesh = vedo.load(args.mesh_file)
    visualize_mesh(pred_mesh)
    visualize_mesh(trth_mesh)
    
# view_mesh(constrain='')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")
    parser.add_argument("-gco", "--pygco", type=ast.literal_eval, metavar="", help="pygco", default=True)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", default='./checkpoints/iMeshSegNet_17_Classes_32_f_best_loss-v1.ckpt')
    parser.add_argument("-mesh", "--mesh_file", type=str, metavar="", help="mesh file", default='./dataset/3D_scans_ds/Z8HJR6YS_upper_FLP_AUG03_.vtk')
    args = parser.parse_args()
    view_mesh('./dataset/3D_scans_ds/014F9HTN_upper_FLP.vtk', args)