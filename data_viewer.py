from utils import visualize_mesh
from utils import read_dir
import vedo
from random import shuffle
def view_mesh(constrain='FLP'):
    files = read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain=constrain)
    shuffle(files)
    for file in files:
        mesh = vedo.load(file)
        visualize_mesh(mesh)
        
view_mesh(constrain='')