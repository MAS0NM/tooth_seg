from utils import visualize_mesh
from utils import read_dir
import vedo
from random import shuffle
from utils import visualize_mesh
import glob
import tkinter as tk


def view_mesh(mesh_path, constrain='FLP', with_color=True):
    if not mesh_path:
        files = read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain=constrain)
        shuffle(files)
        for file in files:
            mesh = vedo.load(file)
            visualize_mesh(mesh, with_color=with_color)
    else:
        mesh = vedo.load(mesh_path)
        visualize_mesh(mesh, with_color=with_color)
        

if __name__ == '__main__':
    window = tk.Tk()
    window.title(f"mesh viewer")
    vtks = glob.glob(f"./dataset/3D_scans_ds/*.vtk")
    
    window.geometry("400x600")
    
    listbox = tk.Listbox(window)
    listbox.pack(fill=tk.BOTH, expand=1)    
    
    for name in vtks:
        listbox.insert(tk.END, name.split('/')[-1])

    listbox.bind("<Double-Button-1>", lambda x:\
        view_mesh(vtks[listbox.curselection()[0]]))
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    window.mainloop()