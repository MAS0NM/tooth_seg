import vedo
from omegaconf import OmegaConf
import torch
import numpy as np
from utils import get_graph_feature
from utils import centring


def gen_metadata_inf(cfg: OmegaConf, mesh: vedo.Mesh, device='cuda'):
    mesh = centring(mesh)
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:,1:]
    cells = points[ids].reshape(N, 9).astype(dtype='float32')
    normals = vedo.vedo2trimesh(mesh).face_normals
    normals.setflags(write=1)
    barycenters = mesh.cell_centers()
    
    #normalized data
    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
        cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
        cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
        barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
        normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

    X = np.column_stack((cells, barycenters, normals))
    X = X.transpose(1, 0)

    meta = dict()
    meta["cells"] = torch.from_numpy(X).unsqueeze(0).to(device, dtype=torch.float)

    print("Getting KG6 and KG12.")
    KG_6 = get_graph_feature(torch.from_numpy(X[9:12, :]).unsqueeze(0), k=6).squeeze(0)
    KG_12 = get_graph_feature(torch.from_numpy(X[9:12, :]).unsqueeze(0), k=12).squeeze(0)
    meta["KG_6"] = KG_6.unsqueeze(0).to(device, dtype=torch.float)
    meta["KG_12"] = KG_12.unsqueeze(0).to(device, dtype=torch.float)

    return meta