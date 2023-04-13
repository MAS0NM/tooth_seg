import pyvista as pv
import numpy as np
from collections import Counter
from vedo import load
import json
import torch
import glob
import os
import re

def get_sample_name(filepath, with_ext=False):
    pattern = r"[\\/](\w+)\.(obj|vtk|json)$"
    match = re.search(pattern, filepath)
    if match:
        return match.group(1) if not with_ext else f'{match.group(1)}.{match.group(2)}'
    else:
        print(filepath, 'no match')
        return ''

def read_filenames(dir_paths):
    '''
        read mesh files or the label files
    '''
    lower = []
    upper = []
    for dir_path in dir_paths:
        for sub_dir in os.listdir(dir_path):
            path = os.path.join(dir_path, sub_dir)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    if 'lower' in file and os.path.isfile(os.path.join(path, file)):
                        lower.append(os.path.join(path, file).replace('\\','/'))
                    elif 'upper' in file and os.path.isfile(os.path.join(path, file)):
                        upper.append(os.path.join(path, file).replace('\\','/'))
    return lower, upper


def filelist_checker(jaw_list, label_list):
    '''
        check if the sample name of mesh and label match and correct them
    '''
    corr_jaw_list, corr_lab_list = [], []
    for i in range(len(jaw_list)):
        jaw_sample_name = get_sample_name(jaw_list[i])
        lab_sample_name = get_sample_name(label_list[i])
        if jaw_sample_name == lab_sample_name:
            corr_jaw_list.append(jaw_list[i].replace('\\','/'))
            corr_lab_list.append(label_list[i].replace('\\','/'))
        else:
            for j in range(len(label_list)):
                lab_sample_name = get_sample_name(label_list[j])
                if jaw_sample_name == lab_sample_name:
                    corr_jaw_list.append(jaw_list[i].replace('\\','/'))
                    corr_lab_list.append(label_list[j].replace('\\','/'))
                    
    return corr_jaw_list, corr_lab_list

def visualize_mesh(mesh, mode='cells'):
    # 17 colors, 0 for gum
    colormap = [
        [255, 255, 255],
        [0, 128, 128],
        [255, 105, 180],
        [128, 0, 128],
        [255, 165, 0],
        [46, 139, 87],
        [255, 0, 0],
        [30, 144, 255],
        [139, 0, 139],
        [50, 205, 50],
        [255, 69, 0],
        [0, 191, 255],
        [128, 0, 0],
        [60, 179, 113],
        [255, 20, 147],
        [0, 128, 0],
        [0, 255, 127]
    ]
    colormap = np.array(colormap, dtype=np.int8)
    
    # reconstruct pv object
    pv_mesh = pv.PolyData(mesh._data)
    plotter = pv.Plotter()
    
    if mode == 'points':
        labels_by_point = mesh.pointdata['labels']
        pv_mesh.point_data['labels'] = labels_by_point
        pv_mesh.point_data['colors'] = colormap[pv_mesh.point_data['labels']]
        
    elif mode == 'cells':
        labels_by_cell = mesh.celldata['labels']
        pv_mesh.cell_data['labels'] = labels_by_cell
        pv_mesh.cell_data['colors'] = colormap[pv_mesh.cell_data['labels']]
        
    pv_mesh.set_active_scalars("colors")  # set labels as the active scalars
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="colors", show_scalar_bar=False)
    plotter.show()
    
    
def set_face_label(mesh, labels_by_point, mode='count'):
    labels_by_face = []
    if mode == 'count':
        for face in mesh.faces():
            vertex_labels = [labels_by_point[i] if i < len(labels_by_point) else 0 for i in face]
            if all(x == vertex_labels[0] for x in vertex_labels):
                # print(face, vertex_labels)
                labels_by_face.append(vertex_labels[0])
            else:
                label_count = Counter(vertex_labels)
                most_common_label = label_count.most_common(1)[0][0]
                labels_by_face.append(most_common_label)
    elif mode == 'distance':
        for face in mesh.faces():
            vertex_labels = [labels_by_point[i] for i in face]
            vertices = mesh.points()[face]
            centroid = np.array([sum([x[i] for x in vertices])/3 for i in range(3)], dtype=np.uint8)
            distances = np.array([sum(np.square(vertices-centroid))], dtype=np.float16)
            nearest_point_index = np.argmin(distances)
            this_label = vertex_labels[nearest_point_index]
            labels_by_face.append(this_label)
    return labels_by_face


def make_labeled_mesh_data(mesh_path, label_path):
    mesh = load(mesh_path)
    with open(label_path, 'r') as f:
        attr = json.load(f)
    labels_by_point = attr['labelByPoints']
    labels_by_face = attr['labelByFaces']
    mesh.pointdata['labels'] = labels_by_point
    mesh.celldata['labels'] = labels_by_face
    return mesh


def dataReader(dir_path='./dataset/FileLists/fileList_lower_ds.txt', visualize=False):
    meshes = []
    with open(dir_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            mesh_path = line[0]
            label_path = line[1]
            mesh = make_labeled_mesh_data(mesh_path, label_path)
            meshes.append(mesh)
            if visualize:
                visualize_mesh(mesh)
    return meshes

def read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain='FLP'):
    files = []
    file_form = dir_path + r'*.' + extension
    paths = glob.glob(file_form)
    for file in paths:
        file = file.replace('\\','/')
        if constrain in file:
            files.append(file)
    return files
            
def centring(mesh):
    mesh.points(pts=mesh.points()-mesh.center_of_mass())
    return mesh
                        

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
            

def get_graph_feature(x, k=20, idx=None, dim9=False, device='cpu'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, num_points, k)
