import numpy as np
import torch
import glob
import os
import re


try:
    import pyvista as pv
except:
    print('cannot import pyvista')
    
    
def get_sample_name(filepath, with_ext=False):
    '''
        extract the sample name from the file path like
        ./dataset/3D_scans_per_patient_obj_files_b1/YVGEP1R6/YVGEP1R6_lower.obj
        to
        YVGEP1R6_lower
        or
        YVGEP1R6_lower.obj
    '''
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


def visualize_mesh(mesh, mode='cells', with_color=True):
    '''
        visualize the mesh with color, input mesh should be vedo.mesh
    '''
    if isinstance(mesh, str):
        import vedo
        mesh = vedo.Mesh(mesh)
        
    pv_mesh = pv.PolyData(mesh._data)
    plotter = pv.Plotter()
    
    if not with_color:
        plotter.add_mesh(pv_mesh, color='white')
    
    if with_color:
        if mode == 'points':
            labels_by_point = mesh.pointdata['labels']
            pv_mesh.point_data['colors'] = labels_by_point
            
        elif mode == 'cells':
            labels_by_cell = mesh.celldata['labels']
            pv_mesh.cell_data['colors'] = labels_by_cell
            
        pv_mesh.set_active_scalars("colors")  # set labels as the active scalars
        plotter.add_mesh(pv_mesh, scalars=pv_mesh.cell_data['colors'], cmap='tab20', show_scalar_bar=False, )
    
    plotter.add_axes()
    plotter.show()


def read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain='FLP'):
    '''
        read all the files in the directory that has the required extension
        return a list
    '''
    files = []
    file_form = dir_path + r'*.' + extension
    paths = glob.glob(file_form)
    for file in paths:
        file = file.replace('\\','/')
        if constrain in file or not constrain == '':
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
    '''
        perform graph cut on the mesh to refine
    '''
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


def read_filelist(file_path):
    filelist = []
    with open(file_path, 'r') as f:
        filelist = f.readlines()
        filelist = [i.strip() for i in filelist]
    return filelist
