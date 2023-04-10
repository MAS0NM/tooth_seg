import numpy as np
import os        
import vedo
import json
import torch
import glob
import vtk
import random
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from utils import *

# 2 batches of scans: 3D_scans_per_patient_obj_files_b1 and 3D_scans_per_patient_obj_files_b2
dir_paths = ['./dataset/3D_scans_per_patient_obj_files_b1', './dataset/3D_scans_per_patient_obj_files_b2']
# label diractory in those 2 batches
label_paths = [dir_path + '/ground-truth_labels_instances_b' + str(idx+1) for idx, dir_path in enumerate(dir_paths)]
# file list path
fileLists_path = './dataset/FileLists'
# down sampled file path
downsampled_dataset = './dataset/3D_scans_ds'

lower_jaws = []
upper_jaws = []
lower_labels = []
upper_labels = []

mesh_counter = 0
print('number of cpus', cpu_count())
lower_jaws, upper_jaws = read_filenames(dir_paths)
lower_labels, upper_labels = read_filenames(label_paths)
lower_jaws, lower_labels = filelist_checker(lower_jaws, lower_labels)
upper_jaws, upper_labels = filelist_checker(upper_jaws, upper_labels)
fileList_lower = '/fileList_lower.txt'
fileList_upper = '/fileList_upper.txt'

for file_list, jaws, labels in [(fileList_lower, lower_jaws, lower_labels), (fileList_upper, upper_jaws, upper_labels)]:
    with open(fileLists_path + file_list, 'w') as f:
        for i in range(len(jaws)):
            f.write('\t'.join([jaws[i], labels[i]])+'\n')
                    
    
def downsample_with_index(mesh, target_cells):
    mesh_ds = mesh.clone()
    mesh_ds = mesh_ds.decimate(target_cells / mesh.ncells)
    indices = np.array([mesh.closest_point(i, return_point_id=True) for i in mesh_ds.points()])
    if mesh_ds.ncells > target_cells - 1:
        mesh_ds.delete_cells([0])
    
    return mesh_ds, indices


def downsample(jaw_path, lab_path, target_cells, fileList, op_dir, all_ds_files):
    '''
        down sample and store in vtk form with celldata['labels'] and pointdata['labels']
    '''
    sampleName = get_sample_name(jaw_path)
    path_mesh = op_dir + sampleName + '.vtk'
    if sampleName in all_ds_files:
        print(sampleName, 'already exists, continue')
        with open(fileList, 'a') as f:
            f.write(path_mesh+'\n')
        return
    
    mesh = vedo.load(jaw_path)
            
    # Add cell data with labels
    with open(lab_path, 'r') as f:
        labels_by_point = np.array(json.load(f)['instances'], dtype=np.uint8)
    
    # raw mesh
    # labels_by_face = set_face_label(mesh, labels_by_point, mode='count')
    # mesh.pointdata['labels'] = labels_by_point
    # mesh.celldata['labels'] = labels_by_face
    
    # downsample
    try:
        mesh_ds, indices = downsample_with_index(mesh, target_cells)
        labels_by_point_ds = labels_by_point[indices].tolist()
        labels_by_face_ds = set_face_label(mesh_ds, labels_by_point_ds, mode='count')
        mesh_ds.pointdata['labels'] = labels_by_point_ds
        mesh_ds.celldata['labels'] = labels_by_face_ds
    except:
        print(jaw_path, lab_path, 'error', mesh.npoints, labels_by_point.shape)
        # visualize_mesh(mesh)
        return
    
    # write filelist
    with open(fileList, 'a') as f:
        f.write(path_mesh+'\n')
    
    # write mesh vtk
    mesh_ds.write(path_mesh)
    
    # print(path_mesh, jaw_path, mesh_ds.ncells)
    global mesh_counter
    mesh_counter += 1


def do_downsample(jaws, labels, target_cells, filelist, ds_dir = './dataset/3D_scans_ds/'):
    with open(filelist, 'w'):
        pass
    # get all files in 3d_scans_ds
    all_ds_files = glob.glob(os.path.join(ds_dir, "*"))
    all_ds_files = [get_sample_name(path) for path in all_ds_files]
    Parallel(n_jobs=cpu_count())(delayed(downsample)(jaw, lab, target_cells, filelist, ds_dir, all_ds_files) for jaw, lab in tqdm(iterable=list(zip(jaws, labels)), desc='down sampling'))
    
def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    # create a vtk transform object
    transform = vtk.vtkTransform()
    # set random rotation angles
    angle_x = np.random.uniform(rotate_X[0], rotate_X[1])
    angle_y = np.random.uniform(rotate_Y[0], rotate_Y[1])
    angle_z = np.random.uniform(rotate_Z[0], rotate_Z[1])
    # set random translation distances
    dist_x = np.random.uniform(translate_X[0], translate_X[1])
    dist_y = np.random.uniform(translate_Y[0], translate_Y[1])
    dist_z = np.random.uniform(translate_Z[0], translate_Z[1])
    # set random scaling factors
    factor_x = np.random.uniform(scale_X[0], scale_X[1])
    factor_y = np.random.uniform(scale_Y[0], scale_Y[1])
    factor_z = np.random.uniform(scale_Z[0], scale_Z[1])
    # apply the transformations
    transform.RotateX(angle_x)
    transform.RotateY(angle_y)
    transform.RotateZ(angle_z)
    transform.Translate(dist_x, dist_y, dist_z)
    transform.Scale(factor_x, factor_y, factor_z)
    # get the transformation matrix
    matrix = transform.GetMatrix()
    return matrix
        
        
def augment(mesh_path, out_dir, mode, aug_num):
    if 'AUG' in mesh_path:
        return
    mesh_name = get_sample_name(mesh_path)
    if mode == 'flip':
        if 'FLP' in mesh_path:
            return
        mesh_op_pth = os.path.join(out_dir, mesh_name + '_FLP' + '.vtk')
        if mesh_op_pth in existing_mesh_files:
            # print(f'{mesh_path} exists, skip')
            return
        mesh = vedo.load(mesh_path)
        mesh.mirror(axis='x')
        mesh = centring(mesh)
        if mesh.celldata['labels'].shape[0] != 10000:
            print(mesh_path, mesh.celldata['labels'].shape)
        mesh.write(mesh_op_pth)
    else:
        for i in range(aug_num):
            mesh_op_pth = os.path.join(out_dir, mesh_name+"_AUG%02d_" %i + '.vtk')
            if mesh_op_pth in existing_mesh_files:
                # print(f'{mesh_path} exists, skip')
                continue
            mesh = vedo.load(mesh_path)
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                    translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                    scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]) #use default random setting
            mesh.apply_transform(vtk_matrix)
            mesh = centring(mesh)
            mesh.celldata['Normal'] = vedo.vedo2trimesh(mesh).face_normals
            if mesh.celldata['labels'].shape[0] != 10000:
                print(mesh_path, mesh.celldata['labels'].shape)
            mesh.write(mesh_op_pth)
        
        
def do_augmentation(ip_dir, op_dir, aug_num):
    # filp the meshes to double the size
    filels = glob.glob(f"{ip_dir}/*.vtk")
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'flip', aug_num) for item in tqdm(filels, desc="flipping"))
    # random tranform
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'trsf', aug_num) for item in tqdm(filels, desc="transforming"))


def make_train_val_list(filelist_path='./dataset/FileLists/filelist_final.csv'):
    filelist = []
    trn_list = []
    val_list = []
    with open(filelist_path, 'r') as f:
        for line in f.readlines():
            filelist.append(line)
    random.shuffle(filelist)
    t_v_ratio = 0.8
    trn_num = int(len(filelist)*t_v_ratio)
    trn_list = filelist[:trn_num]
    val_list = filelist[trn_num:]
    trn_path = './dataset/FileLists/trn_list.csv'
    val_path = './dataset/FileLists/val_list.csv'
    with open(trn_path, 'w') as f:
        for line in trn_list:
            f.write(line)
    with open(val_path, 'w') as f:
        for line in val_list:
            f.write(line)
            
            
def make_filelist_final(mode='upper'):
    filelist = []
    for file in glob.glob(r'./dataset/3D_scans_ds/*.vtk'):
        if mode in file:
            file = file.replace('\\', '/')
            filelist.append(file)
    with open('./dataset/FileLists/filelist_final.csv', 'w', newline='') as f:
        for line in filelist:
            f.write(line+'\n')
    make_train_val_list()


if __name__ == '__main__':
    target_cells = 10001
    existing_mesh_files = read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain='')
    # do_downsample(lower_jaws, lower_labels, target_cells=target_cells, filelist='./dataset/FileLists/fileList_lower_ds.txt', ds_dir = './dataset/3D_scans_ds/')
    # print(f"lower batch is done, {mesh_counter} mesh files down sampled this time")
    do_downsample(upper_jaws, upper_labels, target_cells=target_cells, filelist='./dataset/FileLists/fileList_upper_ds.txt', ds_dir = './dataset/3D_scans_ds/')
    print(f"upper batch is done, {mesh_counter} mesh files down sampled this time")
    do_augmentation(ip_dir='./dataset/3D_scans_ds/', op_dir='./dataset/3D_scans_ds/', aug_num=20)
    make_filelist_final()