import numpy as np
import os        
import vedo
import json
import glob
import random
import h5py
from collections import Counter
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from utils import *
from omegaconf import OmegaConf
from mesh_dataset import gen_metadata
try:
    import vtk
except:
    print('cannot import vtk')

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

            
def rearrange(nparry):
    # 32 permanent teeth
    nparry[nparry == 17] = 1
    nparry[nparry == 37] = 1
    nparry[nparry == 16] = 2
    nparry[nparry == 36] = 2
    nparry[nparry == 15] = 3
    nparry[nparry == 35] = 3
    nparry[nparry == 14] = 4
    nparry[nparry == 34] = 4
    nparry[nparry == 13] = 5
    nparry[nparry == 33] = 5
    nparry[nparry == 12] = 6
    nparry[nparry == 32] = 6
    nparry[nparry == 11] = 7
    nparry[nparry == 31] = 7
    nparry[nparry == 21] = 8
    nparry[nparry == 41] = 8
    nparry[nparry == 22] = 9
    nparry[nparry == 42] = 9
    nparry[nparry == 23] = 10
    nparry[nparry == 43] = 10
    nparry[nparry == 24] = 11
    nparry[nparry == 44] = 11
    nparry[nparry == 25] = 12
    nparry[nparry == 45] = 12
    nparry[nparry == 26] = 13
    nparry[nparry == 46] = 13
    nparry[nparry == 27] = 14
    nparry[nparry == 47] = 14
    nparry[nparry == 18] = 15
    nparry[nparry == 38] = 15
    nparry[nparry == 28] = 16
    nparry[nparry == 48] = 16
    # deciduous teeth
    nparry[nparry == 55] = 3
    nparry[nparry == 55] = 3
    nparry[nparry == 54] = 4
    nparry[nparry == 74] = 4
    nparry[nparry == 53] = 5
    nparry[nparry == 73] = 5
    nparry[nparry == 52] = 6
    nparry[nparry == 72] = 6
    nparry[nparry == 51] = 7
    nparry[nparry == 71] = 7
    nparry[nparry == 61] = 8
    nparry[nparry == 81] = 8
    nparry[nparry == 62] = 9
    nparry[nparry == 82] = 9
    nparry[nparry == 63] = 10
    nparry[nparry == 83] = 10
    nparry[nparry == 64] = 11
    nparry[nparry == 84] = 11
    nparry[nparry == 65] = 12
    nparry[nparry == 85] = 12
    
    return nparry


def downsample_with_index(mesh, target_cells):
    mesh_ds = mesh.clone()
    mesh_ds = mesh_ds.decimate(target_cells / mesh.ncells)
    indices = np.array([mesh.closest_point(i, return_point_id=True) for i in mesh_ds.points()])
    if mesh_ds.ncells > target_cells - 1:
        for i in range(mesh_ds.ncells):
            mesh_cp = mesh_ds.clone()
            mesh_cp.delete_cells([i])
            if mesh_cp.ncells == target_cells - 1:
                return mesh_cp, indices
        
    return mesh_ds, indices


def downsample(jaw_path, lab_path, target_cells, fileList, op_dir, all_ds_files):
    '''
        down sample and store in vtk form with celldata['labels'] and pointdata['labels']
    '''
    sampleName = get_sample_name(jaw_path)
    path_mesh = op_dir + sampleName + '.vtk'
    if sampleName in all_ds_files:
        # print(sampleName, 'already exists, continue')
        with open(fileList, 'a') as f:
            f.write(path_mesh+'\n')
        return
    
    mesh = vedo.load(jaw_path)
            
    # Add cell data with labels
    with open(lab_path, 'r') as f:
        labels_by_point = np.array(json.load(f)['labels'], dtype=np.uint8)
        labels_by_point = rearrange(labels_by_point)
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
        
        
def augment(mesh_path, out_dir, mode, aug_num, existing_mesh_files):
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
        if mesh.ncells != 10000:
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
        
        
def do_augmentation(ip_dir, op_dir, aug_num, existing_mesh_files):
    # filp the meshes to double the size
    filels = glob.glob(f"{ip_dir}/*.vtk")
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'flip', aug_num, existing_mesh_files) for item in tqdm(filels, desc="flipping"))
    # random tranform
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'trsf', aug_num, existing_mesh_files) for item in tqdm(filels, desc="transforming"))


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
    return trn_list, val_list, file_list
            
            
def make_filelist_final(mode='upper'):
    filelist = []
    for file in glob.glob(r'./dataset/3D_scans_ds/*.vtk'):
        if mode in file:
            file = file.replace('\\', '/')
            filelist.append(file)
    with open('./dataset/FileLists/filelist_final.csv', 'w', newline='') as f:
        for line in filelist:
            f.write(line+'\n')
    return make_train_val_list()


def read_filelist(file_path):
    filelist = []
    with open(file_path, 'r') as f:
        filelist = f.readlines()
        filelist = [i.strip() for i in filelist]
    return filelist
    

def parallel_gen_metadata(idx, pth, patch_size, data_dict):
    mesh = None
    try:
        mesh = vedo.load(pth)
    except:
        print(f'error loading {pth}')
        return
    if mesh:
        meta_data = gen_metadata(mesh, patch_size, mode='vedo')
        data_dict["labels"][idx] = meta_data['labels']
        data_dict["cells"][idx] = meta_data['cells']
        data_dict["KG_6"][idx] = meta_data['KG_6']
        data_dict["KG_12"][idx] = meta_data['KG_12']
        print('data compute done')
        

def vtk2h5(cfg, trn_list, val_list, jaw_type, parallel = False):
    '''
        read in all the .vtk files and process the data into h5 format and store
        the .h5 file will contain the train set and the validation set
        NB: h5py file cannot be pickled so parallel is currently not working
    '''
    f = h5py.File(os.path.join(cfg.train.h5_dir, f'{jaw_type}.hdf5'), 'a')
    for item in ["trn", "val"]:
        length = len(eval(f"{item}_list"))
        if item not in f.keys():
            data_group = f.create_group(item)
            data_group.create_dataset("labels", data=np.zeros((length, 1, cfg.model.patch_size), dtype=np.int32), dtype="int32")
            data_group.create_dataset("cells", data=np.zeros((length, cfg.model.num_channels, cfg.model.patch_size), dtype=np.float32), dtype="float32")
            data_group.create_dataset("KG_6", data=np.zeros((length, 3*2, cfg.model.patch_size, 6), dtype=np.float32), dtype="float32")
            data_group.create_dataset("KG_12", data=np.zeros((length, 3*2, cfg.model.patch_size, 12), dtype=np.float32), dtype="float32")
        else:
            data_group = f.require_group(item)
            data_group.require_dataset("labels", shape=(length, 1, cfg.model.patch_size), dtype=np.int32)
            data_group.require_dataset("cells", shape=(length, cfg.model.num_channels, cfg.model.patch_size), dtype=np.float32)
            data_group.require_dataset("KG_6", shape=(length, 3*2, cfg.model.patch_size, 6), dtype=np.float32)
            data_group.require_dataset("KG_12", shape=(length, 3*2, cfg.model.patch_size, 12), dtype=np.float32)
            
        filelist = eval(f"{item}_list")
        if parallel:
            print(f'start parallel generating {item} meta data')
            data_dict = {}
            data_dict["labels"] = np.zeros((length, 1, cfg.model.patch_size), dtype=np.int32)
            data_dict["cells"] = np.zeros((length, cfg.model.num_channels, cfg.model.patch_size), dtype=np.float32)
            data_dict["KG_6"] = np.zeros((length, 3*2, cfg.model.patch_size, 6), dtype=np.float32)
            data_dict["KG_12"] = np.zeros((length, 3*2, cfg.model.patch_size, 12), dtype=np.float32)
            Parallel(n_jobs=cpu_count())(delayed(parallel_gen_metadata)(idx, pth, cfg.model.patch_size, data_dict) for idx, pth in enumerate(tqdm(filelist, desc=f"{item}")))
            print('paralled done')
            f[item]["labels"] = data_dict['labels']
            f[item]["cells"] = data_dict['cells']
            f[item]["KG_6"] = data_dict['KG_6']
            f[item]["KG_12"] = data_dict['KG_12']
        else:
            print(f'start generating {item} meta data')
            for idx, pth in enumerate(tqdm(filelist, desc=f"{item}")):
                try:
                    mesh = vedo.load(pth)
                except:
                    print(f'error loading {pth}')
                    continue
                if not mesh:
                    print(f'error loading {pth}')
                    continue
                meta_data = gen_metadata(mesh, cfg.model.patch_size, mode='vedo')
                f[item]["labels"][idx] = meta_data['labels']
                f[item]["cells"][idx] = meta_data['cells']
                f[item]["KG_6"][idx] = meta_data['KG_6']
                f[item]["KG_12"][idx] = meta_data['KG_12']
    f.close()
    
    
if __name__ == '__main__':
    cfg = OmegaConf.load("config/default.yaml")
    target_cells = 10001
    existing_mesh_files = read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain='')
    do_downsample(upper_jaws, upper_labels, target_cells=target_cells, filelist='./dataset/FileLists/fileList_upper_ds.txt', ds_dir = './dataset/3D_scans_ds/')
    print(f"upper batch is done")
    do_augmentation(ip_dir='./dataset/3D_scans_ds/', op_dir='./dataset/3D_scans_ds/', aug_num=4, existing_mesh_files=existing_mesh_files)
    trn_list, val_list, file_list = make_filelist_final()
    trn_list = read_filelist('./dataset/FileLists/trn_list.csv')
    val_list = read_filelist('./dataset/FileLists/val_list.csv')
    vtk2h5(cfg, trn_list, val_list, 'upper', parallel=False)
    # h5repack -v -f GZIP=9 ./dataset/3D_scans_h5/upper.hdf5 ./dataset/3D_scans_h5/compressed_upper.hdf5