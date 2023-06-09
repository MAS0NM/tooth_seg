import numpy as np
import os        
import vedo
import json
import glob
import random
import h5py
import csv
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
    
    pairs = [(17, 37, 1), (16, 36, 2), (15, 35, 3), (14, 34, 4), (13, 33, 5), (12, 32, 6), (11, 31, 7),\
        (21, 41, 8), (22, 42, 9), (23, 43, 10), (24, 44, 11), (25, 45, 12), (26, 46, 13), (27, 47, 14),\
            (18, 38, 15), (28, 48, 16)]
    
    for x, y, z in pairs:
        index_x = np.where(nparry == x)
        index_y = np.where(nparry == y)
        np.put(nparry, index_x, z)
        np.put(nparry, index_y, z)
    
    return nparry


def flip_relabel(nparry):
    # 1 14, 2 13, 3 12, 4 11, 5 10, 6 9, 7 8, 15 16
    pairs = [(1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (15, 16)]
    for x, y in pairs:
        index_x = np.where(nparry == x)
        index_y = np.where(nparry == y)
        np.put(nparry, index_x, y)
        np.put(nparry, index_y, x)
        
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
    mesh = centring(mesh)
            
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
                                scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2],
                                isRotate=True, isTranslate=True, isScale=True):
    # create a vtk transform object
    transform = vtk.vtkTransform()
    # set random rotation angles
    if isRotate:
        if type(rotate_X) == list and type(rotate_Y) == list and type(rotate_Z) == list:
            angle_x = np.random.uniform(rotate_X[0], rotate_X[1])
            angle_y = np.random.uniform(rotate_Y[0], rotate_Y[1])
            angle_z = np.random.uniform(rotate_Z[0], rotate_Z[1])
            transform.RotateX(angle_x)
            transform.RotateY(angle_y)
            transform.RotateZ(angle_z)
        elif type(rotate_X) == int and type(rotate_Y) == int and type(rotate_Z) == int:
            transform.RotateX(rotate_X)
            transform.RotateY(rotate_Y)
            transform.RotateZ(rotate_Z)
            
    # set random translation distances
    if isTranslate:
        dist_x = np.random.uniform(translate_X[0], translate_X[1])
        dist_y = np.random.uniform(translate_Y[0], translate_Y[1])
        dist_z = np.random.uniform(translate_Z[0], translate_Z[1])
        transform.Translate(dist_x, dist_y, dist_z)
    # set random scaling factors
    if isScale:
        factor_x = np.random.uniform(scale_X[0], scale_X[1])
        factor_y = np.random.uniform(scale_Y[0], scale_Y[1])
        factor_z = np.random.uniform(scale_Z[0], scale_Z[1])
        transform.Scale(factor_x, factor_y, factor_z)
    # get the transformation matrix
    matrix = transform.GetMatrix()
    return matrix
        
        
def augment(mesh_path, out_dir, mode, aug_num, existing_mesh_files, isRotate, isTranslate, isScale):
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
        mesh.celldata['labels'] = flip_relabel(mesh.celldata['labels'])
        mesh = centring(mesh)
        mesh.write(mesh_op_pth)
    else:
        for i in range(aug_num):
            transform_type = ''
            if isRotate:
                transform_type += 'R'
            if isTranslate:
                transform_type += 'T'
            if isScale:
                transform_type += 'S'
            mesh_op_pth = os.path.join(out_dir, mesh_name+'_'+transform_type+"_AUG%02d_" %i + '.vtk')
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                    translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                    scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2],
                                                    isRotate=isRotate, isTranslate=isTranslate, isScale=isScale) #use default random setting

            # rotate along the 3 axis seperately
            # rotate_angle = 30
            # for j in range(3):
            #     if j == 0:
            #         vtk_matrix = GetVTKTransformationMatrix(rotate_X=-180+rotate_angle*i, rotate_Y=0, rotate_Z=0)
            #         mesh_op_pth = os.path.join(out_dir, mesh_name+'_'+transform_type+"_AUG%02d_" %i + '_x' + '.vtk')
            #     elif j == 1:
            #         vtk_matrix = GetVTKTransformationMatrix(rotate_X=0, rotate_Y=-180+rotate_angle*i, rotate_Z=0)
            #         mesh_op_pth = os.path.join(out_dir, mesh_name+'_'+transform_type+"_AUG%02d_" %i + '_y' + '.vtk')
            #     elif j == 2:
            #         vtk_matrix = GetVTKTransformationMatrix(rotate_X=0, rotate_Y=0, rotate_Z=-180+rotate_angle*i)
            #         mesh_op_pth = os.path.join(out_dir, mesh_name+'_'+transform_type+"_AUG%02d_" %i + '_z' + '.vtk')
                    
            mesh = vedo.load(mesh_path)
            mesh.apply_transform(vtk_matrix)
            mesh = centring(mesh)
            mesh.celldata['Normal'] = vedo.vedo2trimesh(mesh).face_normals
            mesh.write(mesh_op_pth)
            
        
def do_augmentation(ip_dir, op_dir, aug_num, existing_mesh_files):
    '''
        this function will first look into the ip_dir to make a filelist according to the files in there
        then preform augmentation on each .vtk file
    '''
    # filp the meshes to double the size
    filels = glob.glob(f"{ip_dir}/*.vtk")
    print(f'get {len(filels)} item to augment with the number of {aug_num}')
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'flip', aug_num, existing_mesh_files, False, False, False) for item in tqdm(filels, desc="flipping"))
    # random tranform
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'trsf', aug_num, existing_mesh_files, True, True, True) for item in tqdm(filels, desc="transforming"))
    
    
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
    return trn_list, val_list, filelist
            
            
def make_filelist_final():
    filelist = []
    for file in glob.glob(r'./dataset/3D_scans_ds/*.vtk'):
        file = file.replace('\\', '/')
        filelist.append(file)
    with open('./dataset/FileLists/filelist_final.csv', 'w', newline='') as f:
        for line in filelist:
            f.write(line+'\n')
    return make_train_val_list()


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
        

def vtk2h5(cfg, jaw_type, parallel = False, is_new=False):
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
                meta_data = gen_metadata(mesh, cfg.model.patch_size, mode='vedo', is_new=is_new)
                f[item]["labels"][idx] = meta_data['labels']
                f[item]["cells"][idx] = meta_data['cells']
                f[item]["KG_6"][idx] = meta_data['KG_6']
                f[item]["KG_12"][idx] = meta_data['KG_12']
    f.close()


def make_labeled_mesh(jaw_path, lab_path, op_dir):
    sampleName = get_sample_name(jaw_path)
    path_mesh = op_dir + sampleName + '.vtk'
    
    mesh = vedo.load(jaw_path)
    print(mesh.ncells)
            
    # Add cell data with labels
    with open(lab_path, 'r') as f:
        labels_by_point = np.array(json.load(f)['labels'], dtype=np.uint8)
        labels_by_point = rearrange(labels_by_point)
    # raw mesh
    labels_by_face = set_face_label(mesh, labels_by_point, mode='count')
    mesh.pointdata['labels'] = labels_by_point
    mesh.celldata['labels'] = labels_by_face
        
    # write mesh vtk
    mesh.write(path_mesh)


def vtk2stl(file_list, des_dir):
    for file in file_list:
        mesh = vedo.Mesh(file)
        name = os.path.splitext(os.path.basename(file))[0]
        des_file = os.path.join(des_dir, name + '.stl')
        mesh.write(des_file)
    
    
if __name__ == '__main__':
    cfg = OmegaConf.load("config/default.yaml")
    target_cells = 10001
    dataset_num = 200
    # existing_mesh_files = read_dir(dir_path='./dataset/3D_scans_ds/', extension='vtk', constrain='')
    # do_downsample(upper_jaws[:dataset_num], upper_labels[:dataset_num], target_cells=target_cells, filelist='./dataset/FileLists/fileList_upper_ds.txt', ds_dir = './dataset/3D_scans_ds/')
    # print(f"upper batch is done")
    # do_downsample(lower_jaws[:dataset_num], lower_labels[:dataset_num], target_cells=target_cells, filelist='./dataset/FileLists/fileList_upper_ds.txt', ds_dir = './dataset/3D_scans_ds/')
    # print(f"lower batch is done")
    # do_augmentation(ip_dir='./dataset/3D_scans_ds/', op_dir='./dataset/3D_scans_ds/', aug_num=4, existing_mesh_files=existing_mesh_files)
    # trn_list, val_list, file_list = make_filelist_final()
    trn_list = read_filelist('./dataset/FileLists/trn_list.csv')
    val_list = read_filelist('./dataset/FileLists/val_list.csv')
    print(f'training set size: {len(trn_list)}\nvalidation set size: {len(val_list)}')
    # vtk2stl(existing_mesh_files, des_dir='./dataset/3D_scans_stl')
    # vtk2h5(cfg, 'mix_ori', parallel=False)
    
        
    # below is the command line to compress the h5 file
    # h5repack -v -f GZIP=9 ./dataset/3D_scans_h5/upper.hdf5 ./dataset/3D_scans_h5/compressed_upper.hdf5
    
    
    test_set = upper_jaws+lower_jaws
    test_set_labels = upper_labels+lower_labels
    with open('./dataset/FileLists/tst_list.csv', 'w', newline='') as f:
        trn_list_lookup = [get_sample_name(i) for i in trn_list]
        val_list_lookup = [get_sample_name(i) for i in val_list]
        lookup_list = trn_list_lookup + val_list_lookup
        writer = csv.writer(f)
        for idx, line in enumerate(test_set):
            if get_sample_name(line) not in lookup_list:
                writer.writerow([line, test_set_labels[idx]])