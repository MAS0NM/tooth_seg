import vedo
from utils import read_filenames, filelist_checker, visualize_mesh
from omegaconf import OmegaConf
from model.LitModule import LitModule
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
import time
from tqdm import tqdm
import tkinter as tk
import torch
import glob
import vtk


def downsample(mesh, target_cells=10000):
    '''
        downsample the mesh to a certain amount of cells
        to be noticed, the result may not be in the exact number as your input
        if you input 10,000, it could downsample to 9,999
    '''
    mesh_ds = mesh.clone()
    mesh_ds = mesh_ds.decimate(target_cells / mesh.ncells)
    return mesh_ds


def graph_cut(mesh, raw_output, num_classes):
    '''
        mesh must be vedo.mesh
        raw_output refers to the tensor form output from the model
    '''
    raw_output = raw_output.cpu().numpy()
    round_factor = 100
    raw_output[raw_output<1.0e-6] = 1.0e-6

    # unaries
    unaries = -round_factor * np.log10(raw_output)
    unaries = unaries.astype(np.int32)
    unaries = unaries.reshape(-1, num_classes)

    # parawise
    pairwise = (1 - np.eye(num_classes, dtype=np.int32))

    #edges
    cell_ids = np.asarray(mesh.faces())

    lambda_c = 30
    edges = np.empty([1, 3], order='C')
    normals = vedo.vedo2trimesh(mesh).face_normals
    barycenters = mesh.cell_centers()
    # print('start loop')
    for i_node in tqdm(range(mesh.ncells)):
        # Find neighbors
        nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
        nei_id = np.where(nei==2)
        for i_nei in nei_id[0][:]:
            if i_node < i_nei:
                cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                if cos_theta >= 1.0:
                    cos_theta = 0.9999
                theta = np.arccos(cos_theta)
                phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                if theta > np.pi/2.0:
                    edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                else:
                    beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                    edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
    edges = np.delete(edges, 0, 0)
    edges[:, 2] *= lambda_c*round_factor
    edges = edges.astype(np.int32)

    refine_labels = cut_from_graph(edges, unaries, pairwise)
    refine_labels = refine_labels.reshape([-1, 1])

    # output refined result
    mesh3 = mesh.clone()
    mesh3.celldata['labels'] = refine_labels
    return mesh3


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def knn_label(sub_mesh, obj_mesh, sub_label, k=1):
    '''
        use knn to project the labels
        sub_mesh (low resolution) -> obj_mesh (high resolution)
        return a np.array of predicted labels
    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    sub_mesh = np.array(sub_mesh)
    knn.fit(sub_mesh, sub_label)
    y_pred = knn.predict(obj_mesh)
    return y_pred
    
    
def high_resolution_restore(mesh_sbj, mesh_obj, raw_output, refine=False, num_classes=17):
    '''
        use knn to predict the labels on the original scale
    '''
    sbj_labels = mesh_sbj.celldata['labels']
    centers_sbj = mesh_sbj.cell_centers()
    centers_obj = mesh_obj.cell_centers()
    k = 1
    time0 = time.time()
    pred_labels = knn_label(centers_sbj, centers_obj, sbj_labels, k)
    time1 = time.time()
    print(f'knn time: {time1 - time0}')
    mesh_obj.celldata['labels'] = pred_labels
    if refine:
        mesh_obj = graph_cut(mesh_obj, raw_output, num_classes)
        time2 = time.time()
        print(f'graph cut time: {time2 - time1}')
    return mesh_obj


def centering(mesh):
    mesh.points(pts=mesh.points()-mesh.center_of_mass())
    return mesh


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


def gen_metadata_inf(cfg: OmegaConf, mesh: vedo.Mesh, device='cuda', with_new_features=False):
    mesh = centering(mesh)
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

    if with_new_features:
        faces = cells
        face_centers = np.mean(faces.reshape(-1, 3, 3), axis=1)
        face_normals = normals
        corner_vectors = np.hstack((faces[:,0:3] - face_centers,
                                    faces[:,3:6] - face_centers,
                                    faces[:,6:9] - face_centers))
        X = np.column_stack((corner_vectors, face_centers, face_normals))
    
    else:
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
    KG_6 = get_graph_feature(torch.from_numpy(X[9:12, :]).unsqueeze(0), k=6).squeeze(0)
    KG_12 = get_graph_feature(torch.from_numpy(X[9:12, :]).unsqueeze(0), k=12).squeeze(0)
    meta["KG_6"] = KG_6.unsqueeze(0).to(device, dtype=torch.float)
    meta["KG_12"] = KG_12.unsqueeze(0).to(device, dtype=torch.float)

    return meta


def infer(cfg, model, mesh_file, cfg_path=None, ckpt_path=None, refine=True, device='cuda', print_time=False, with_raw_output=False, with_new_features=False):
    if not cfg and cfg_path:
        cfg = OmegaConf.load(cfg_path)
    if len(cfg.infer.devices) == 1 and cfg.infer.accelerator == "gpu":
        device = f"cuda:{cfg.infer.devices[0]}"
    elif len(cfg.infer.devices) > 1 and cfg.infer.accelerator == "gpu":
        device = "cuda:0"
    if not model and ckpt_path:
        module = LitModule(cfg).load_from_checkpoint(ckpt_path)
        model = module.model.to(device)
        model.eval()
    
    start_time = time.time()
    if type(mesh_file) == str:
        mesh = vedo.load(mesh_file)
    else:
        mesh = mesh_file
    
    # transform = vtk.vtkTransform()
    # transform.RotateX(90)
    # matrix = transform.GetMatrix()
    # mesh.apply_transform(matrix)
        
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:,1:]
    cells = points[ids].reshape(N, 9).astype(dtype='float32')
    normals = vedo.vedo2trimesh(mesh).face_normals
    barycenters = mesh.cell_centers()

    mesh_d = mesh.clone()
    predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
    input_data = gen_metadata_inf(cfg, mesh, device, with_new_features)
    load_time = time.time() - start_time

    infer_start_time = time.time()
    with torch.no_grad():
        tensor_prob_output = model(input_data["cells"], input_data["KG_12"],input_data["KG_6"])
    inf_time = time.time() - infer_start_time
    patch_prob_output = tensor_prob_output.cpu().numpy()

    for i_label in range(cfg.model.num_classes):
        predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

    # output downsampled predicted labels
    mesh2 = mesh_d.clone()
    mesh2.celldata['labels'] = predicted_labels_d
    
    if not refine:
        if print_time:
            print(f"mesh loading time:{load_time}\ninference time:{inf_time}\ntotal time:{total_time}")
        if with_raw_output:
            return mesh2, tensor_prob_output
        return mesh2
    else:
        post_start_time = time.time()
        # refinement
        # print('\tRefining by pygco...')
        round_factor = 100
        patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, cfg.model.num_classes)

        # parawise
        pairwise = (1 - np.eye(cfg.model.num_classes, dtype=np.int32))

        #edges
        cell_ids = np.asarray(mesh_d.faces())

        lambda_c = 30
        edges = np.empty([1, 3], order='C')
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei==2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi/2.0:
                        edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                        edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c*round_factor
        edges = edges.astype(np.int32)

        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        # output refined result
        mesh3 = mesh_d.clone()
        mesh3.celldata['labels'] = refine_labels
        post_time = time.time()-post_start_time
        total_time = time.time()-start_time
        if print_time:
            print(f"mesh loading time:{load_time}\ninference time:{inf_time}\npost processing time:{post_time}\ntotal time:{total_time}")
        if with_raw_output:
            return mesh3, tensor_prob_output
        return mesh3
    

def test_inf(mesh_path, cfg, model, with_refine=True, with_new_features=False, with_high_res=True):
    print('start inf')
    time0 = time.time()
    mesh = vedo.Mesh(mesh_path)
    mesh_ds = downsample(mesh) if mesh.ncells > 10001 else mesh
    time1 = time.time()
    print(f'loading and downsample time: {time1 - time0}')
    mesh_ds_pred, pred_tensor = infer(cfg=cfg, model=model, mesh_file=mesh_ds, refine=with_refine, with_raw_output=True, with_new_features=with_new_features)
    time2 = time.time()
    print(f'inference time: {time2 - time1}')
    # visualize_mesh(mesh_ds_pred)
    time3 = time.time()
    mesh_obj = high_resolution_restore(mesh_ds_pred, mesh, pred_tensor) if with_high_res else mesh_ds_pred
    print(f'high resolution restore time: {time3 - time2}')
    print(f'total time: {time3-time0}')
    visualize_mesh(mesh_obj)
    
    
if __name__ == '__main__':
    mode = 'ori'
    with_new_features = True if mode == 'new' else False
    
    # dir_paths = ['./dataset/3D_scans_per_patient_obj_files_b1', './dataset/3D_scans_per_patient_obj_files_b2']
    # label_paths = [dir_path + '/ground-truth_labels_instances_b' + str(idx+1) for idx, dir_path in enumerate(dir_paths)]
    # lower_jaws, upper_jaws = read_filenames(dir_paths)
    # lower_labels, upper_labels = read_filenames(label_paths)
    # lower_jaws, lower_labels = filelist_checker(lower_jaws, lower_labels)
    # upper_jaws, upper_labels = filelist_checker(upper_jaws, upper_labels)
    # # upper_jaws = glob.glob(f"./dataset/test_set_stl/*.stl")
    # # upper_jaws = glob.glob(f"./dataset/NDCS_dataset/a/*.obj")
    samples = glob.glob(f"./dataset/test_set/*.vtk")
    
    print('loading model')
    cfg = OmegaConf.load("config/default.yaml")
    module = LitModule(cfg).load_from_checkpoint(f'./checkpoints/iMeshSegNet_mix_{mode}_17_Classes_32_f_best_DSC.ckpt')
    model = module.model.to('cuda')
    model.eval()
    print('ready')
    # mesh_idx = 0
    # mesh_path = upper_jaws[mesh_idx]
    # mesh_label_path = upper_labels[mesh_idx]
    
    window = tk.Tk()
    window.title(f"mesh viewer {mode}")
    
    window.geometry("400x600")
    
    listbox = tk.Listbox(window)
    listbox.pack(fill=tk.BOTH, expand=1)    
    
    for name in samples:
        listbox.insert(tk.END, name.split('/')[-1])

    listbox.bind("<Double-Button-1>", lambda x:\
        test_inf(samples[listbox.curselection()[0]], cfg, model, with_refine=True, with_new_features=with_new_features, with_high_res=False))
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    window.mainloop()