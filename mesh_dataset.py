from torch.utils.data import Dataset
import pandas as pd
import torch
from utils import get_graph_feature
import numpy as np
try:
    import vtk
    import vedo
except:
    print('cannot import vtk or vedo')

def vtk2np(pointdata):
    length = pointdata.GetNumberOfTuples()
    dim = pointdata.GetNumberOfComponents()
    numpy_array = np.empty((length, dim))
    for i in range(length):
        numpy_array[i] = pointdata.GetTuple(i)
    return numpy_array


def compute_normals(mesh):
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SetComputeCellNormals(True)
    normals.Update()
    
    cell_normals = vtk2np(normals.GetOutput().GetCellData().GetNormals())

    return cell_normals


def get_cell_centers(mesh):
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(mesh)
    cell_centers.Update()

    cell_centers_polydata = cell_centers.GetOutput()
    cell_centers_polydata =vtk2np(cell_centers_polydata.GetPoints().GetData())
    return cell_centers_polydata



def gen_metadata(mesh, patch_size, mode='vedo', is_new=False):
    '''
        to form a N x 15 vector
        input mesh form should be vedo.mesh.object
        which includes attributes: mesh.celldata['labels']
    '''
    if mode == 'vtk':
        N = mesh.GetNumberOfCells()
        points = vtk2np(mesh.GetPoints().GetData())
        # get cells' points indices
        ids = vtk2np(mesh.GetPolys().GetData()).astype(dtype='int32').reshape((N, -1))[:,1:]
        # get the points in coordinates
        cells = points[ids].reshape(N, 9).astype(dtype='float32')
        labels = vtk2np(mesh.GetCellData().GetArray("labels")).astype('int32').reshape(-1, 1)
        normals = compute_normals(mesh)
        # barycenters = get_cell_centers(mesh)
    elif mode == 'vedo':
        N = mesh.ncells
        points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
        ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:,1:]
        cells = points[ids].reshape(N, 9).astype(dtype='float32')
        labels = mesh.celldata["labels"].astype('int32').reshape(-1, 1)
        mesh.compute_normals()
        normals = mesh.celldata['Normals']
        barycenters = mesh.cell_centers()
    

    # form the vectors
    if not is_new:
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
        Y = labels
    
    elif is_new:
        faces = cells
        face_centers = np.mean(faces.reshape(-1, 3, 3), axis=1)
        face_normals = normals
        corner_vectors = np.hstack((faces[:,0:3] - face_centers,
                                    faces[:,3:6] - face_centers,
                                    faces[:,6:9] - face_centers))
        X = np.column_stack((corner_vectors, face_centers, face_normals))
        Y = labels
        
    # initialize batch of input and label
    X_train = np.zeros([patch_size, X.shape[1]], dtype='float32')
    Y_train = np.zeros([patch_size, Y.shape[1]], dtype='int32')

    # calculate number of valid cells (tooth instead of gingiva)
    positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
    negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

    num_positive = len(positive_idx) # number of selected tooth cells

    if num_positive > patch_size: # all positive_idx in this patch
        positive_selected_idx = np.random.choice(positive_idx, size=patch_size, replace=False)
        selected_idx = positive_selected_idx
    else:   # patch contains all positive_idx and some negative_idx
        num_negative = patch_size - num_positive # number of selected gingiva cells
        positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
        negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
        selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

    selected_idx = np.sort(selected_idx, axis=None)

    X_train[:] = X[selected_idx, :]
    Y_train[:] = Y[selected_idx, :]

    X_train = X_train.transpose(1, 0)
    Y_train = Y_train.transpose(1, 0)
    
    KG_6 = get_graph_feature(torch.from_numpy(X_train[9:12, :]).unsqueeze(0), k=6).squeeze(0).numpy()
    KG_12 = get_graph_feature(torch.from_numpy(X_train[9:12, :]).unsqueeze(0), k=12).squeeze(0).numpy()

    metadata = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'KG_6': KG_6, 'KG_12': KG_12,}
    
    return metadata


# this is not used if train with h5 file
class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=6000):
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_mesh = self.data_list.iloc[idx][0]
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(i_mesh)
        reader.Update()
        mesh = reader.GetOutput()
        sample = gen_metadata(mesh, self.patch_size)
        return sample

# if __name__ == '__main__':
#     path = './dataset/FileLists/fileList_lower.txt'
#     dataset = Mesh_Dataset('./train_list_1.csv')
#     print(dataset.__getitem__(0))
