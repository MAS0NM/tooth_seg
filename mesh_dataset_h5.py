from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5_Mesh_Dataset(Dataset):
    def __init__(self,
                cfg,
                data_type: str,
                file_path: str,):
        super(H5_Mesh_Dataset, self).__init__()
        self.cfg = cfg
        self.data_type = data_type
        self.file_path = file_path
        self.hdf5 = h5py.File(self.file_path, 'r')
        self.length = len(self.hdf5[self.data_type]["labels"])
        self.dataset = self.hdf5[self.data_type]

    def __len__(self):
        
        return self.length

    def __getitem__(self, idx: int):
        sample = dict()
        sample["cells"] = torch.from_numpy(self.dataset["cells"][idx])
        sample["labels"] = torch.from_numpy(self.dataset["labels"][idx].astype(np.int64))
        sample["KG_6"] = torch.from_numpy(self.dataset["KG_6"][idx])
        sample["KG_12"] = torch.from_numpy(self.dataset["KG_12"][idx])
        return sample
    