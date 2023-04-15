from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5_Mesh_Dataset(Dataset):
    '''
        since h5py object does not support pickle thus it can be mulit-processed
        so when using parallel dataloader, the __getitem__ function has to open and close the hdf5 file every iteration
        thus theres no apperent difference in speed between num_workers == 0 and num_workers == 8
    '''
    def __init__(self,
                cfg,
                data_type: str,
                file_path: str,):
        super(H5_Mesh_Dataset, self).__init__()
        self.cfg = cfg
        self.data_type = data_type
        self.file_path = file_path
        self.hdf5 = None
        self.length = None
        self.dataset = None
        if self.cfg.dataloader.num_workers == 0:
            self.hdf5 = h5py.File(self.file_path, 'r')
            self.length = len(self.hdf5[self.data_type]["labels"])
            self.dataset = self.hdf5[self.data_type]

    def __len__(self):
        if self.cfg.dataloader.num_workers == 0:
            return self.length
        with h5py.File(self.file_path, 'r', libver='latest', swmr=True) as hdf5:
            length = len(hdf5[self.data_type]['labels'])
        return length

    def __getitem__(self, idx: int):
        sample = dict()
        if self.cfg.dataloader.num_workers == 0 and self.dataset:
            dataset = self.dataset
            sample["cells"] = torch.from_numpy(dataset["cells"][idx])
            sample["labels"] = torch.from_numpy(dataset["labels"][idx].astype(np.int64))
            sample["KG_6"] = torch.from_numpy(dataset["KG_6"][idx])
            sample["KG_12"] = torch.from_numpy(dataset["KG_12"][idx])
        else:
            with h5py.File(self.file_path, 'r', libver='latest', swmr=True) as hdf5:
                dataset = hdf5[self.data_type]
                sample["cells"] = torch.from_numpy(dataset["cells"][idx])
                sample["labels"] = torch.from_numpy(dataset["labels"][idx].astype(np.int64))
                sample["KG_6"] = torch.from_numpy(dataset["KG_6"][idx])
                sample["KG_12"] = torch.from_numpy(dataset["KG_12"][idx])
        return sample
    