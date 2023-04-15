from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from os import cpu_count
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from mesh_dataset import Mesh_Dataset
from mesh_dataset_h5 import H5_Mesh_Dataset
import h5py

class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf):
        super(LitDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers if cfg.dataloader.num_workers < cpu_count() else cpu_count()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if self.cfg.train.data_mode == 'vtk':
            trn_data = Mesh_Dataset(data_list_path=self.cfg.dataloader.trn_list,
                        num_classes=self.cfg.model.num_classes,
                        patch_size=self.cfg.model.patch_size,
                        data_mode=self.cfg.train.data_mode)
            val_data = Mesh_Dataset(data_list_path=self.cfg.dataloader.val_list,
                        num_classes=self.cfg.model.num_classes,
                        patch_size=self.cfg.model.patch_size,
                        data_mode=self.cfg.train.data_mode)
            tst_data = Mesh_Dataset(data_list_path=self.cfg.dataloader.val_list,
                        num_classes=self.cfg.model.num_classes,
                        patch_size=self.cfg.model.patch_size,
                        data_mode=self.cfg.train.data_mode)
        elif self.cfg.train.data_mode == 'h5':
            print('initializing h5 dataset')
            trn_data = H5_Mesh_Dataset(self.cfg, 'trn', self.cfg.train.hdf5_path)
            val_data = H5_Mesh_Dataset(self.cfg, 'val', self.cfg.train.hdf5_path)
            tst_data = H5_Mesh_Dataset(self.cfg, 'val', self.cfg.train.hdf5_path)
            
        if stage == "fit" or stage is None:
            self.trn_dataset = trn_data
            self.val_dataset = val_data
        if stage == "test" or stage is None:
            self.tst_dataset = tst_data

    def train_dataloader(self):
        return self._dataloader(self.trn_dataset, train=True, val=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self):
        return self._dataloader(self.tst_dataset)

    def _dataloader(self, dataset, train: bool = False, val: bool = False):
        return DataLoader(dataset,
                        batch_size=self.batch_size,
                        shuffle=True if train and val else False,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        drop_last=True if train and val else False,
                        )
if __name__ == '__main__':
    import torch.nn as nn
    import torch
    from omegaconf import OmegaConf
    import numpy as np
    import h5py
    
    cfg = OmegaConf.load("config/default.yaml")
    hdf5 = h5py.File(cfg.train.hdf5_path, 'r')
    ldm = LitDataModule(cfg)
    ldm.setup()
    ds = ldm
    one_batch_label = ds[0]["labels"].unsqueeze(0)
    print(one_batch_label.shape)
    print(one_batch_label.min())
    print(one_batch_label.max())
    np_one_batch_label=one_batch_label.numpy()
    zero_ind=torch.from_numpy(np.where(np_one_batch_label==0)[-1])
    print(zero_ind)
    one_hot_labels = nn.functional.one_hot(one_batch_label[:, 0, :], num_classes=cfg.model.num_classes)
    print(one_hot_labels[0][zero_ind])
    