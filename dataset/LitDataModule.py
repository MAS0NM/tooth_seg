import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from os import cpu_count
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from mesh_dataset import Mesh_Dataset

class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf):
        super(LitDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers if cfg.dataloader.num_workers < cpu_count() else cpu_count() - 2
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        train_data = Mesh_Dataset(data_list_path=self.cfg.dataloader.trn_list,
                    num_classes=self.cfg.model.num_classes,
                    patch_size=self.cfg.model.patch_size)
        val_data = Mesh_Dataset(data_list_path=self.cfg.dataloader.val_list,
                    num_classes=self.cfg.model.num_classes,
                    patch_size=self.cfg.model.patch_size)
        test_data = Mesh_Dataset(data_list_path=self.cfg.dataloader.val_list,
                    num_classes=self.cfg.model.num_classes,
                    patch_size=self.cfg.model.patch_size)
        if stage == "fit" or stage is None:
            self.train_dataset = train_data
            self.val_dataset = val_data
        if stage == "test" or stage is None:
            self.test_dataset = test_data

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset, train: bool = False, val: bool = False):
        return DataLoader(dataset,
                        batch_size=self.batch_size,
                        shuffle=True if train and val else False,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        drop_last=True if train and val else False,
                        )