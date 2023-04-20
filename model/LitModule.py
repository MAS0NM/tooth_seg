import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import StepLR
from losses_and_metrics.losses_and_metrics import *
from losses_and_metrics.lovasz_losses import *
from losses_and_metrics import Lovasz_Softmax_Flat
from .imeshsegnet import iMeshSegNet
from collections import OrderedDict

class LitModule(pl.LightningModule):
    def __init__(self,cfg: OmegaConf):
        super(LitModule, self).__init__()
        self.cfg = cfg
        self.cfg_train = cfg.train
        self.cfg_model = cfg.model
        
        self.model = iMeshSegNet(num_classes=self.cfg_model.num_classes, 
                                num_channels=self.cfg_model.num_channels, 
                                with_dropout=self.cfg_model.with_dropout, 
                                dropout_p=self.cfg_model.dropout_p)
        self.lovasz_fn = Lovasz_Softmax_Flat()
        self.ce_fn = nn.CrossEntropyLoss()
    
        self.loss_fn = Generalized_Dice_Loss()
        self.hparams.learning_rate = cfg.train.learning_rate
        self.save_hyperparameters()
        
    def forward(self, X):
        outputs = self.model(X['cells'], X['KG_12'], X['KG_6'])
        return outputs
    
    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = create_optimizer_v2(self.parameters(),
                                        opt=self.cfg_train.optimizer,
                                        lr=self.cfg_train.learning_rate,
                                        weight_decay=self.cfg_train.weight_decay,
                                        )
        scheduler = StepLR(optimizer,
                        step_size=self.cfg.train.step_size,
                        gamma=self.cfg.train.gamma,
                        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        
    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k[6:] # remove the 'model.' in ckpt['state_dict']['model.*']
            new_state_dict[name] = v
        epoch, global_step = ckpt['epoch'], ckpt['global_step']
        print(f'loading checkpoint with epoch:{epoch} and global step: {global_step}')
        self.model.load_state_dict(new_state_dict, strict=True)
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
    
    def _step(self, batch, step):
        outputs = self(batch)
        
        class_weights = torch.ones(self.cfg_model.num_classes, device=self.device)
        one_hot_labels = nn.functional.one_hot(batch['labels'].long()[:, 0, :], num_classes=self.cfg_model.num_classes)
        
        gum_weight = 0.9
        train_weights = class_weights.clone()
        train_weights /= (1 - gum_weight) / self.cfg_model.num_classes
        train_weights[0] = gum_weight
        dice_loss = self.loss_fn(outputs, one_hot_labels, train_weights)
        lovasz_loss = self.lovasz_fn(outputs.view(-1, self.cfg_model.num_classes).unsqueeze(-1).unsqueeze(-1), batch["labels"].view(-1).long())
        ce_loss = self.ce_fn(outputs.view(-1, self.cfg_model.num_classes), batch["labels"].view(-1).long())   
        loss = dice_loss * 0.45 + lovasz_loss * 0.45 + ce_loss * 0.1

        self.log(f"{step}_loss", loss, sync_dist=True)
        
        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_DSC", dsc, sync_dist=True, prog_bar=True)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_SEN", sen, sync_dist=True, prog_bar=True)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_PPV", ppv, sync_dist=True, prog_bar=True)
        
        return loss