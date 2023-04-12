import os
import sys

sys.path.append(os.getcwd())

import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import *

from dataset.LitDataModule import LitDataModule

from model.LitModule import LitModule


def train(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    pl.seed_everything(cfg.seed)

    datamodule = LitDataModule(cfg)

    datamodule.setup()

    module = LitModule(cfg)

    if cfg.train.pretrain_file is not None and os.path.exists(cfg.train.pretrain_file):
        module.load_model(cfg.train.pretrain_file)
    
    latest_model_to_load = f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f_best_loss"
    if latest_model_to_load is not None and os.path.exists(latest_model_to_load):
        module.load_model(latest_model_to_load)

    dsc_model_checkpoint = ModelCheckpoint(dirpath=cfg.train.checkpoint_dir,
                                    monitor="val_DSC",
                                    mode="max",
                                    filename=f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f_best_DSC",
                                    verbose="True",
                                    every_n_epochs=1,
                                    )
    loss_model_checkpoint = ModelCheckpoint(dirpath=cfg.train.checkpoint_dir,
                                    monitor="val_loss",
                                    mode="min",
                                    filename=f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f_best_loss",
                                    verbose="True",
                                    every_n_epochs=1,
                                    )
    
    trainer = pl.Trainer(callbacks=[dsc_model_checkpoint, loss_model_checkpoint],
                        benchmark=cfg.train.benchmark,
                        deterministic=cfg.train.deterministic,
                        accelerator=cfg.train.accelerator, 
                        devices=cfg.train.devices,
                        # strategy=DDPStrategy(find_unused_parameters=False) if cfg.train.strategy == "DDP" else cfg.train.strategy,
                        max_epochs=cfg.train.epochs,
                        precision=cfg.train.precision,
                        log_every_n_steps=cfg.logger.log_every_n_steps,
                        logger=WandbLogger(project=cfg.logger.project, name=f"{cfg.model.name}-{cfg.train.precision}f") if cfg.logger.use == True else False,
                        # logger=cfg.logger.use, # using default tensorboard
                        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
                        auto_lr_find=cfg.train.auto_lr_find,
                        auto_scale_batch_size=cfg.train.auto_scale_batch_size,
                        fast_dev_run=cfg.train.fast_dev_run)

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule, ckpt_path=os.path.join(cfg.train.checkpoint_dir, f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f.ckpt") if os.path.exists(os.path.join(cfg.train.checkpoint_dir, f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f.ckpt")) else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-cfg", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")

    args = parser.parse_args()
    train(args.config_file)
