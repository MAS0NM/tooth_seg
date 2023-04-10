import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
import utils
import pandas as pd
from mesh_dataset import Mesh_Dataset
from model.imeshsegnet import iMeshSegNet
from losses_and_metrics.losses_and_metrics import *

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
trn_list = './dataset/FileLists/trn_list.csv'
val_list = './dataset/FileLists/val_list.csv'

model_dir = './checkpoints'
model_name = 'latest_checkpoint.pth'

num_classes = 17
num_channels = 15
num_epochs = 300
num_workers = 8
patch_size = 6000
trn_batch_size = 16
val_batch_size = 16
num_batches_to_print = 20
epoch_init = 0
losses = []
mdsc = []
msen = []
mppv = []
val_losses = []
val_mdsc = []
val_msen = []
val_mppv = []
CUDA_LAUNCH_BLOCKING=1

trn_set = Mesh_Dataset(data_list_path=trn_list,
                       num_classes=num_classes,
                       patch_size=patch_size)
val_set = Mesh_Dataset(data_list_path=val_list,
                       num_classes=num_classes,
                       patch_size=patch_size)

trn_loader = DataLoader(dataset=trn_set,
                        batch_size=trn_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        )
val_loader = DataLoader(dataset=val_set,
                        batch_size=val_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        )
torch.backends.cudnn.benchmark = True
model = iMeshSegNet()
opt = optim.Adam(model.parameters(), amsgrad=True)
checkpoint = None
try:
    checkpoint = torch.load(os.path.join(model_dir, model_name), map_location='cpu')
except:
    pass
if checkpoint:
    model.load_state_dict = checkpoint['model_state_dict']
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_init = checkpoint['epoch']
    losses = checkpoint['losses']
    mdsc = checkpoint['mdsc']
    msen = checkpoint['msen']
    mppv = checkpoint['mppv']
    val_losses = checkpoint['val_losses']
    val_mdsc = checkpoint['val_mdsc']
    val_msen = checkpoint['val_msen']
    val_mppv = checkpoint['val_mppv']
    del checkpoint

# start trainning

model.to(device)

class_weights = torch.ones(num_classes).to(device, dtype=torch.float)

if __name__ == '__main__':
    for epoch in range(epoch_init, num_epochs):
        
        print(f'starting epoch: {epoch}')
        model.train()
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        for i_batch, batched_sample in enumerate(trn_loader):
            # print(f'batch: {i_batch}')
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            KG_6 = batched_sample['KG_6'].to(device, dtype=torch.float)
            KG_12 = batched_sample['KG_12'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)
            opt.zero_grad()
                
            # forward + backward + optimize
            outputs = model(inputs, KG_6, KG_12)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()
            
            # print statistics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()
            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(trn_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print))
                running_loss = 0.0
                running_mdsc = 0.0
                running_msen = 0.0
                running_mppv = 0.0
        # record losses and metrics
        losses.append(loss_epoch/len(trn_loader))
        mdsc.append(mdsc_epoch/len(trn_loader))
        msen.append(msen_epoch/len(trn_loader))
        mppv.append(mppv_epoch/len(trn_loader))
        
        #reset
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
            
        # validation
        model.eval()
        with torch.no_grad():
            print('starting validation')
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):

                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                KG_6 = batched_val_sample['KG_6'].to(device, dtype=torch.float)
                KG_12 = batched_val_sample['KG_12'].to(device, dtype=torch.float)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)
                outputs = model(inputs, KG_6, KG_12)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                    print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print))
                    running_val_loss = 0.0
                    running_val_mdsc = 0.0
                    running_val_msen = 0.0
                    running_val_mppv = 0.0

            # record losses and metrics
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            best_val_dsc = max(val_mdsc)
            # reset
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0

            # output current status
            print('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
            

        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_dir+model_name)

        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                        model_dir+'{}_best.tar'.format(model_name))

        # save all losses and metrics data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv('losses_metrics_vs_epoch.csv')