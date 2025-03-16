import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import nni
import datetime as dt
from utils.utils import EarlyStopper, compute_polar_metric
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import polars as pl

def get_batch_val(args,val_data,random_perm,i):
    val_x,val_y = val_data
    
    batch_size = args.batch_size
    i = i%(val_data[0].shape[0]//batch_size)
   
    batch_idx = random_perm[i * batch_size:(i+1) * batch_size]
    return val_x[batch_idx], val_y[batch_idx]

class modeltrainer():
    def __init__(self, args, model, model_name, device, epochs, retrain, early_stop=True):
        self.args = args
        self.model = model
        self.optimizers = model.set_optimizer() # dict of optimizers
        self.criterion = torch.nn.BCELoss()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.n_epoch = epochs
        self.batch_size = self.args.batch_size
        self.model_path = 'checkpoints/' + model_name + '_' + args.fs  + '_' + args.dataset + '/'
        if early_stop:
            self.early_stopper = EarlyStopper(patience=args.patience)
        else:
            self.early_stopper = None
        self.retrain = retrain


    def train_one_epoch(self, train_data,  epoch_i, val_data,val_dataloader):
        self.model.train()
        bce_loss_per_epoch = []
        fs_loss_per_epoch = []
        
        train_x, train_y = train_data
        assert train_x.device.type != 'cpu' # in gpu train
        
        polar_metric_per_epoch = []
        
        random_perm = torch.randperm(train_x.shape[0])
        val_random_perm = torch.randperm(val_data[0].shape[0])
        val_iter = iter(val_dataloader)
        for i in tqdm(range(self.args.num_batch)):
            
            batch_idx = random_perm[i*self.batch_size: (i+1)*self.batch_size]
            # print(train_x.shape[0], self.batch_size,max(batch_idx))
            (x, y) = train_x[batch_idx], train_y[batch_idx]
            
            y_pred = self.model(x, current_epoch=epoch_i, current_step=i)
            loss = self.criterion(y_pred, y.float().reshape(-1, 1))
            
            bce_loss_per_epoch.append(loss.item())

             #  fs loss
            if hasattr(self.model.fs, 'fs_loss') and not self.retrain:
                fs_loss = self.model.fs.fs_loss()
                loss += fs_loss
                fs_loss_per_epoch.append(fs_loss.item())
            else:
                fs_loss_per_epoch.append(0)

            
            # polarization_metric
            if hasattr(self.model.fs, "get_fea_importance") and not self.retrain:
                with torch.no_grad():
                    w = self.model.fs.get_fea_importance()
                    polar_metric_per_epoch.append(compute_polar_metric(w))
            else:
                polar_metric_per_epoch.append(0)
                
                

            # optimization parameter
            self.model.zero_grad()
            loss.backward()
            self.optimizers['optimizer_bb'].step()
            
            if not self.retrain:
                if self.args.fs == 'lpfs':
                    p = self.optimizers['optimizer_fs'].param_groups[0]['params'][0]
                    self.optimizers['optimizer_fs'].step()
                    thr = 0.01 * self.args.learning_rate
                    in1 = p.data > thr
                    in2 = p.data < -thr
                    in3 = ~(in1 | in2)
                    p.data[in1] -= thr
                    p.data[in2] += thr
                    p.data[in3] = 0.0
                elif self.args.fs == 'autofield' and i % self.model.fs.update_frequency == 0:
                    self.optimizers['optimizer_fs'].zero_grad()

                    x_,y_ = get_batch_val(self.args, val_data,val_random_perm, i)
                    y_pred_ = self.model(x_, current_epoch=epoch_i, current_step=i)
                    loss_ = self.criterion(y_pred_, y_.float().reshape(-1, 1))
                    loss_.backward()
                    self.optimizers['optimizer_fs'].step()
                
                elif hasattr(self.model.fs, 'fs_loss'):
                    self.optimizers['optimizer_fs'].zero_grad()

                    x_,y_ = get_batch_val(self.args, val_data,val_random_perm, i)
                    
                    y_pred_ = self.model(x_, current_epoch=epoch_i, current_step=i)
                    loss_ = self.criterion(y_pred_, y_.float().reshape(-1, 1))
                    loss_ += self.model.fs.fs_loss()

                    loss_.backward()
                    self.optimizers['optimizer_fs'].step()
            

        return bce_loss_per_epoch, fs_loss_per_epoch, polar_metric_per_epoch
            
    def fit(self, train_data, val_data):

        val_x, val_y = val_data
        val_dataloader = DataLoader(TensorDataset(val_x.cpu(), val_y.cpu()), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        BCE_loss = []
        FS_loss = []
        Polar_metric = []
        VAL_AUC = []

        all_start_time = dt.datetime.now()
        epoch_time_lis = []
        
        for epoch_i in range(self.n_epoch):
            
            epoch_start_time = dt.datetime.now()
            train_process_data = self.train_one_epoch(train_data, epoch_i,val_data,val_dataloader)
            epoch_end_time = dt.datetime.now()
            
            
            bce_loss_per_epoch, fs_loss_per_epoch, polarization_metric_per_epoch = train_process_data

            BCE_loss.extend(bce_loss_per_epoch)
            FS_loss.extend(fs_loss_per_epoch)
            Polar_metric.extend(polarization_metric_per_epoch)

            epoch_time_lis.append((epoch_end_time - epoch_start_time).total_seconds())
            print('epoch:', epoch_i,' train_loss: ', np.mean(bce_loss_per_epoch),' fs_loss :',np.mean(fs_loss_per_epoch))
            
            if self.model.fs is not None and self.args.fs == 'shuffle_gate' and self.model.fs.mode != 'retrain':
                print((torch.sigmoid(self.model.fs.theta * 5)>0.5).sum().item())

            
            if val_data:
            
                
                auc = self.validate_gpu(val_data,epoch_i )
                VAL_AUC.append(auc)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                
                
                if self.early_stopper is not None and self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    
                    break

        all_end_time = dt.datetime.now()
        print('all training time: {} s'.format((all_end_time - all_start_time).total_seconds()))
        print('average epoch time: {} s'.format(sum(epoch_time_lis) / len(epoch_time_lis)))
        
        return pl.DataFrame({'bce_loss':BCE_loss,'fs_loss':FS_loss,'polar_metric':Polar_metric}), VAL_AUC
        


        

    def validate_gpu(self,val_data, epoch_i):
        val_x,val_y = val_data
        assert val_x.device.type !='cpu'
        assert val_y.device.type !='cpu'

        self.model.eval()
        targets, predicts = list(), list()
        val_batch_size = 8192
        num_batches = val_x.shape[0] // val_batch_size
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                batch_i,batch_j = i * val_batch_size, (i+1)*val_batch_size
                x,y = val_x[batch_i:batch_j], val_y[batch_i:batch_j]
                y_pred = self.model(x,epoch_i, i, validation=True) # current_epoch=None means not in training mode
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        # print(self.model.fs.theta)
        auc = roc_auc_score(np.asarray(targets),np.asarray(predicts))
        return auc


    def test_eval(self,test_data):
        test_x,test_y = test_data
        self.model.eval()
        targets, predicts = list(), list()
        test_batch_size = 8192
        num_batches =test_y.shape[0]//test_batch_size
        avg_loss = 0
        criterion = torch.nn.BCELoss()
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                batch_i,batch_j = i * test_batch_size, (i+1)*test_batch_size
                x,y = test_x[batch_i:batch_j], test_y[batch_i:batch_j]
                x = x.to('cuda')
                y = y.to('cuda')
                y_pred = self.model(x,None, i)
            
                bce_loss = criterion(y_pred, y.float().reshape(-1, 1))
                avg_loss += bce_loss.item()
            
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        
        auc = roc_auc_score(np.asarray(targets),np.asarray(predicts))
        avg_loss /= num_batches
        
        return auc, avg_loss

    def save_model(self,path):
        torch.save(self.model.state_dict(), path)


    