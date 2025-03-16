import torch
import torch.nn as nn
import numpy as np
import copy
from utils.utils import compute_polar_metric, shuffle_features_across_batch_v1, shuffle_features_across_batch_v2


class shuffle_gate(nn.Module):
    def __init__(self, args, unique_values, features):
        super(shuffle_gate, self).__init__()
        
        self.feature_num = len(features)
        self.features = features
        self.theta = nn.Parameter(torch.ones(self.feature_num, 1).to(args.device))
        
        self.epochs = args.epoch
        self.pretrain_epochs = 0
        self.temp = 5
        self.device = args.device
        self.args = args
        

        self.load_checkpoint = False
        self.optimizer_method = 'darts'

        # self.max_polar =  0
        # self.best_polar_theta = None
    

    def forward(self, x, current_epoch, current_step, raw_data, validation = False):
        if self.mode == 'retrain':
            return x
        
        shuffle_x = shuffle_features_across_batch_v2(x)
        
        g = torch.sigmoid(self.theta * self.temp)
        # g = ffn(g)
        x_ = x * g + (1-g) * shuffle_x.detach()

        # if validation:
        #     with torch.no_grad():

        #         polar_metric = compute_polar_metric(g)
                
        #         if polar_metric > self.max_polar:
        #             self.max_polar = polar_metric
        #             self.best_polar_theta = copy.deepcopy(self.theta)
                
        return x_
    
    
    def fs_loss(self):
        l1_loss = torch.mean(torch.sigmoid(self.theta * self.temp))
        return l1_loss * self.args.fs_weight
    
    def get_fea_importance(self):
        return torch.sigmoid(self.theta * self.temp).reshape(self.feature_num)
    
    def save_selection(self):
        from utils.utils import save_fea_weight
        gate = self.get_fea_importance().detach().cpu().numpy()
        return save_fea_weight(self.features,gate)