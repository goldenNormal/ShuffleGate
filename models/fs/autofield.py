import torch
import torch.nn as nn
import numpy as np

class autofield(nn.Module):

    def __init__(self, args, unique_values, features):
        super(autofield, self).__init__()

        self.feature_num = len(unique_values)
        self.device = args.device
        self.args = args
        self.features = np.array(features)

        self.gate = {features[field_idx]: torch.Tensor(np.ones([1,2])*0.5).to(self.device) for field_idx in range(self.feature_num)}
        self.gate = {features[field_idx]: nn.Parameter(self.gate[features[field_idx]], requires_grad=True) for field_idx in range(self.feature_num)}
        self.gate = nn.ParameterDict(self.gate)
        self.tau = 1.0

        self.mode = 'train'
        self.optimizer_method = 'darts'
        self.update_frequency = args.fs_config[args.fs]['update_frequency']
        self.load_checkpoint = False

    def forward(self, x, current_epoch, current_step, raw_data, validation = False):
        b,f,e = x.shape
        if self.mode == 'retrain':
            return x
        elif self.mode == 'train':
            if self.tau > 0.01:
                self.tau -= 0.00005
        gate_ = torch.ones([1,f,1]).to(self.device)
        for field_idx in range(self.feature_num):
            gate_[:,field_idx,:] = torch.nn.functional.gumbel_softmax(self.gate[self.features[field_idx]], tau=self.tau, hard=False, dim=-1)[:,-1].reshape(1,1,1)
        x = x * gate_
        return x
    
    def get_fea_importance(self):
        gate = torch.concat([self.gate[self.features[field_idx]] for field_idx in range(self.feature_num)], dim=0)[:,-1]
        return gate
    
    def save_selection(self):
        from utils.utils import save_fea_weight
        gate = self.get_fea_importance()
        
        importance = gate.detach().cpu().numpy()
        return save_fea_weight(self.features, importance)
    
        
