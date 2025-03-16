import torch
import torch.nn as nn
import numpy as np
import tqdm
import numpy as np

class sfs(nn.Module):

    def __init__(self, args, unique_values, features):
        super(sfs, self).__init__()
        self.load_checkpoint = False
        self.optimizer_method = 'normal'

        self.feature_num = len(unique_values)
        self.device = args.device
        self.args = args
        self.criterion = torch.nn.BCELoss()
        self.features = np.array(features)    

        opt = args.fs_config[args.fs]
        #self.cr = opt['cr']
        self.num_batch_sampling = opt['num_batch_sampling']

        self.mode = 'train'
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))
        print(self.offsets)
        print(self.feature_num)
        self.mask = nn.Parameter(torch.ones([self.feature_num,1]))
        self.mask.requires_grad = False
    
    def forward(self, x, current_epoch, current_step, raw_data, validation = False):
        return x*self.mask
    
    def save_selection(self):
        from utils.utils import save_fea_weight
        def prun(data,model):
            model.fs.mask.requires_grad = True
            val_x,val_y = data
            random_perm = torch.randperm(val_x.shape[0])
            batch_size = 8196
            num_batch = val_x.shape[0]//batch_size
            for i in tqdm.tqdm(range(num_batch)):    
                if i == model.fs.num_batch_sampling:
                    break
                batch_idx = random_perm[i*batch_size: (i+1)*batch_size]
                (c_data, labels) = val_x[batch_idx], val_y[batch_idx]

                out = model(c_data,0,i)
                loss =self.criterion(out, labels.float().unsqueeze(-1))
                model.zero_grad()
                loss.backward()            
                grads = torch.abs(model.fs.mask.grad)
                if i == 0:
                    moving_average_grad = grads
                else:
                    moving_average_grad =  ((moving_average_grad * i) + grads) / (i + 1)
            grads = torch.flatten(moving_average_grad)
            importance = (grads / grads.sum()).detach().cpu().numpy()
            importance = importance.reshape(self.feature_num)
            return save_fea_weight(self.features, importance)
        return prun
       

    


