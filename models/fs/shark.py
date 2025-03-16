import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

class shark(nn.Module):
    def __init__(self, args, unique_values, features):
        super(shark, self).__init__()
        self.feature_num = len(unique_values)
        self.features = np.array(features)
        # 必需的参数
        self.load_checkpoint = False
        self.optimizer_method = 'normal'
        self.criterion = torch.nn.BCELoss()
        self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))

    def forward(self, x, current_epoch, current_step, raw_data, validation = False):
        return x
    
    def save_selection(self):
        from utils.utils import save_fea_weight
        def selection(data, model):
            # tk0 = tqdm.tqdm(test_dataloader, desc="f-permutation", smoothing=0, mininterval=1.0)
            
            # model = model.to(device)
            num = 0
            # importance = torch.zeros(len(model.offsets)).to(device) # save importance for each field
            importance = np.zeros(len(model.offsets))
            val_x,val_y = data
            # print(val_x.shape)
            expectation = torch.zeros((len(model.offsets))).to(val_x.device)
            random_perm = torch.randperm(val_x.shape[0])
            batch_size = 8196
            num_batch = val_x.shape[0]//batch_size
            for i in tqdm.tqdm(range(num_batch)):
                batch_idx = random_perm[i*batch_size: (i+1)*batch_size]
                x,y = val_x[batch_idx], val_y[batch_idx]
                embs = model.embedding(x + x.new_tensor(self.offsets))
                if len(expectation.shape) == 1:
                    expectation = torch.zeros((len(model.offsets), embs.shape[2])).to(x.device)
                expectation += torch.sum(embs, dim=0)
                num += x.shape[0]
            expectation = expectation / num
            expectation = expectation.reshape(1, len(model.offsets), -1)
            # expectation = torch.zeros((1, len(model.offsets), 8)).to(device)
            num = 0
            # new_dataloader = torch.utils.data.DataLoader(test_dataloader.dataset, batch_size=1, num_workers=16)
            # tk0 = tqdm.tqdm(new_dataloader, desc="f-permutation", smoothing=0, mininterval=1.0)
            for i in tqdm.tqdm(range(val_x.shape[0])):
                x,y = val_x[i:i+1],val_y[i:i+1]
                
                model.zero_grad()
                embs = model.embedding(x + x.new_tensor(self.offsets))
                # expectation = torch.mean(embs, dim=0)
                expectation_resize = expectation.repeat(x.shape[0], 1,1)
                right_part = expectation_resize - embs
                y_pred = model(x, current_epoch=None, current_step=i)
                loss = self.criterion(y_pred, y.float().reshape(-1, 1))
                # cal gradient for each embedding
                loss.backward()
                # get gradient
                gradients = F.embedding(x + x.new_tensor(self.offsets),model.embedding.weight.grad).to(val_x.device)
                # use the torch.gradient
                # cal importance
                error = gradients * right_part # b,f,e
                error = torch.sum(error, dim=2) # b,f
                error = torch.sum(abs(error), dim=0) # f
                importance += error.detach().cpu().numpy()
                num += x.shape[0]
            importance = importance / num
            importance = importance.reshape(self.feature_num)
            return save_fea_weight(self.features, importance)
        return selection