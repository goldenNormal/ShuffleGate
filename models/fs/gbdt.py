import torch
import torch.nn as nn
import torch.nn.functional as F
# ML
class gbdt(nn.Module):
    def __init__(self, args, unique_values, features):
        super(gbdt, self).__init__()

        # 必需的参数
        self.load_checkpoint = False
        self.optimizer_method = 'normal'

    def forward(self, x, current_epoch, current_step, raw_data,validation = False):
        return x