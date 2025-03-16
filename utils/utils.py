import random
import numpy as np
import torch
import os
import copy
import importlib
import datetime
import argparse
import polars as pl

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name: str, model_type: str):
    """
    Automatically select model class based on model name

    Args:
        model_name (str): model name
        model_type (str): rec, fs, es

    Returns:
        Recommender: model class
        Dict: model configuration dict
    """
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['models', model_type, model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    else:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    return model_class

@torch.no_grad()
def compute_polar_metric(w):
    return (torch.mean(torch.abs(w - torch.mean(w)))).item()

class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
        
    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True
        
        
def save_fea_weight(features, field_importance):
    return pl.DataFrame({'fea':features,'importance':field_importance}).sort('importance',descending=True)

def machine_learning_selection(args, fs, features, unique_values, data):
    train_x, train_y, val_x, val_y, test_x, test_y = data
    features = np.array(features)
    if fs == 'lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso(
            alpha=args.fs_config[args.fs]['alpha'],
            fit_intercept=args.fs_config[args.fs]['fit_intercept'],
            copy_X=args.fs_config[args.fs]['copy_X'],
            max_iter=args.fs_config[args.fs]['max_iter'],
            tol=args.fs_config[args.fs]['tol'],
            positive=args.fs_config[args.fs]['positive'],
            selection=args.fs_config[args.fs]['selection']
        )
        lasso.fit(train_x, train_y)
        field_importance = abs(lasso.coef_)
        return save_fea_weight(features, field_importance)
    elif fs == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier
        gbdt = GradientBoostingClassifier(
            learning_rate=args.fs_config[args.fs]['learning_rate'],
            n_estimators=args.fs_config[args.fs]['n_estimators'],
            subsample=args.fs_config[args.fs]['subsample'],
            min_samples_split=args.fs_config[args.fs]['min_samples_split'],
            min_samples_leaf=args.fs_config[args.fs]['min_samples_leaf'],
            min_weight_fraction_leaf=args.fs_config[args.fs]['min_weight_fraction_leaf'],
            max_depth=args.fs_config[args.fs]['max_depth'],
            n_iter_no_change=args.fs_config[args.fs]['n_iter_no_change'],
            verbose=1
        )
        gbdt.fit(train_x, train_y)
        field_importance = gbdt.feature_importances_
        return save_fea_weight(features, field_importance)

    elif fs == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, max_depth=None, n_jobs=6, verbose=1).fit(train_x, train_y)
        field_importance = model.feature_importances_
        return save_fea_weight(features, field_importance)
    elif fs == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=10, max_depth=None, n_jobs=6, verbose=1).fit(train_x, train_y)
        field_importance = model.feature_importances_
        return save_fea_weight(features, field_importance)

    
def print_time(message):
    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S '), message)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def shuffle_features_across_batch_v2(x):
    """
    对输入的 tensor 进行特征 shuffle，每个特征在 batch 维度上进行独立的 shuffle。
    
    参数:
        x (torch.Tensor): 输入的 tensor，形状为 (b, f, e)，其中 b 是 batch size，f 是特征数量，e 是 embedding size。
    
    返回:
        torch.Tensor: shuffle 后的 tensor，形状与输入相同。
    """
    b, f, e = x.shape
    
    # 生成随机排列的索引
    rand_values = torch.rand(f, b,device=x.device)  # 形状: (f, b)
    indices = rand_values.argsort(dim=1)  # 形状: (f, b)
    
    # 扩展索引以匹配 x 的形状
    indices = indices.unsqueeze(-1).expand(-1, -1, e)  # 形状: (f, b, e)
    
    # 使用 torch.gather 进行 shuffle
    shuffled_x = torch.gather(x.permute(1, 0, 2), dim=1, index=indices)  # 沿着 batch 维度 shuffle
    shuffled_x = shuffled_x.permute(1, 0, 2)  # 恢复原始形状 (b, f, e)
    
    return shuffled_x

def shuffle_features_across_batch_v1(x):
    """
    对输入的 tensor 进行特征 shuffle，每个特征在 batch 维度上进行独立的 shuffle。
    
    参数:
        x (torch.Tensor): 输入的 tensor，形状为 (b, f, e)，其中 b 是 batch size，f 是特征数量，e 是 embedding size。
    
    返回:
        torch.Tensor: shuffle 后的 tensor，形状与输入相同。
    """
    b, f, e = x.shape
    
    # 为每个特征生成一个独立的随机排列索引
    indices = torch.stack([torch.randperm(b,device=x.device) for _ in range(f)], dim=0)  # 形状: (f, b)
    
    # 扩展索引以匹配 x 的形状
    indices = indices.unsqueeze(-1).expand(-1, -1, e)  # 形状: (f, b, e)
    
    # 使用 torch.gather 进行 shuffle
    shuffled_x = torch.gather(x.permute(1, 0, 2), dim=1, index=indices)  # 沿着 batch 维度 shuffle
    shuffled_x = shuffled_x.permute(1, 0, 2)  # 恢复原始形状 (b, f, e)
    
    return shuffled_x


def load_warmup_parameter(model,args):
    
    model_state_dict = model.state_dict()
    
    old_model_state_dict=torch.load(f'{args.save_dir}/warmup_ckpt/{args.dataset}-{args.model}-0.pth')
    
    # 遍历模型 B 的 state_dict
    for name, param in model_state_dict.items():
        # 如果模型 A 中有同名的参数
        if name in old_model_state_dict:
            # 检查形状是否匹配
            if param.shape == old_model_state_dict[name].shape:
                # 加载模型 A 的参数到模型 B
                model_state_dict[name] = old_model_state_dict[name]
            else:
                print(f"参数 {name} 形状不匹配，跳过加载")
        else:
            print(f"参数 {name} 在模型 A 中不存在，跳过加载")

    # 将更新后的 state_dict 加载到模型 B
    model.load_state_dict(model_state_dict, strict=False)
    
    return model