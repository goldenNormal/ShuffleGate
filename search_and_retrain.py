import sys
import pandas as pd
import numpy as np
import torch
import os
import argparse
import yaml
import nni
import time
import datetime as dt
from tqdm import tqdm
import utils.utils as utils
from utils.fs_trainer import modeltrainer
from utils.datasets import quick_read_dataset
from models.basemodel import BaseModel
import polars as pl

def search_stage_main(args,seed):
    if args.fs == 'no_selection':
        return
    
    if args.fs in ['gbdt', 'lasso','rf','xgb']:
        feature_importance_csv_path = f'{args.save_path}/ml_feature_importance/{args.fs}-{args.dataset}-{fs_seed}.csv'
    else:
        feature_importance_csv_path = f'{args.save_path}/fea_importance/{args.fs}-{args.dataset}-{args.model}-{seed}.csv'
    
    if os.path.exists(feature_importance_csv_path):
        print('hit... cache..')
        return

    
    
    utils.seed_everything(seed)
    features, _, unique_values, data_df = quick_read_dataset(args.dataset)
    train_x, train_y, val_x, val_y, test_x, test_y = data_df

    import math
    args.num_batch =math.floor (train_x.shape[0] / args.batch_size)
    print(args.num_batch)
    print('-'*10)
    print('features and unique_values:')
    print(features)
    print(unique_values)
    print('-'* 10)

    

    if args.fs in ['gbdt', 'lasso','rf','xgb']: # machine learning feature selection
        ml_start_time = dt.datetime.now()
        fea_df = utils.machine_learning_selection(args, args.fs, features, unique_values, data_df)
        ml_end_time = dt.datetime.now()
        print('machine learning feature selection time: {} s'.format((ml_end_time - ml_start_time).total_seconds()))
        
        fea_df.write_csv(feature_importance_csv_path)

    else:
        # deep method
        model = BaseModel(args, args.model, args.fs, unique_values, features)


        model.fs.mode = 'train'
        trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=False, early_stop=True)
        
        #### prepare data to device to speed up
        train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
        train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
        
        train_data = (train_x.to('cuda'), train_y.to('cuda'))
        val_data = (val_x.to('cuda'),val_y.to('cuda'))
        test_data = (test_x, test_y)
        ####
        s = time.time()
        train_process, Val_AUC = trainer.fit(train_data,val_data)
        consume_time = time.time() - s
        test_auc, test_bce_loss = trainer.test_eval(test_data)
        print(f'test_auc: {test_auc}, test_loss: {test_bce_loss}')

        print('write train process...')


        train_metric_csv_path = f'{args.save_path}/train_metric/search-{args.fs}-{args.dataset}-{args.model}-{seed}.csv'
        pl.DataFrame({'epoch':range(len(Val_AUC)+3),'metric': ['val_auc'] * len(Val_AUC) + ['test_auc','test_bce_loss','search_time'] ,
                       'value':Val_AUC + [test_auc, test_bce_loss,consume_time]}).write_csv(train_metric_csv_path)
        
      
        if hasattr(model.fs, 'save_selection'):
            
            res = model.fs.save_selection()
            if isinstance(res,pl.dataframe.frame.DataFrame):
                fea_df = res
            else:
                
                fea_df = res(val_data, model)

            fea_df.write_csv(feature_importance_csv_path)



def retrain_stage_main(args, seed, fs_seed):
    utils.seed_everything(seed)
    # 1. Get the corresponding feature_importance, and use importance to analyze and filter features...
    
    if args.fs != 'no_selection':
        if fs in ['gbdt','lasso', 'rf', 'xgb']:
            fea_importance_csv_path = f'{args.save_path}/ml_feature_importance/{args.fs}-{args.dataset}-{fs_seed}.csv'    
        else:
            fea_importance_csv_path = f'{args.save_path}/fea_importance/{args.fs}-{args.dataset}-{args.model}-{fs_seed}.csv'
        fea_df = pl.read_csv(fea_importance_csv_path)
   
        top_k = int(fea_df.shape[0] * args.percent)
        assert top_k > 0
        
        selected_features = list(fea_df.sort(pl.col('importance'),descending=True)['fea'][:top_k])
        
        selected_features_str = str(sorted(set(selected_features)))
    else:
        selected_features_str = 'all'
        selected_features = None
    
    print(f'select: {selected_features_str}')
    # 2. Go to record_auc to find out if there is already a corresponding auc. 
    # If so, reuse it. If not, retrain the model and add a new auc data.
    record_path = f'{args.save_path}/retrain_result.csv'
    metric_df = pl.read_csv(record_path)
    
    
    filter_df = metric_df.filter(
        pl.col('dataset') == args.dataset,
        pl.col('selected_features') == selected_features_str,
        pl.col('model') == args.model,
        pl.col('seed') == seed
    )
    match_before = filter_df.shape[0]>0
    if match_before:
        test_auc,test_bce_loss = filter_df['auc'][0], filter_df['bce_loss'][0]
        print(f'match cache... test_auc is {test_auc}, test_bce_loss: {test_bce_loss}')
        
    else:
        
        features, _, unique_values, data_df = quick_read_dataset(args.dataset, selected_features=selected_features)
        train_x, train_y, val_x, val_y, test_x, test_y = data_df

        import math
        args.num_batch =math.floor (train_x.shape[0] / args.batch_size)
        

        print('-'*10)
        print('features and unique_values:')
        print(features)
        print(unique_values)
        print('-'* 10)

        utils.print_time('start retrain...')
        print(args.fs,args.dataset)
        model = BaseModel(args, args.model, 'no_selection', unique_values, features)

        model.fs.mode = 'retrain'
        
        trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=True)
        #### prepare data to device to speed up
        train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
        train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
        
        train_data = (train_x.to('cuda'), train_y.to('cuda'))
        val_data = (val_x.to('cuda'),val_y.to('cuda'))
        test_data = (test_x, test_y)
        ####

        train_process, Val_AUC = trainer.fit(train_data, val_data)
        test_auc, test_bce_loss = trainer.test_eval(test_data)

        print(f'retrain finished...\n test_auc is {test_auc}, test_bce_loss: {test_bce_loss}')

        # 写入 csv 数据
        new_record = pl.DataFrame({
                    'dataset':args.dataset,
                    'selected_features':selected_features_str,
                    'model': args.model,
                    'seed': seed,
                    'auc': test_auc,
                    'bce_loss':test_bce_loss})
        
        pl.concat([metric_df,new_record]).write_csv(record_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='widedeep', help='mlp, widedeep...')
    # parser.add_argument('--dataset', type=str, default='avazu', help='avazu, criteo, movielens-1m, aliccp')  
    # parser.add_argument('--fs', type=str, default='no_selection', help='feature selection methods: \
                        # no_selection, shuffle_gate, autofield, gbdt, lasso, lpfs,  \
                        #  rf, sfs, shark, xgb...')
    
    parser.add_argument('--device',type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu, cuda')
    parser.add_argument('--data_path', type=str, default='quick_data/', help='data path')
    
    parser.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension')
    
    parser.add_argument('--percent', type=float, default=0.5, help='top_percent_features_keeped')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=32, help='num_workers')
    
    args = parser.parse_args()

    with open('models/config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    args.__dict__.update(data)
    
    
    args.timestr = str(time.time())
    fs_methods = [
        'no_selection',
         'shuffle_gate',
                   'autofield', 'sfs', 'shark', 'lpfs', 
                  'gbdt', 'lasso', 'rf', 'xgb']

    data_list = ['movielens-1m','aliccp', 'avazu', 'criteo']

    fs_weight_map = {'movielens-1m':0.1,'aliccp':0.00125,'avazu':0.005,'criteo':0.02}


    model_list = ['widedeep']
    
    args.save_path = './exp_save/'


    for dataset in data_list:
        args.dataset = dataset
        if args.dataset == 'movielens-1m':
            args.batch_size = 256

        else:
            args.batch_size = 4096
        
        args.fs_weight = fs_weight_map[dataset]

        for fs in fs_methods:
            args.fs = fs

            # print args
            for key in args.__dict__:
                if key not in ['fs_config', 'rec_config']:
                    print(key, ':', args.__dict__[key])
                else:
                    print(key, ':')
                    for key2 in args.__dict__[key]:
                        if key2 in [args.model, args.fs]:
                            print('\t', key2, ':', args.__dict__[key][key2])

            for fs_seed in [0,1]:
                args.fs_seed = fs_seed
                if args.fs != 'no_selection':
                    search_stage_main(args, fs_seed)

            for fs_seed in [0,1]:
                args.fs_seed = fs_seed      
                for seed in [0,1,2]:
                    args.seed = seed
                    retrain_stage_main(args,seed, fs_seed )
