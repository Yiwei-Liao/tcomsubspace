#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Section 1: Imports & Global Settings
# =============================================================================
import os
import os.path as osp
import time
import argparse
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports
from layers import *
from models import *
from preprocessing import *
from convert_datasets_to_pygDataset import dataset_Hypergraph

# Fix Random Seeds
np.random.seed(0)
torch.manual_seed(0)


# =============================================================================
# Section 2: Utility Functions (Noise, Evaluation, Helper)
# =============================================================================

def str2bool(v):
    """用于Argparse的布尔值解析工具"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_parameters(model):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_noise(data, noise_type, snr):
    """
    向 PyTorch 张量添加高斯噪声或瑞利噪声，并根据信噪比控制噪声强度。
    """
    data_power = torch.mean(data ** 2)
    snr_linear = 10 ** (snr / 10)
    noise_power = data_power / snr_linear
    
    if noise_type == "gaussian":
        noise = torch.randn_like(data) * torch.sqrt(noise_power)
    elif noise_type == "rayleigh":
        sigma = torch.sqrt(noise_power / 2)
        noise = sigma * torch.sqrt(torch.rand_like(data) ** 2 + torch.rand_like(data) ** 2)
    else:
        raise ValueError("Unsupported noise type. Choose 'gaussian' or 'rayleigh'.")
    
    data_with_noise = data + noise
    return data_with_noise

def eval_acc(y_true, y_pred, name):
    """计算准确率"""
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    
    if len(correct) == 0:
        acc_list.append(0.0)
    else:    
        acc_list.append(float(np.sum(correct)) / len(correct))
    
    return sum(acc_list) / len(acc_list)

@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    """常规评估函数"""
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx['train']], out[split_idx['train']], name='train')
    valid_acc = eval_func(data.y[split_idx['valid']], out[split_idx['valid']], name='valid')
    test_acc  = eval_func(data.y[split_idx['test']],  out[split_idx['test']],  name='test')

    train_loss = F.nll_loss(out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss  = F.nll_loss(out[split_idx['test']],  data.y[split_idx['test']])
    
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out

@torch.no_grad()
def evaluate_noise(model, data, split_idx, eval_func, snr, noise_type, result=None):
    """带噪声的测试集评估函数"""
    model.eval()
    m_data = data.clone()
    m_data.x = add_noise(m_data.x, noise_type, snr)
    out = model(m_data)
    out = F.log_softmax(out, dim=1)

    test_acc = eval_func(m_data.y[split_idx['test']], out[split_idx['test']], name='test')
    return test_acc


# =============================================================================
# Section 3: Model Parsing & Logger
# =============================================================================

def parse_method(args, data):
    """根据args.method初始化对应的模型"""
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features, hid_dim=args.MLP_hidden, out_dim=args.num_classes,
                      num_layers=args.All_num_layers, dropout=args.dropout, Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features, hid_dim=args.MLP_hidden, out_dim=args.num_classes,
                      num_layers=args.All_num_layers, heads=args.heads, output_heads=args.output_heads,
                      dropout=args.dropout, Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0], E=He_dict, X=data.x,
                         num_features=args.num_features, num_layers=args.All_num_layers,
                         num_classses=args.num_classes, args=args)
                         
    elif args.method in ['sheafHyperGCNDiag', 'sheafHyperGCNOrtho', 'sheafHyperGCNGeneral', 'sheafHyperGCNLowRank']:
        sheaf_type_map = {
            'sheafHyperGCNDiag': 'Diagsheafs',
            'sheafHyperGCNOrtho': 'Orthosheafs',
            'sheafHyperGCNGeneral': 'Generalsheafs',
            'sheafHyperGCNLowRank': 'LowRanksheafs'
        }
        He_dict = get_HyperGCN_He_dict(data) 
        model = sheafHyperGCN(V=data.x.shape[0], num_features=args.num_features,
                              num_layers=args.All_num_layers, num_classses=args.num_classes,
                              args=args, sheaf_type=sheaf_type_map[args.method])

    elif args.method == 'HGNN':
        args.use_attention = False
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)

    elif args.method in ['sheafHyperGNNDiag', 'sheafHyperGNNOrtho', 'sheafHyperGNNGeneral', 'sheafHyperGNNLowRank']:
        model = sheafHyperGNN(args, args.method)
    
    else:
        raise ValueError(f"Unknown method: {args.method}")

    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


# =============================================================================
# Section 4: Configuration & Data Loading
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1 ,2 ,3], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2, type=int)
    parser.add_argument('--MLP_hidden', default=64, type=int)
    parser.add_argument('--Classifier_num_layers', default=2, type=int)
    parser.add_argument('--Classifier_hidden', default=64, type=int)
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default=True)
    parser.add_argument('--GPR', action='store_false')
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)
    parser.add_argument('--num_classes', default=0, type=int)
    parser.add_argument('--feature_noise', default='1', type=str)
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    
    # Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', default=True, type=bool)
    
    # Args for Attentions
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    
    # Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    
    # Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    
    # Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default=0)
    parser.add_argument('--UniGNN_degE', default=0)
    
    # Args for Noise Experiment
    parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
    parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'rayleigh'])
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the data')
    
    # Args for Sheaves
    parser.add_argument('--init_hedge', default="rand", type=str, choices=['rand', 'avg']) 
    parser.add_argument('--use_attention', type=str2bool, default=True) 
    parser.add_argument('--sheaf_normtype', type=str, default='degree_norm', choices=['degree_norm', 'block_norm', 'sym_degree_norm', 'sym_block_norm'])
    parser.add_argument('--sheaf_act', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'none'])
    parser.add_argument('--sheaf_dropout', type=str2bool, default=False)
    parser.add_argument('--sheaf_left_proj', type=str2bool, default=False)
    parser.add_argument('--dynamic_sheaf', type=str2bool, default=False)
    parser.add_argument('--sheaf_special_head', type=str2bool, default=False)
    parser.add_argument('--sheaf_pred_block', type=str, default="MLP_var1")
    parser.add_argument('--sheaf_transformer_head', type=int, default=1)
    parser.add_argument('--AllSet_input_norm', default=True)
    parser.add_argument('--residual_HCHA', default=False)
    parser.add_argument('--rank', default=0, type=int, help='rank for dxd blocks in LowRanksheafs')
    parser.add_argument('--n_sub', default=0, type=int, help='the number of sheaf')
    
    # Scheduler
    parser.add_argument('--lr_patience', type=int, default=10, help='Patience for reducing learning rate.')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor by which the learning rate will be reduced.')

    # Set Defaults
    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HCHA_symdegnorm=False)

    args = parser.parse_args()
    return args

def load_and_preprocess_data(args):
    """加载并预处理数据"""
    existing_dataset = [
        'coauthor_cora', 'coauthor_dblp', 'house-committees', 'house-committees-100',
        'cora', 'citeseer', 'pubmed', 'congress-bills', 'senate-committees', 
        'senate-committees-100', 'congress-bills-100', '20newsW100', 'ModelNet40', 
        'zoo', 'NTU2012', 'Mushroom','yelp',
    ]
    
    if args.method in ['sheafHyperGCNLowRank', 'LowRanksheafsDiffusion', 'sheafEquivSetGNN_LowRank']:
        assert args.rank <= args.heads // 2

    synthetic_list = ['house-committees', 'house-committees-100', 'congress-bills', 
                      'senate-committees', 'senate-committees-100', 'congress-bills-100']
    
    # 1. Load Data Source
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, feature_noise=f_noise, p2raw=p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = './data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = './data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = './data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, root='./data/pyg_data/hypergraph_dataset_updated/', p2raw=p2raw)
            
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        
        # Adjust Y labels for specific datasets
        if args.dname in ['house-committees', 'house-committees-100', 'senate-committees', 
                          'senate-committees-100', 'congress-bills', 'congress-bills-100',
                          'yelp', 'ModelNet40', 'zoo']:
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
            
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            data.num_hyperedges = torch.tensor([data.edge_index[0].max() - data.n_x[0] + 1])
        assert data.y.min().item() == 0

    # 2. Preprocessing based on Method
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
        data = norm_contruction(data, option=args.normtype)

    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')

    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)

    elif args.method in ['sheafHyperGCNDiag', 'sheafHyperGCNOrtho', 'sheafHyperGCNGeneral', 'sheafHyperGCNLowRank']:
        data = ExtractV2E(data)
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ['HCHA', 'HGNN', 'Diagsheafs', 'Orthosheafs', 'Generalsheafs', 'LowRanksheafs', 
                         'sheafHyperGNNDiag', 'sheafHyperGNNOrtho', 'sheafHyperGNNGeneral', 'sheafHyperGNNLowRank']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data.edge_index[1] -= data.edge_index[1].min()

    return dataset, data, args


# =============================================================================
# Section 5: Main Execution Loop
# =============================================================================

def main():
    args = get_args()
    
    # 1. Logging Setup
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_folder, f"training_log_{current_time}.txt")

    # 2. Data Loading
    dataset, data, args = load_and_preprocess_data(args)

    # 3. Generate Splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)

    # 4. Device Setup & Model Init
    model = parse_method(args, data)
    if args.cuda in [0, 1, 2, 3]:
        device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model, data = model.to(device), data.to(device)

    # 5. Training & Evaluation
    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    runtime_list = []
    noise_acc = []
    
    total_start_time = time.time()

    # --- Run Loop ---
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # Scheduler Init
        if args.lr_patience > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience
            )

        # --- Epoch Loop ---
        for epoch in range(args.epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            # Evaluate
            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])

            # File Logging
            with open(log_file_path, "a") as log_file:
                if epoch % args.display_step == 0 and args.display_step > 0:
                    log_message = (
                        f'Epoch: {epoch:02d}, '
                        f'Train Loss: {loss:.4f}, Valid Loss: {result[4]:.4f}, Test Loss: {result[5]:.4f}, '
                        f'Train Acc: {100 * result[0]:.2f}%, Valid Acc: {100 * result[1]:.2f}%, Test Acc: {100 * result[2]:.2f}%\n'
                    )
                    log_file.write(log_message)

            # Scheduler Step
            if args.lr_patience > 0:
                val_loss = result[4]
                scheduler.step(val_loss)

        # Noise Evaluation (Per Run)
        if args.add_noise:
            noise_acc_snr = []
            for snr in range(-6, 11, 2):
                acc = evaluate_noise(model, data, split_idx, eval_func, snr, args.noise_type)
                noise_acc_snr.append(acc)
            noise_acc.append(noise_acc_snr)

        end_time = time.time()
        runtime_list.append(end_time - start_time)

    # 6. Post-Processing & Saving Results
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
    best_val, best_test = logger.print_statistics()

    # Save Noise Results
    noise_root = 'noise_results'
    res_root = 'hyperparameter_tuning'
    os.makedirs(noise_root, exist_ok=True)
    os.makedirs(res_root, exist_ok=True)

    if args.add_noise:
        noise_acc = np.array(noise_acc)
        noise_acc_mean = np.mean(noise_acc, 0)
        noise_acc_std = np.std(noise_acc, 0)
        
        noise_filename = f'{noise_root}/{args.dname}_noise.txt'
        print(f"Saving noise results to {noise_filename}")
        
        with open(noise_filename, 'a+') as write_obj:
            header = f"[{args.method}] lr={args.lr}, wd={args.wd}, heads={args.heads}, noise={args.noise_type}"
            write_obj.write(header + '\n')
            write_obj.write("-" * 40 + '\n')
            for snr, mean, std in zip(range(-6, 11, 2), noise_acc_mean, noise_acc_std):
                write_obj.write(f"SNR {snr:2d}dB: {mean:.3f} ± {std:.3f}\n")
            write_obj.write("-" * 40 + '\n\n')

        print("\nNoise Test Results:")
        print("-" * 40)
        for snr, mean, std in zip(range(-6, 11, 2), noise_acc_mean, noise_acc_std):
            print(f"SNR {snr:2d}dB: {mean:.3f} ± {std:.3f}")
        print("-" * 40)

    # Save Summary CSV
    filename = f'{res_root}/{args.dname}_.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s\n'
        write_obj.write(cur_line)

    # Save All Args
    all_args_file = f'{res_root}/all_args_{args.dname}_.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args) + '\n')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"TIME FOR ONE EXPERIMENT WITH {args.runs} RUNS: \\ Minutes: {total_time//60}, seconds {total_time%60}")
    print('All done! Exit python code')

if __name__ == '__main__':
    main()