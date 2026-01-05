import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet

import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add

# =========================================================
# Fix for PyTorch > 1.8: Using torch.fft module
# =========================================================
import torch.fft 

np.set_printoptions(precision=4)


def set_gpu(gpus):

	print('set gpu :',gpus)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

def get_logger(name, log_dir, config_dir):

	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def get_combined_results(left_results):
	results = {}
	count   = float(left_results['count'])

	results['mr']		= round(left_results ['mr'] /count, 5)
	results['mrr']		= round(left_results ['mrr']/count, 5)
	results['count']=left_results['count']
	for k in range(10):
		results['hits@{}'.format(k+1)]		= round(left_results ['hits@{}'.format(k+1)]/count, 5)

	for j in range(2,7):
		if '{}_ary_count'.format(j) in left_results:
			count	=float(left_results['{}_ary_count'.format(j)])
			results['{}_ary_count'.format(j)]=left_results['{}_ary_count'.format(j)]
			results['{}_ary_mr'.format(j)] = round(left_results['{}_ary_mr'.format(j)]/count, 5)
			results['{}_ary_mrr'.format(j)] = round(left_results['{}_ary_mrr'.format(j)]/count, 5)
			for k in range(1,11):
				hit_key='{}_ary_hits@{}'.format(j,k)
				if hit_key not in left_results:
					results[hit_key]=0
				else:
					results[hit_key]=round(left_results[hit_key]/count, 5)
	return results

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

# 注意：新版 PyTorch FFT 自动处理复数，不需要手动分离实部虚部。
# 以下两个函数(com_mult, conj)是为旧版代码保留的，
# 但在新的 cconv 和 ccorr 中不再使用它们。
def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

# =========================================================
# Updated Functions for PyTorch 1.8+
# =========================================================

def cconv(a, b):
    """
    Circular Convolution using new torch.fft API
    """
    # 1. Real -> Complex FFT
    a_fft = torch.fft.rfft(a, dim=-1)
    b_fft = torch.fft.rfft(b, dim=-1)
    
    # 2. Element-wise multiplication (complex tensors support * directly)
    prod = a_fft * b_fft
    
    # 3. Complex -> Real Inverse FFT
    # n=a.shape[-1] ensures the output size matches input size
    return torch.fft.irfft(prod, n=a.shape[-1], dim=-1)

def ccorr(a, b):
    """
    Circular Correlation using new torch.fft API
    """
    # 1. Real -> Complex FFT
    a_fft = torch.fft.rfft(a, dim=-1)
    b_fft = torch.fft.rfft(b, dim=-1)
    
    # 2. Complex Conjugate of a
    a_fft = torch.conj(a_fft)
    
    # 3. Element-wise multiplication
    prod = a_fft * b_fft
    
    # 4. Complex -> Real Inverse FFT
    return torch.fft.irfft(prod, n=a.shape[-1], dim=-1)