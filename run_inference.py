import os
import argparse

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-gpu', '--device', type=str)
args1, unknown = parser.parse_known_args()
if args1.device:
	os.environ["CUDA_VISIBLE_DEVICES"] = args1.device

import copy
from tqdm import tqdm
from helper import *
from data_loader import *
from models import *
import msgpack
import msgpack_numpy as m

from natsort import natsorted

class Runner(object):

	"""
	The entry class for this project contains data reading and processing, training, evaluation, prediction, etc.

	"""
	def load_data(self):
		"""
		Functions to read data files and pre-process them for use in other parts of this project

		"""

		ent_set, rel_set,unique_entities = OrderedSet(), OrderedSet(), OrderedSet()
		rows=[]
		self.data = ddict(list)
		self.data_sub_order=ddict(list)
		self.data_instance = ddict(list)
		sr2o = ddict(set)
		if self.p.opn=='norela':
			self.norepeat_col2row=ddict(set)

		self.p.data_name=self.p.dataset.split('/')[-1]

		if self.p.data_name=='FB15k-237' or self.p.data_name=='WN18RR' or self.p.data_name=='YAGO3-10':
			self.p.arity=2
		for split in ['train', 'test', 'valid']:
			filename=f'{self.p.dataset}/{split}.kb'
			with open(filename,"rb") as f:  
				kb_file = msgpack.unpack(f)
			np_kb_file = np.array(kb_file, dtype=np.int32)  

			for n,np_kb_row in enumerate(np_kb_file):
				for i,nk in enumerate(np_kb_row):
					if self.p.data_name=='FB-AUTO':
						if np_kb_row[i]==3388 and(np_kb_row[i+1]==3388 or np_kb_row[i+1]==1):
							np_kb_row[i]=-1					
					if self.p.data_name=='JF17K':
						if np_kb_row[i]==29259 and(np_kb_row[i+1]==29259 or np_kb_row[i+1]==1):
							np_kb_row[i]=-1
					if i ==len(np_kb_row)-1:
						np_kb_row[i]=-1
			data_list=np_kb_file
			if self.p.arity==2:
				np_kb_file=np.concatenate((np_kb_file,np.zeros((np_kb_file.shape[0],7-np_kb_file.shape[1]), dtype=np.int32)-1),1)
		
			data_tuple_list=[]
			for data in data_list:
				data_tuple=tuple(data)
				data_tuple_list.append(data_tuple)
				
			norepeat_sub=[]
			norepeat_rows=[]
			self.data_instance[split]=data_tuple_list
			for data in data_tuple_list:
				edge=[]
				for i,d in enumerate(data):
					if i==0:
						rel_set.add(d)
						r=d
					else:
						if d!=-1:
							ent_set.add(d)
							edge.append(d)

				directed=False
				if not directed:
					for ei,e in enumerate(edge):
						edge_cp=edge.copy()
						edge_cp.remove(e)
						rows.append(tuple(edge_cp+[-1]*(5-len(edge_cp))))
						
						if self.p.opn=='norela':
							norela_row=[]
							for cp in edge_cp:
								if cp not in self.norepeat_col2row[e]:
									norela_row.append(cp)
							if len(norela_row)>0:
								norepeat_sub.append(e)
								norepeat_rows.append(tuple((norela_row+[-1]*(5-len(norela_row)))))

				if self.p.opn=='norela':
					for i,d in enumerate(data):
						if i!=0 and d!=-1:
							for ii,dd in enumerate(data):
								if ii!=0 and dd!=d and dd!=-1:
									self.norepeat_col2row[d].add(dd)

		if self.p.opn=='norela':
			rows_norepeat_set=list(set(norepeat_rows))
			self.p.norepeat_sub=norepeat_sub
			self.p.norepeat_rows=norepeat_rows

		rows_set=list(set(rows))
		rows_set_loop=[]
		for e in ent_set:
			rows_set_loop.append(tuple([e]+[-1]*4))

		use_sort=False
		if use_sort:
			rows_set.sort(key=rows.index)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.p.row2id2id={self.tuple2id(data,self.ent2id):i for i,data in enumerate(rows_set)} 
		self.p.row2id2id_loop={self.tuple2id(data,self.ent2id):i for i,data in enumerate(rows_set_loop)} 

		self.p.id2entity_instance=torch.tensor(list(self.p.row2id2id.keys()))
		self.p.id2entity_instance_loop=torch.tensor(list(self.p.row2id2id_loop.keys()))

		self.p.row2id={data:i for i,data in enumerate(rows_set)} 
		self.p.row2id_loop={data:i for i,data in enumerate(rows_set_loop)} 

		if self.p.opn=='norela':
			self.p.row_norepeat_2id={data:i for i,data in enumerate(rows_norepeat_set)} 
			self.p.id2row_norepeat={i:data for i,data in enumerate(rows_norepeat_set)} 

			self.p.row_norepeat_2id2id={self.tuple2id(data,self.ent2id):i for i,data in enumerate(rows_norepeat_set)} 
			self.p.id2norepeat_entity_instance=torch.tensor(list(self.p.row_norepeat_2id2id.keys()))

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
		self.p.id2row={i:data for i,data in enumerate(rows_set)} 

		self.p.num_ent		= len(self.ent2id)

		self.p.num_rel		= len(self.rel2id)
		self.p.num_row=len(self.p.row2id)
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
		
		self.p.ent_edge_matrix=torch.zeros((self.p.num_ent,self.p.num_rel)).cuda()

		sub_pos_set=OrderedSet()
		for split in ['train', 'test', 'valid']:
			for data in tqdm(self.data_instance[str(split)]):
				edge=[]
				for i,d in enumerate(data):
					if i==0:
						r=d
					else:
						if d!=-1:
							edge.append(d)
				directed=False
				if not directed:
					for ei,e in enumerate(edge):
						edge_cp=edge.copy()
						sub_edge_order=list(range(0,len(edge)))
						edge_cp.remove(e)
						sub_edge_order.remove(ei)

						sub=self.p.row2id[tuple(edge_cp+[-1]*(5-len(edge_cp)))]
						obj=self.ent2id[e]
						rel=self.rel2id[r]
						sub_index=sub_edge_order+[6]*(5-len(edge_cp))
						
						obj_index=ei

						self.data_sub_order[split].append(sub_index.copy())

						sub_index.append(ei)
						if self.p.position:
							sub=tuple([sub,tuple(sub_index)])
							sub_pos_set.append(sub)

						self.data[split].append((sub, rel, obj))

						if split == 'train':  
							sr2o[(sub, rel)].add(obj)
							self.p.ent_edge_matrix[obj][rel]=1

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				
		if self.p.position:
			self.p.id2sub_pos=[data for i,data in enumerate(sub_pos_set)]
			self.p.sub_pos2id={data:i for i,data in enumerate(sub_pos_set)}

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)

		if self.p.position:
			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(self.p.sub_pos2id[sub], rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

			for split in ['test', 'valid']:
				for sub, rel, obj in self.data[split]:
					self.triples['{}_{}'.format(split, 'tail')].append({'triple': (self.p.sub_pos2id[sub], rel, obj), 'label': self.sr2o_all[(sub, rel)]})
		else:
			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

			for split in ['test', 'valid']:
				for sub, rel, obj in self.data[split]:
					self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})

		self.triples = dict(self.triples)
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.p.id2entity_instance = self.p.id2entity_instance.cuda()
			self.p.id2entity_instance_loop = self.p.id2entity_instance_loop.cuda()
			if self.p.opn == 'norela':
				self.p.id2norepeat_entity_instance = self.p.id2norepeat_entity_instance.cuda()

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		self.edge_index, self.edge_type = self.construct_adj()
		
		if self.p.opn=='norela':
			self.p.norepeat_edge_index, self.p.norepeat_edge_type = self.construct_norepeat_adj()

	def construct_adj(self):

		edge_index, edge_type = [], []
		edge_order=[]

		for sub, rel, obj in tqdm(self.data['train']):
			if self.p.position:
				sub=sub[0]
			edge_index.append((sub, obj))
			edge_type.append(rel)

		for sub_order in tqdm(self.data_sub_order['train']):
			edge_order.append(sub_order)
		edge_order	= torch.LongTensor(np.array(edge_order)).to('cuda')
		edge_index	= torch.LongTensor(edge_index).to('cuda').t()
		edge_type	= torch.LongTensor(edge_type). to('cuda')

		return (edge_index,edge_order),edge_type

	def construct_norepeat_adj(self):

		edge_index, edge_type = [], []
		

		for sub,obj in zip(self.p.norepeat_sub,self.p.norepeat_rows):
			edge_index.append((self.ent2id[sub], self.p.row_norepeat_2id[obj]))
			edge_type.append(0)

		edge_index	= torch.LongTensor(edge_index).to('cuda').t()
		edge_type	= torch.LongTensor(edge_type).to('cuda')

		return edge_index,edge_type
			
	def tuple2id(self,ent_tuple,ent2id):
		return tuple([ent2id[t] if t!=-1 else -1 for t in ent_tuple])

	def __init__(self, params):
		"""
		Constructor of the runner class	
		"""
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = self.add_optimizer(self.model.parameters())


	def add_model(self, model, score_func):

		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'rhkh_transe': 	model = RHKH_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'rhkh_distmult': 	model = RHKH_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'rhkh_conve': 	model = RHKH_ConvE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'rhkh_boxe': 	model = RHKH_Box(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

		model.to('cuda')

		if torch.cuda.device_count() > 1:
			print('torch.cuda.device_count():',torch.cuda.device_count())
			model = torch.nn.DataParallel(model)
		return model

	def add_optimizer(self, parameters):

		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):

		if split == 'train':
			triple, label = [ _.to('cuda') for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to('cuda') for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
			# ================== 新增代码开始 ==================
			# 获取父目录路径 (例如 ./checkpoints)
			directory = os.path.dirname(save_path)
			# 如果目录不存在，则递归创建
			if not os.path.exists(directory):
				os.makedirs(directory, exist_ok=True)
			# ================== 新增代码结束 ==================

			state = {
				'state_dict'	: self.model.state_dict(),
				'best_val'	: self.best_val,
				'best_epoch'	: self.best_epoch,
				'optimizer'	: self.optimizer.state_dict(),
				'args'		: vars(self.p)
			}
			torch.save(state, save_path)

	def load_model(self, load_path):

		state			= torch.load(load_path, weights_only=False)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		"""
		Functions for evaluation using models

		"""
		left_results  = self.predict(split=split, mode='tail_batch')
		results       = get_combined_results(left_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}'.format(epoch, split, results['mrr']))

		return results

	def predict(self, split='valid', mode='tail_batch'):
		"""
		Functions for prediction using models

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):

				sub, rel, obj, label	= self.read_batch(batch, split)

				ary_list=[]
				for s in sub:
					if self.p.position:
						sub0=self.p.id2sub_pos[s]
						s,pos=sub0[0],sub0[1]
					if torch.is_tensor(s):
						s=s.item()
					sub_tuple=self.p.id2row[s]
					ary=np.count_nonzero(np.array(list(sub_tuple))+1)+1
					ary_list.append(ary)

				pred			= self.model.forward(sub, rel)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)
				for ai,ary in enumerate(ary_list):
				
					results['{}_ary_count'.format(ary)]	=1+ results.get('{}_ary_count'.format(ary), 0.0)
					results['{}_ary_mr'.format(ary)] = ranks[ai].item() 	+ results.get('{}_ary_mr'.format(ary),    0.0)
					results['{}_ary_mrr'.format(ary)] = 1.0/ranks[ai].item()   + results.get('{}_ary_mrr'.format(ary),   0.0)
					import math
					rank=math.ceil(ranks[ai].item())
					if rank<=10:
						for j in range(rank,10+1):
							results['{}_ary_hits@{}'.format(ary,j)] =1+ results.get('{}_ary_hits@{}'.format(ary,j), 0.0)
				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
		
		return results


	def run_epoch(self, epoch, val_mrr = 0):
		"""
		Function for a single epoch that runs during training

		"""
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')
			pred	= self.model.forward(sub, rel)
			
			if torch.cuda.device_count() > 1:
				loss	= self.model.module.loss(pred, label)
			else:
				loss	= self.model.loss(pred, label)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss


	def fit(self):
		"""
		Functions for training and evaluating models

		"""
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr)
			val_results = self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 80: 
					self.logger.info("Early Stopping!!")
					break

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		test_results = self.evaluate('test', epoch)
		test_results=natsorted(list(test_results.items()))
		self.logger.info(test_results)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='./data/JF17K',            help='Dataset to use, default: JF17K')
	parser.add_argument('-model',		dest='model',		default='RHKH',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in RHKH')

	parser.add_argument('-batch',           dest='batch_size',      default=256,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-id_drop',  	dest='id_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')


	parser.add_argument('-id_drop2',  	dest='id_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-arity',          dest='arity',            type=int, default=5, help='Arity of nodes')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	parser.add_argument('-norm',          dest='norm',      default=1,         type=int, help='0:no norm,1:old norm, 2:new norm')
	parser.add_argument('-margin',          dest='margin',            type=int, default=0, help='Whether to use margin loss')
	parser.add_argument("-neg", dest='neg', default=-1,type=int, help='Ratio of valid to invalid triples for training')

	parser.add_argument('-entity_conv',          dest='entity_conv',            type=int, default=0, help='Whether to pass convolution layers when aggregating other entity representations connected by an entity')
	parser.add_argument('-dist_type',          dest='dist_type',            type=int, default=0, help='0: Distance from the center of the ball, 1:boxe, 2:new score func')

	parser.add_argument('-r2fc',          dest='r2fc',            type=int, default=0, help='Whether to generate r_center after passing r through fc layers')
	parser.add_argument('-multi_r',          dest='multi_r',            type=int, default=0, help='Whether the relation has different representations in different locations')
	
	parser.add_argument('-position',          dest='position',            type=int, default=0, help='whether to consider position')
	parser.add_argument('-shift',          dest='shift',            type=int, default=0, help='0:no shift,1:one hot shift,2:rotate shift')

	# 是否添加噪声
	parser.add_argument('--add_noise', action='store_true', help='Whether to add noise to embeddings')

	# 噪声类型：高斯 or 瑞利
	parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'rayleigh'], help="Type of noise to add: 'gaussian' or 'rayleigh'")

	# 信噪比（SNR）控制噪声强度，单位为 dB（默认20）
	parser.add_argument('--snr', type=float, default=20.0, help="Signal-to-Noise Ratio in dB (higher = less noise)")

	parser.add_argument('-heads', type=int, default=2, help='Number of stalks/heads for Sheaf')
	parser.add_argument('-MLP_hidden', type=int, default=100, help='Hidden dim for Subspace MLP')
	args = parser.parse_args()
	args_copy=copy.deepcopy(args)
	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
	args.ngpus=set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Runner(args)
	model.fit()
	#print(args_copy)
