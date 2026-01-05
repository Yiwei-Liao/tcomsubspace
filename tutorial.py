# This is a simple example to show how to perform inference on the sheafGNN/sheafHGCN models.
import torch

from torch_geometric.data import Data
from argparse import Namespace
from layers import *
from models import *



args_dict = {
    'num_features': 10,     # number of node features
    'num_classes': 4,       # number of classes
    'All_num_layers': 2,    # number of layeers
    'dropout': 0.3,         # dropout rate
    'MLP_hidden': 256,      # dimension of hidden state (for most of the layers)
    'AllSet_input_norm': True,  # normalising the input at each layer
    'residual_HCHA': False, # using or not a residual connectoon per sheaf layer

    'heads': 6,             # dimension of reduction map (d)
    'init_hedge': 'avg',    # how to compute hedge features when needed. options: 'avg'or 'rand'
    'sheaf_normtype': 'sym_degree_norm',  # the type of normalisation for the sheaf Laplacian. options: 'degree_norm', 'block_norm', 'sym_degree_norm', 'sym_block_norm'
    'sheaf_act': 'tanh',    # non-linear activation used on tpop of the d x d restriction maps. options: 'sigmoid', 'tanh', 'none'
    'sheaf_left_proj': False,   # multiply to the left with IxW or not
    'dynamic_sheaf': False, # infering a differetn sheaf at each layer or use ta shared one

    'sheaf_pred_block': 'cp_decomp', # indicated the type of model used to predict the restriction maps. options: 'MLP_var1', 'MLP_var3' or 'cp_decomp' 
    'sheaf_dropout': False, # use dropout in the sheaf layer or not
    'sheaf_special_head': False,    # if True, add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
    'rank': 2,              # only for LowRank type of sheaf. mention the rank of the reduction matrix

    'HyperGCN_mediators': True, #only for the Non-Linear sheaf laplacian. Indicates if mediators are used when computing the non-linear Laplacian (same as in HyperGCN)
    'cuda': 0

}

args = Namespace(**args_dict)

if args.cuda in [0, 1]:
    device = torch.device('cuda:'+str(args.cuda)
                            if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# create a random hypergraph to run inference for
num_nodes = 25
features = torch.rand(num_nodes, args.num_features)
edge_index = torch.tensor([[0,1,2,0,1,3,4,1,2,4],[0,0,0,1,1,1,1,2,2,2]])
labels = torch.randint(args.num_classes, (num_nodes,))
data = Data(x = features, edge_index = edge_index, y = labels).to(device)

# Running Linear sheafHNN. 
# To change the type of restrictian map change between
# sheaf_type= 'sheafHyperGNNDiag'/'sheafHyperGNNGeneral'/'sheafHyperGNNOrtho'/'sheafHyperGNNLowRank'
model = sheafHyperGNN(args, sheaf_type='sheafHyperGNNDiag').to(device)
out = model(data)
print(out.shape)

# Running Non-Linear sheafHNN. 
# To change the type of restrictian map change between
# sheaf_type= 'Diagsheafs'/'Generalsheafs'/'Orthosheafs'/'LowRanksheafs'
model = Smodel = sheafHyperGCN(V=data.x.shape[0],
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args, sheaf_type= 'Diagsheafs'
                         ).to(device)
out = model(data)
print(out.shape)

