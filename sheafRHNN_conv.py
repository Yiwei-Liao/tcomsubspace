import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from helper import get_param

# 确保导入路径正确
from sheaf_builder import sheafBuilderOrtho, sheafBuilderDiag, sheafBuilderGeneral
from layers import HyperDiffusionOrthosheafConv, HyperDiffusionDiagsheafConv, HyperDiffusionGeneralsheafConv, MLP

class sheafRHNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        # --- 参数适配 ---
        self.d = getattr(self.p, 'heads', 2) 
        self.p.heads = self.d
        self.p.MLP_hidden = getattr(self.p, 'MLP_hidden', in_channels // self.d)
        self.p.sheaf_dropout = getattr(self.p, 'sheaf_dropout', 0.0)
        self.p.sheaf_pred_block = getattr(self.p, 'sheaf_pred_block', 'MLP_var1')
        self.p.sheaf_act = getattr(self.p, 'sheaf_act', 'tanh')
        self.p.AllSet_input_norm = getattr(self.p, 'AllSet_input_norm', False)
        self.p.sheaf_special_head = False
        
        # --- Builder 选择 ---
        self.builder_type = 'Diag' 
        
        if self.builder_type == 'Ortho':
            self.sheaf_builder = sheafBuilderOrtho(self.p)
            self.conv_layer = HyperDiffusionOrthosheafConv(
                in_channels=self.p.MLP_hidden, 
                out_channels=self.p.MLP_hidden, 
                d=self.d, 
                device=None,
                norm_type='degree_norm',
                residual=True
            )
        elif self.builder_type == 'Diag':
            self.sheaf_builder = sheafBuilderDiag(self.p)
            self.conv_layer = HyperDiffusionDiagsheafConv(
                in_channels=self.p.MLP_hidden, out_channels=self.p.MLP_hidden, d=self.d, 
                device=None, norm_type='degree_norm', residual=True
            )
        
        # --- 输入投影 ---
        self.lin_in = MLP(in_channels=in_channels, 
                          hidden_channels=self.p.MLP_hidden,
                          out_channels=self.p.MLP_hidden * self.d,
                          num_layers=1, dropout=0.0, Normalization='ln', InputNorm=False)

        self.lin_edge = MLP(in_channels=self.p.init_dim, 
                            hidden_channels=self.p.MLP_hidden,
                            out_channels=self.p.MLP_hidden * self.d,
                            num_layers=1, dropout=0.0, Normalization='ln', InputNorm=False)
        
        # 关系变换参数 (用于返回 transform 后的 relation embedding)
        self.w_rel = get_param((in_channels, out_channels))
        
        # Loop 边参数
        self.loop_rel = get_param((1, self.p.init_dim))
        
        # Multi-Relation Loop 参数 (用于 multi_r 模式)
        # 形状对应 (1, 6, dim)
        self.multi_loop_rel = get_param((6, self.p.init_dim)).view(1, 6, self.p.init_dim)

        self.bn = torch.nn.BatchNorm1d(out_channels)
        
        if self.p.bias: 
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_order, edge_type, rel_embed, multi_rel_embed=None): 
        self.device = x.device
        self.conv_layer.device = self.device

        # 1. 计算 Row (Hyperedge) Embedding
        if self.p.id2entity_instance.device != self.device:
            self.p.id2entity_instance = self.p.id2entity_instance.to(self.device)
        
        id2entity_instance = self.p.id2entity_instance
        input_add1 = torch.cat((x, torch.zeros(1, x.size()[1]).to(self.device)), 0)
        
        sign_a = torch.sign(id2entity_instance + 1).int()
        non_zero_a = torch.count_nonzero(sign_a, dim=1).reshape(-1, 1).to(self.device)
        instance_dict_embedding = torch.sum(input_add1[id2entity_instance], 1) / non_zero_a

        # 2. 构建联合特征矩阵 (Entities + Rows)
        num_ent = x.shape[0]
        num_row = instance_dict_embedding.shape[0]
        x_all = torch.cat([x, instance_dict_embedding], dim=0)

        # 3. 调整索引: Row ID 偏移
        target_indices = edge_index[0] 
        source_indices = edge_index[1] + num_ent 
        
        # 4. 构建 Incidence Matrix
        num_edges = edge_index.shape[1]
        edge_ids = torch.arange(num_edges, device=self.device)
        
        nodes = torch.cat([source_indices, target_indices])
        edges = torch.cat([edge_ids, edge_ids])
        H_index = torch.stack([nodes, edges], dim=0)

        # 5. 准备 Relation Features
        rel_embed_full = torch.cat([rel_embed, self.loop_rel], dim=0)
        
        # 处理 Multi-Relation (关键修复点)
        if multi_rel_embed is not None:
            multi_rel_embed = multi_rel_embed.to(self.device)
            self.multi_loop_rel = self.multi_loop_rel.to(self.device)
            multi_rel_embed_full = torch.cat([multi_rel_embed, self.multi_loop_rel], dim=0)
        else:
            multi_rel_embed_full = None

        # 6. 特征投影
        x_proj = self.lin_in(x_all)
        x_view = x_proj.view((num_ent + num_row) * self.d, self.p.MLP_hidden)
        
        current_edge_attrs = rel_embed_full[edge_type]
        edge_attr_proj = self.lin_edge(current_edge_attrs)
        edge_attr_view = edge_attr_proj.view(num_edges * self.d, self.p.MLP_hidden)

        # 7. Infer sheaf & Diffusion
        h_sheaf_index, h_sheaf_attributes = self.sheaf_builder(x_view, edge_attr_view, H_index)
        
        out_all = self.conv_layer(x_view, 
                                  hyperedge_index=h_sheaf_index, 
                                  alpha=h_sheaf_attributes, 
                                  num_nodes=num_ent + num_row, 
                                  num_edges=num_edges)
        
        out_all = F.elu(out_all)
        out_all = F.dropout(out_all, p=self.p.dropout, training=self.training)
        
        # 8. 提取 Entity 结果
        out_all = out_all.view(num_ent + num_row, self.d * self.p.MLP_hidden)
        out_ent = out_all[:num_ent]

        if self.p.bias: out_ent = out_ent + self.bias
        out_ent = self.bn(out_ent)

        # ================== 修复核心：正确返回 multi_rel_embed ==================
        # 对 Relation 应用线性变换 (w_rel) 并移除最后一位 (Loop边)
        r_out = torch.matmul(rel_embed_full, self.w_rel)[:-1]
        
        if multi_rel_embed_full is not None:
            # 同样对 multi_rel 进行变换
            multi_r_out = torch.matmul(multi_rel_embed_full, self.w_rel)[:-1]
            return self.act(out_ent), r_out, multi_r_out
        else:
            return self.act(out_ent), r_out, 1