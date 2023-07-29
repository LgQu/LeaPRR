import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.parameter import Parameter
from .RerankTransformer import RerankTransformer


class MutualRerankTransformer(nn.Module):
    def __init__(self, topk_i2t, topk_t2i, topk_dim=512, embed_dim=2048, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(MutualRerankTransformer, self).__init__()
        self.Eiters = 0
        self.topk_i2t = topk_i2t
        self.topk_t2i = topk_t2i

        self.model_i2t = RerankTransformer(embed_dim=embed_dim, topk_dim=topk_dim, topk=topk_i2t, 
                              depth=depth, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                              drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

        self.model_t2i = RerankTransformer(embed_dim=embed_dim, topk_dim=topk_dim, topk=topk_t2i, 
                              depth=depth, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                              drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

    def set_mode(self, mode='normal'):
        if mode == 'normal':
            self.model_i2t.n_inter_neighbor = self.topk_i2t
            self.model_t2i.n_inter_neighbor = self.topk_t2i
        elif mode == 'align':
            self.model_i2t.n_inter_neighbor = self.topk_t2i
            self.model_t2i.n_inter_neighbor = self.topk_i2t
        else:
            assert False, 'Unknown mode: ' + mode

    @autocast()
    def forward_feature_without_mapping(self, x_topk_i2t: Tensor, x_topk_t2i: Tensor, topk_indices_i2t: Tensor=None, topk_indices_t2i: Tensor=None):
        sim_i2t, norm_x_i2t = self.model_i2t.forward_feature_without_mapping(x_topk_i2t, x_topk_i2t)
        sim_t2i, norm_x_t2i = self.model_t2i.forward_feature_without_mapping(x_topk_t2i, x_topk_t2i)
        return (sim_i2t, norm_x_i2t), (sim_t2i, norm_x_t2i)

    @autocast()
    def forward_feature(self, x_topk_i2t: Tensor, x_topk_t2i: Tensor, A_mat_i2t: Tensor, A_mat_t2i: Tensor, topk_indices_i2t: Tensor=None, topk_indices_t2i: Tensor=None):
        sim_i2t, norm_x_i2t = self.model_i2t.forward_feature(x_topk_i2t, A_mat_i2t, topk_indices_i2t)
        sim_t2i, norm_x_t2i = self.model_t2i.forward_feature(x_topk_t2i, A_mat_t2i, topk_indices_t2i)
        return (sim_i2t, norm_x_i2t), (sim_t2i, norm_x_t2i)

    @autocast()
    def forward(self, x_topk_i2t: Tensor, x_topk_t2i: Tensor, A_mat_i2t: Tensor, A_mat_t2i: Tensor, topk_indices_i2t: Tensor=None, topk_indices_t2i: Tensor=None):
        exp_sim_i2t, MSE_loss_i2t, Hinge_loss_i2t, sim_i2t, norm_x_i2t = self.model_i2t(x_topk_i2t, A_mat_i2t, topk_indices_i2t)
        exp_sim_t2i, MSE_loss_t2i, Hinge_loss_t2i, sim_t2i, norm_x_t2i = self.model_t2i(x_topk_t2i, A_mat_t2i, topk_indices_t2i)

        return (exp_sim_i2t, MSE_loss_i2t, Hinge_loss_i2t, sim_i2t, norm_x_i2t), \
                (exp_sim_t2i, MSE_loss_t2i, Hinge_loss_t2i, sim_t2i, norm_x_t2i)