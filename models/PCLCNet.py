import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils_ import misc
from models import so3conv as M
import vgtk.so3conv.functional as L
import numpy as np
from utils_.rotations import rotate, reverse_rotate
from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d
import os


class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
   
class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


####################################### EPN #######################################  
class RegressorFC(nn.Module):
    def __init__(self, latent_dim=128, bn=False):
        super(RegressorFC, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        return x # B, 3, 3

class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, classifier=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.classifier = classifier

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf
        if self.classifier == False:
            fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        else:
            fc_layer = nn.Linear(self.n_features[-1], output_pts)
        model.append(fc_layer)
        # add by XL
        if self.classifier == False:
            acti_layer = nn.Sigmoid()
        else:
            acti_layer = nn.Softmax()
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        if self.classifier == False:
            x = x.view((-1, 3, self.output_pts))
        return x
    
class EPN(nn.Module):
    def __init__(self, num_anchors=60):
        super(EPN, self).__init__()
        # params = self.get_parameters(num_anchors=num_anchors)
        # self.backbone = nn.ModuleList()
        # for block_param in params['backbone']:
        #     self.backbone.append(M.BasicSO3ConvBlock(block_param))
        # self.num_anchor = num_anchors
        # self.invariance = True
        # self.num_heads = 1
        # self.classifier = None
        # self.outblockRT = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method='max',
        #                                       pred_t=True, feat_mode_num=self.num_anchor, num_heads=self.num_heads)
        # self.outblockN = M.InvOutBlockOurs(params['outblock'], norm=1, pooling_method='max', pooling_x=True)
        # # self.anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).cuda()
        # self.anchors = torch.from_numpy(L.get_anchors(self.num_anchor)).cuda()
        self.encoder = InvSO3ConvModel(num_anchors=num_anchors, pooling_x=True, mlps=[[32,32], [64,64], [128,128], [256, 256]], out_mlps=[128, 128], strides=[2, 2, 2, 2])
        self.regressor = RegressorFC(None, bn=False)
        # completion
        self.decoder = DecoderFC((256, 256), 128, 1024, None)
    
    def get_parameters(self, num_anchors=60,
                        mlps=[[32,32], [64,64], [128,128], [256, 256]],
                        out_mlps=[128, 128],
                        strides=[2, 2, 2, 2],
                        initial_radius_ratio = 0.2,
                        sampling_ratio = 0.8,
                        sampling_density = 0.5,
                        kernel_density = 1,
                        kernel_multiplier = 2,
                        sigma_ratio= 0.5, # 1e-3, 0.68
                        xyz_pooling = None, # None, 'no-stride'
                        to_file=None):

        input_num = 1024
        dropout_rate= 0.
        temperature= 3
        so3_pooling =  'rotation'
        input_radius = 0.4
        kpconv = False
        na = num_anchors

        # to accomodate different input_num
        if input_num > 1024:
            sampling_ratio /= (input_num / 1024)
            strides[0] = int(2 * (input_num / 1024))
            print("Using sampling_ratio:", sampling_ratio)
            print("Using strides:", strides)
        print("[MODEL] USING RADIUS AT %f"%input_radius)
        params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'na': na
              }
        dim_in = 1

        # process args
        n_layer = len(mlps)
        stride_current = 1
        stride_multipliers = [stride_current]
        for i in range(n_layer):
            stride_current *= 2
            stride_multipliers += [stride_current]

        num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]
        # print("num center", num_centers)

        radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]
        # print("radius_ratio", radius_ratio)

        # radius_ratio = [0.25, 0.5]
        radii = [r * input_radius for r in radius_ratio]

        weighted_sigma = [sigma_ratio * radii[0]**2]
        for idx, s in enumerate(strides):
            weighted_sigma.append(weighted_sigma[idx] * s)

        for i, block in enumerate(mlps):
            block_param = []
            for j, dim_out in enumerate(block):
                lazy_sample = i != 0 or j != 0

                stride_conv = i == 0 or xyz_pooling != 'stride'

                # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
                neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

                if i == 0 and j == 0:
                    neighbor *= int(input_num / 1024)

                kernel_size = 1
                if j == 0:
                    # stride at first (if applicable), enforced at first layer
                    inter_stride = strides[i]
                    nidx = i if i == 0 else i+1
                    # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                    if stride_conv:
                        neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                        kernel_size = 1 # if inter_stride < 4 else 3
                else:
                    inter_stride = 1
                    nidx = i+1

                # one-inter one-intra policy
                block_type = 'inter_block' if na != 60  else 'separable_block'

                inter_param = {
                    'type': block_type,
                    'args': {
                        'dim_in': dim_in,
                        'dim_out': dim_out,
                        'kernel_size': kernel_size,
                        'stride': inter_stride,
                        'radius': radii[nidx],
                        'sigma': weighted_sigma[nidx],
                        'n_neighbor': neighbor,
                        'lazy_sample': lazy_sample,
                        'dropout_rate': dropout_rate,
                        'multiplier': kernel_multiplier,
                        'activation': 'leaky_relu',
                        'pooling': xyz_pooling,
                        'kanchor': na,
                    }
                }
                block_param.append(inter_param)

                dim_in = dim_out

            params['backbone'].append(block_param)
        representation = 'quat'
        params['outblock'] = {
                'dim_in': dim_in,
                'mlp': out_mlps,
                'fc': [64],
                'k': 40,
                'kanchor': na,
                'pooling': so3_pooling,
                'representation': representation,
                'temperature': temperature,
        }

        return params
    
    def forward(self, x):
        # # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        # if x.shape[-1] > 3:
        #     x = x.permute(0, 2, 1).contiguous()
        # x = M.preprocess_input(x, self.num_anchor, False)

        # for block_i, block in enumerate(self.backbone):
        #     x = block(x)
        # # if self.t_method_type < 0:
        # #     output = self.outblockR(x, self.anchors)
        # # else:
        # #     output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        # output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        # manifold_embed    = self.outblockN(x)
        # output['0'] = manifold_embed
        # output['xyz'] = x.xyz
        output = self.encoder(x)
        output['recon'] = self.decoder(output['0'])

        return  output

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

    def load_ckpt(self, name=None, model_dir=None):
        """load checkpoint from saved checkpoint"""
        if name == 'latest':
            pass
        elif name == 'best':
            pass
        else:
            name = "ckpt_epoch{}".format(name)

        load_path = os.path.join(model_dir, "{}.pth".format(name))

        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path, map_location='cuda:0')
        print("Loading checkpoint from {} ...".format(load_path))

        return checkpoint['model_state_dict']
        # if isinstance(self.net, nn.DataParallel):
        #     self.net.module.load_state_dict(checkpoint['model_state_dict'])
        # else:
        #     self.net.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.clock.restore_checkpoint(checkpoint['clock'])

class InvSO3ConvModel(nn.Module):
    def __init__(self, num_anchors=60, pooling_x=False,  mlps=[[32,32], [64,64], [128,128]], out_mlps=[128, 128], strides=[2, 2, 2]):
        super(InvSO3ConvModel, self).__init__()
        params = self.get_parameters(num_anchors=num_anchors, mlps=mlps, out_mlps=out_mlps, strides=strides)
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.num_anchor = num_anchors
        self.invariance = True
        # self.config = config
        # self.t_method_type = config.t_method_type

        # if 'ycb' in config.task and config.instance is None:
        #     self.num_heads = config.DATASET.num_classes
        #     self.classifier = nn.Linear(params['outblock']['mlp'][-1], self.num_heads)
        # else:
        #     self.num_heads = 1
        #     self.classifier = None
        self.num_heads = 1
        self.classifier = None
        # per anchors R, T estimation
        # if config.t_method_type == -1:    # 0.845, R_i * delta_T
        #     self.outblockR = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method=config.model.pooling_method, pred_t=config.pred_t, feat_mode_num=self.na_in)
        # elif config.t_method_type == 0:   # 0.847, R_i0 * delta_T
        #     self.outblockRT = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                      pred_t=config.pred_t, feat_mode_num=self.na_in, num_heads=self.num_heads)
        # elif config.t_method_type == 1: # 0.8472,R_i0 * (xyz + Scalar*delta_T)_mean, current fastest
        #     self.outblockRT = M.SO3OutBlockRT(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                       global_scalar=True, use_anchors=False, feat_mode_num=self.na_in, num_heads=self.num_heads)
        # elif config.t_method_type == 2: # 0.8475,(xyz + R_i0 * Scalar*delta_T)_mean, current best
        #     self.outblockRT = M.SO3OutBlockRT(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                       global_scalar=True, use_anchors=True, feat_mode_num=self.na_in, num_heads=self.num_heads)
        # elif config.t_method_type == 3: # (xyz + R_i0 * delta_T)_mean
        #     self.outblockRT = M.SO3OutBlockRT(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                       feat_mode_num=self.na_in, num_heads=self.num_heads)
        self.outblockRT = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method='max',
                                              pred_t=True, feat_mode_num=self.num_anchor, num_heads=self.num_heads)

        # invariant feature for shape reconstruction
        self.outblockN = M.InvOutBlockOurs(params['outblock'], norm=1, pooling_method='max', pooling_x=pooling_x)
        # self.anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).cuda()
        self.anchors = torch.from_numpy(L.get_anchors(self.num_anchor)).cuda()
    
    def get_parameters(self, num_anchors=60,
                       mlps=[[32,32], [64,64], [128,128]],
                        out_mlps=[128, 128], # [64， 64]
                        strides=[2, 2, 2],
                        initial_radius_ratio = 0.2,
                        sampling_ratio = 0.8,
                        sampling_density = 0.5,
                        kernel_density = 1,
                        kernel_multiplier = 2,
                        sigma_ratio= 0.5, # 1e-3, 0.68
                        xyz_pooling = None, # None, 'no-stride'
                        to_file=None):

        input_num = 2048
        dropout_rate= 0.
        temperature= 3
        so3_pooling =  'rotation'
        input_radius = 0.4
        kpconv = False
        na = num_anchors

        # to accomodate different input_num
        if input_num > 1024:
            sampling_ratio /= (input_num / 1024)
            # strides[0] = int(2 * (input_num / 1024))
            print("Using sampling_ratio:", sampling_ratio)
            # print("Using strides:", strides)
        print("[MODEL] USING RADIUS AT %f"%input_radius)
        params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'na': na
              }
        dim_in = 1

        # process args
        n_layer = len(mlps)
        stride_current = 1
        stride_multipliers = [stride_current]
        for i in range(n_layer):
            stride_current *= 2
            stride_multipliers += [stride_current]

        num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]
        # print("num center", num_centers)

        radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]
        # print("radius_ratio", radius_ratio)

        # radius_ratio = [0.25, 0.5]
        radii = [r * input_radius for r in radius_ratio]

        weighted_sigma = [sigma_ratio * radii[0]**2]
        for idx, s in enumerate(strides):
            weighted_sigma.append(weighted_sigma[idx] * s)

        for i, block in enumerate(mlps):
            block_param = []
            for j, dim_out in enumerate(block):
                lazy_sample = i != 0 or j != 0

                stride_conv = i == 0 or xyz_pooling != 'stride'

                # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
                neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

                if i == 0 and j == 0:
                    neighbor *= int(input_num / 1024)

                kernel_size = 1
                if j == 0:
                    # stride at first (if applicable), enforced at first layer
                    inter_stride = strides[i]
                    nidx = i if i == 0 else i+1
                    # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                    if stride_conv:
                        neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                        kernel_size = 1 # if inter_stride < 4 else 3
                else:
                    inter_stride = 1
                    nidx = i+1

                # one-inter one-intra policy
                block_type = 'inter_block' if na != 60  else 'separable_block'

                inter_param = {
                    'type': block_type,
                    'args': {
                        'dim_in': dim_in,
                        'dim_out': dim_out,
                        'kernel_size': kernel_size,
                        'stride': inter_stride,
                        'radius': radii[nidx],
                        'sigma': weighted_sigma[nidx],
                        'n_neighbor': neighbor,
                        'lazy_sample': lazy_sample,
                        'dropout_rate': dropout_rate,
                        'multiplier': kernel_multiplier,
                        'activation': 'leaky_relu',
                        'pooling': xyz_pooling,
                        'kanchor': na,
                    }
                }
                block_param.append(inter_param)

                dim_in = dim_out

            params['backbone'].append(block_param)
        representation = 'quat'
        params['outblock'] = {
                'dim_in': dim_in,
                'mlp': out_mlps,
                'fc': [64],
                'k': 40,
                'kanchor': na,
                'pooling': so3_pooling,
                'representation': representation,
                'temperature': temperature,
        }

        return params

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        if x.shape[-1] > 3:
            x = x.permute(0, 2, 1).contiguous()
        x = M.preprocess_input(x, self.num_anchor, False)

        for block_i, block in enumerate(self.backbone):
            # print("x features shape is", x.feats.shape)
            # print("x coord shape is", x.xyz.shape)
            x = block(x)
        # print("x features shape is", x.feats.shape)
        # print("x coord shape is", x.xyz.shape)
        # if self.t_method_type < 0:
        #     output = self.outblockR(x, self.anchors)
        # else:
        #     output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        manifold_embed    = self.outblockN(x)
        output['0'] = manifold_embed
        output['xyz'] = x.xyz
        # output['xyz']     = x.xyz

        return  output

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


######################################## Grouper ########################################  
class POSE_Grouper(nn.Module):
    def __init__(self, k = 16, num_anchors=60, freeze_epn=True, anchors=None):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        self.num_anchors = num_anchors
        # self.knn = KNN(k=k, transpose_mode=False)
        self.freeze_epn = freeze_epn
        if self.freeze_epn == False:
            self.anchors = anchors
            # self.regitration_trans = InvSO3ConvModel(num_anchors = self.num_anchors)
            self.regitration_trans = EPN(num_anchors = self.num_anchors)
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.num_features = 128
        self.build_loss_func()
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x
    def fps_downsample_(self, xyz, num_group):
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(), fps_idx
            )
        )

        return new_xyz.transpose(1, 2).contiguous()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        # print("input coord shape", coor.shape)
        if self.freeze_epn == False:
            # input_pts = self.fps_downsample_(coor, 1024)
            input_pts = self.fps_downsample_(coor.transpose(-1, -2).contiguous(), 1024).transpose(-1, -2).contiguous()
            f = self.regitration_trans(input_pts)
            pred_R = f['R']
            pred_T = f['T']
            output_pts = f['recon'].permute(0, 2, 1).contiguous()
            # f = f['0']
            nb, nr, na = pred_R.shape
            out_dim = output_pts.shape[-2]
            rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
            translation = pred_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
            if na > 1:
                translation = torch.matmul(self.anchors, translation) # [na, 3, 3], [nb, na, 3, 1] --> [nb, na, 3, 1]
            qw, qxyz = torch.split(pred_R.permute(0, 2, 1).contiguous(), [1, 3], dim=-1)
            theta_max= torch.Tensor([36 / 180 * np.pi]).cuda()
            qw = torch.cos(theta_max) + (1 - torch.cos(theta_max)) * F.sigmoid(qw)
            constrained_quat = torch.cat([qw, qxyz], dim=-1)
            ranchor_pred = rotation_mapping(constrained_quat.view(-1, nr)).view(nb, -1, 3, 3)
            rotation = torch.matmul(self.anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]
            constrained_quat_tiled = constrained_quat.unsqueeze(2).contiguous().repeat(1, 1, out_dim, 1).contiguous() # nb, na, np, 4
            # canon_pts= pred_fine.contiguous() - 0.5 # nb, np, 3
            canon_pts = output_pts
            canon_pts -= 0.5
            canon_pts_tiled = canon_pts.unsqueeze(1).contiguous().repeat(1, na, 1, 1).contiguous() # nb, na, np, 3
            transformed_pts = rotate(constrained_quat_tiled, canon_pts_tiled) # nb, na, np, 3
            transformed_pts = torch.matmul(self.anchors, transformed_pts.permute(0, 1, 3, 2).contiguous()) + translation # nb, na, 3, np,
            transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous() # nb, na, np, 3
            distance_1, distance_2 = self.loss_func(transformed_pts.view(-1, out_dim, 3).contiguous(), input_pts.unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, out_dim, 3).contiguous(), return_raw=True)
            # print("the shape of distance_1", distance_1.shape)
            # print("the shape of distance_2", distance_2.shape)
            all_distance = (distance_2).mean(-1).view(nb, -1).contiguous()
            loss_regu_quat = torch.mean(torch.pow( torch.norm(constrained_quat, dim=-1) - 1, 2))
            loss_pose, min_indices = torch.min(all_distance, dim=-1) # we only allow one mode to be True
            r_pred = rotation[torch.arange(0, nb), min_indices].detach().clone() # correct R by searching
            t_pred = translation.squeeze(-1)[torch.arange(0, nb), min_indices]
            coor = torch.matmul(r_pred.permute(0, 2, 1).contiguous(), coor - t_pred.unsqueeze(-1).contiguous()).permute(0, 2, 1).contiguous() + 0.5 # registration the input pc
            coor = coor.transpose(-1, -2).contiguous()
        f = self.input_trans(coor)
        f = self.get_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        if coor.shape[-1] > num[0]:
            coor_q, f_q = self.fps_downsample(coor, f, num[0])
        else:
            coor_q, f_q = coor, f
        f = self.get_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)

        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        if coor.shape[-1] > num[1]:
            coor_q, f_q = self.fps_downsample(coor, f, num[1])
        else:
            coor_q, f_q = coor, f
        f = self.get_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        if self.freeze_epn == False:
            return coor, f, r_pred, t_pred, torch.mean(loss_pose), loss_regu_quat
        else:
            return coor, f

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class SimpleEncoder(nn.Module):
    def __init__(self, k = 32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k

        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1] 

        center = misc.fps(xyz, n_group) # B G 3
            
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'
        
        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
            
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
            
        features = self.embedding(neighborhood) # B G C
        
        return center, features

######################################## Fold ########################################    
class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

######################################## PCTransformer ########################################   
class PCTransformer(nn.Module):
    def __init__(self, config, anchors=None):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        self.freeze_epn = config.freeze_epn
        assert self.encoder_type in ['pose', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'pose':
            self.grouper = POSE_Grouper(k = 16, num_anchors=config.num_anchors, freeze_epn=self.freeze_epn, anchors=anchors)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        if self.freeze_epn == False:
            coor, f, pred_R, pred_T, loss_pose, loss_quad_regu = self.grouper(xyz, self.center_num) # b n c
        else:
            coor, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(coor)
        x = self.input_proj(f)

        x = self.encoder(x + pe, coor) # b n c
        global_feature = self.increase_dim(x) # B 1024 N 
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

        mem = self.mem_link(x)

        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64     

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            if self.freeze_epn == False:
                return q, coarse, denoise_length, pred_R, pred_T, loss_pose, loss_quad_regu
            else:
                return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            if self.freeze_epn == False:
                return q, coarse, 0, pred_R, pred_T, loss_pose, loss_quad_regu
            else:
                return q, coarse, 0

######################################## PCLCNet ########################################  

@MODELS.register_module()
class PCLCNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.freeze_epn = config.freeze_epn
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.num_anchors = config.num_anchors
        self.anchors = torch.from_numpy(L.get_anchors(self.num_anchors)).cuda()
        self.base_model = PCTransformer(config, self.anchors)
        
        if self.decoder_type == 'fold':
            self.factor = self.fold_step**2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step**2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()
        if self.freeze_epn == True: # use the pretrained EPN model
            self.epn = EPN()
            net_ckpt = self.epn.load_ckpt("best", "/data/xhm/equi-pose/experiment/model/pcn/1.0/checkpoints/") # "best_recon"
            if isinstance(self.epn, nn.DataParallel):
                self.epn.module.load_state_dict(net_ckpt)
            else:
                self.epn.load_state_dict(net_ckpt)
            # freeze the parameter
            for (name, param) in self.epn.named_parameters():
                if name in net_ckpt:
                    param.requires_grad = False
                else:
                    pass
            self.epn.eval()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
    
    def fps_downsample(self, xyz, num_group):
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(), fps_idx
            )
        )

        return new_xyz.transpose(1, 2).contiguous()

    def search_pose(self, pc, gt, pred_R, pred_T, shift_dis=None, xyz=None):
        # search the real R and T for in 60 anchors
        nb, nr, na = pred_R.shape
        bgt, ngt = gt.shape[0:2]
        if ngt > 1024:
            gt = self.fps_downsample(gt, 1024)
            ngt = 1024
        out_dim = pc.shape[-2]
        if out_dim > 1024:
            pc = self.fps_downsample(pc, 1024)
            out_dim = 1024
        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
        # ranchor_pred = rotation_mapping(pred_R.transpose(1,2).contiguous().view(-1, nr)).view(nb, -1, 3, 3)
        # rotation = torch.matmul(self.anchors, ranchor_pred) # [na, 3, 3], [nb, na, 3, 3] --> [nb, na, 3, 3]
        translation = pred_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
        if na > 1:
            translation = torch.matmul(self.anchors, translation) # [na, 3, 3], [nb, na, 3, 1] --> [nb, na, 3, 1]
        qw, qxyz = torch.split(pred_R.permute(0, 2, 1).contiguous(), [1, 3], dim=-1)
        theta_max= torch.Tensor([36 / 180 * np.pi]).cuda()
        qw = torch.cos(theta_max) + (1 - torch.cos(theta_max)) * F.sigmoid(qw)
        constrained_quat = torch.cat([qw, qxyz], dim=-1)
        ranchor_pred = rotation_mapping(constrained_quat.view(-1, nr)).view(nb, -1, 3, 3)
        rotation = torch.matmul(self.anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]
        constrained_quat_tiled = constrained_quat.unsqueeze(2).contiguous().repeat(1, 1, out_dim, 1).contiguous() # nb, na, np, 4

        # canon_pts= pred_fine.contiguous() - 0.5 # nb, np, 3
        canon_pts = pc
        canon_pts -= 0.5
        gt -= shift_dis
        # if self.freeze_epn == True:
            # canon_pts -= 0.5
            # gt -= shift_dis
        canon_pts_tiled = canon_pts.unsqueeze(1).contiguous().repeat(1, na, 1, 1).contiguous() # nb, na, np, 3
        transformed_pts = rotate(constrained_quat_tiled, canon_pts_tiled) # nb, na, np, 3
        transformed_pts = torch.matmul(self.anchors, transformed_pts.permute(0, 1, 3, 2).contiguous()) + translation # nb, na, 3, np,
        transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous() # nb, na, np, 3
        if xyz is not None:
            input_pts_tiled= (xyz - shift_dis).unsqueeze(1).repeat(1, na, 1, 1).contiguous() # nb, na, np, 3
            input_pts_tiled = torch.matmul(self.anchors.transpose(1,2).contiguous(), (input_pts_tiled.permute(0, 1, 3, 2).contiguous() - translation)) # nb, na, 3, np,
            input_pts_tiled = input_pts_tiled.permute(0, 1, 3, 2).contiguous() # nb, na, np, 3 
            registration_input_pts = reverse_rotate(constrained_quat.unsqueeze(2).contiguous().repeat(1, 1, input_pts_tiled.shape[-2], 1).contiguous(), input_pts_tiled) # nb, na, np, 3

        # print("the shape of transformed_pts", transformed_pts.shape)
        # print("the shape of gt", gt.shape)
        distance_1, distance_2 = self.loss_func(transformed_pts.view(-1, out_dim, 3).contiguous(), gt.unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, ngt, 3).contiguous(), return_raw=True)
        # print("the shape of distance_1", distance_1.shape)
        # print("the shape of distance_2", distance_2.shape)
        all_distance = (distance_2).mean(-1).view(nb, -1).contiguous()
        loss_regu_quat = torch.mean(torch.pow( torch.norm(constrained_quat, dim=-1) - 1, 2))
        if na > 1:
            _, min_indices = torch.min(all_distance, dim=-1) # we only allow one mode to be True
            # print("the min_indices", min_indices)
            if self.freeze_epn == True:
                transformed_pts = transformed_pts[torch.arange(0, bgt), min_indices] + shift_dis
                if xyz is not None:
                    registration_input_pts = registration_input_pts[torch.arange(0, bgt), min_indices] + 0.5
            else:
                transformed_pts = None
            r_pred = rotation[torch.arange(0, bgt), min_indices].detach().clone() # correct R by searching
            # print("The predeict r is:", r_pred)
            # r_pred.requires_grad = False
            t_pred = translation.squeeze(-1)[torch.arange(0, bgt), min_indices] + shift_dis.squeeze()
            # print("The predeict t is:", t_pred)
        else:
            r_pred = rotation.squeeze()
            t_pred = translation[:, 0, :, :].squeeze() + shift_dis.squeeze()
            transformed_pts = None

        if self.freeze_epn == True and xyz is not None:
            return r_pred, t_pred, transformed_pts, registration_input_pts
        else:
            return r_pred, t_pred, loss_regu_quat

    def transform_pc(self, pc, r, t):
        transformed_pc = torch.matmul(r, (pc - 0.5).permute(0, 2, 1).contiguous()) + t.unsqueeze(-1).contiguous()
        return transformed_pc.permute(0, 2, 1).contiguous()

    def get_loss(self, ret, gt, epoch=1):
        if self.freeze_epn == False:
            pred_coarse, denoised_coarse, denoised_fine, pred_fine, pred_R, pred_T, shift_dis, loss_pose, loss_regu_quat = ret
            # r, t, loss_regu_quat = self.search_pose(pred_fine, gt, pred_R, pred_T, shift_dis)
            # # use the searched real R and T to transform the pred_coarse, denoised_coarse, denoised_fine, pred_fine
            pred_coarse = self.transform_pc(pred_coarse, pred_R, pred_T + shift_dis.squeeze())
            pred_fine = self.transform_pc(pred_fine, pred_R, pred_T + shift_dis.squeeze())
            denoised_coarse = self.transform_pc(denoised_coarse, pred_R, pred_T + shift_dis.squeeze())
            denoised_fine = self.transform_pc(denoised_fine, pred_R, pred_T + shift_dis.squeeze())
        else:
            xyz, canno_recon, posed_recon, pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        # pose regularization loss
        if self.freeze_epn == False:
            loss_regu_quat = 0.1 * loss_regu_quat
            loss_pose = 0.5 * loss_pose
            return loss_denoised, loss_recon, loss_regu_quat, loss_pose
        else:
            return loss_denoised, loss_recon


    def forward(self, xyz):
        if self.freeze_epn == False:
            center_pt = (xyz.max(dim = 1, keepdim=True)[0] + xyz.min(dim = 1, keepdim=True)[0]) / 2
            xyz = xyz - center_pt
            shift_dis = xyz.mean(dim=1, keepdim=True)
            xyz = xyz - shift_dis
            shift_dis += center_pt
            q, coarse_point_cloud, denoise_length, pred_R, pred_T, loss_pose, loss_quad_regu = self.base_model(xyz) # B M C and B M 3
        else:
            input_pts = self.fps_downsample(xyz, 1024)
            # print(input_pts.shape)
            # print(input_pts.max(dim = 1, keepdim=True)[0].shape)
            # print(input_pts.min(dim = 1, keepdim=True)[0].shape)
            center_pt = (input_pts.max(dim = 1, keepdim=True)[0] + input_pts.min(dim = 1, keepdim=True)[0]) / 2
            input_pts = input_pts - center_pt
            shift_dis = input_pts.mean(dim=1, keepdim=True)
            input_pts -= shift_dis
            output = self.epn(input_pts)
            r, t, pose_recon, xyz = self.search_pose(output['recon'].permute(0, 2, 1).contiguous(), input_pts, output['R'], output['T'], shift_dis, xyz - center_pt)
            # print(t.unsqueeze(-1).contiguous().shape)
            pose_recon = pose_recon + center_pt
            t = t.unsqueeze(-1).contiguous()
            # xyz = torch.matmul(r.permute(0, 2, 1).contiguous(), (xyz - shift_dis - center_pt).permute(0, 2, 1).contiguous() - t).permute(0, 2, 1).contiguous() + 0.5 # registration the input pc
            
            q, coarse_point_cloud, denoise_length = self.base_model(xyz) # B M C and B M 3
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        # NOTE: foldingNet
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature) # B M C
            relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            if self.freeze_epn == False:
                # pred_coarse = self.transform_pc(pred_coarse, pred_R, pred_T + shift_dis)
                # pred_fine = self.transform_pc(pred_fine, pred_R, pred_T + shift_dis)
                # denoised_coarse = self.transform_pc(denoised_coarse, pred_R, pred_T + shift_dis)
                # denoised_fine = self.transform_pc(denoised_fine, pred_R, pred_T + shift_dis)
                ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine, pred_R, pred_T, shift_dis, loss_pose, loss_quad_regu)
            else:
                t = t + center_pt.permute(0, 2, 1).contiguous()
                pred_coarse = (torch.matmul(r, (pred_coarse - 0.5).permute(0, 2, 1).contiguous()) + t).permute(0, 2, 1).contiguous() + shift_dis
                pred_fine = (torch.matmul(r, (pred_fine - 0.5).permute(0, 2, 1).contiguous()) + t).permute(0, 2, 1).contiguous() + shift_dis
                denoised_coarse = (torch.matmul(r, (denoised_coarse - 0.5).permute(0, 2, 1).contiguous()) + t).permute(0, 2, 1).contiguous() + shift_dis
                denoised_fine = (torch.matmul(r, (denoised_fine - 0.5).permute(0, 2, 1).contiguous()) + t).permute(0, 2, 1).contiguous() + shift_dis
                ret = (xyz, output['recon'].permute(0, 2, 1).contiguous(), pose_recon, pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            if self.freeze_epn == False:
                # coarse_point_cloud = self.transform_pc(coarse_point_cloud, pred_R, pred_T + shift_dis)
                # rebuild_points = self.transform_pc(rebuild_points, pred_R, pred_T + shift_dis)
                ret = (coarse_point_cloud, rebuild_points, pred_R, pred_T, shift_dis, loss_pose, loss_quad_regu)
            else:
                t = t + center_pt.permute(0, 2, 1).contiguous()
                coarse_point_cloud = (torch.matmul(r, (coarse_point_cloud - 0.5).permute(0, 2, 1).contiguous()) + t).permute(0, 2, 1).contiguous() + shift_dis
                rebuild_points = (torch.matmul(r, (rebuild_points - 0.5).permute(0, 2, 1).contiguous()) + t).permute(0, 2, 1).contiguous() + shift_dis
                ret = (xyz, output['recon'].permute(0, 2, 1).contiguous(), pose_recon, coarse_point_cloud, rebuild_points)
            return ret