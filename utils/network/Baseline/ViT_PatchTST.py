
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ...layers.pos_encoding import *
from ...layers.basics import *
from ...layers.attention import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [1, grid_size*grid_size, embed_dim] or [1, 1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.reshape(1, grid_size * grid_size, embed_dim)  # Reshape to [1, grid_size*grid_size, embed_dim]

    if cls_token:
        cls_token_embed = np.zeros([1, 1, embed_dim])  # Adding cls token at the beginning
        pos_embed = np.concatenate([cls_token_embed, pos_embed], axis=1)

    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class ViTEncoder(nn.Module):
    """
    out: (Batchsize, 2, emd_dim) color and depth
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        
        super().__init__()

        #self.patch_embed_color = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_embed_depth = PatchEmbed(img_size, patch_size, 1, embed_dim)

        num_patches = self.patch_embed_depth.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, 14, cls_token=True)).float(), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, img):
        # embed patches
        #x = img[:, :3, :, :]
        depth = img

        #x = self.patch_embed_color(x)
        x_depth = self.patch_embed_depth(depth)

        # add pos embed w/o cls token
        #x = x + self.pos_embed[:, 1:, :]
        x_depth = x_depth + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        # append cls token

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_depth.shape[0], -1, -1)

        #x = torch.cat((cls_tokens, x), dim=1)
        x_depth = torch.cat((cls_tokens, x_depth), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            #x = blk(x)
            x_depth = blk(x_depth)
        #x = self.norm(x)
        x_depth = self.norm(x_depth)

        #x = torch.mean(x, dim=1, keepdim=True)
        x_depth = torch.mean(x_depth, dim=1, keepdim=True)

        return x_depth

          
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self,  c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "pretrain", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        # Backbone
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((context_window - patch_len)/stride + 1)
        self.context_window = context_window                                
        self.backbone = PatchTSTEncoder(c_in, num_patch=self.patch_num, patch_len=self.patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.image_encoder = ViTEncoder()

        self.n_vars = c_in
        self.head_type = head_type
        self.hidden_dim = 512
        self.fc_ts = nn.Linear(self.n_vars * 128 * self.patch_num, self.hidden_dim)
        self.fc_img = nn.Linear(768, self.hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=3)
        self.fc_out = nn.Linear(3 * self.hidden_dim, 100)
        self.dropout = nn.Dropout(0.2)

    def forward(self, pre_angle,pre_imu, image):
        angle = pre_angle[:,0:1,:]
        z = torch.cat((angle,pre_imu),axis=1)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,2,1,3)

        img1 = torch.squeeze(image[:, 0:1, :, :, :], dim=1)  # Input_image
        img2 = torch.squeeze(image[:, 1:2, :, :, :], dim=1)  # Second channel
        batch_size = img1.shape[0]

        feature1 = self.image_encoder(img1).view(batch_size, -1)
        feature2 = self.image_encoder(img2).view(batch_size, -1)

        img_feature1 = self.fc_img(feature1)
        img_feature2 = self.fc_img(feature2)
        
        img_feature1 = self.dropout(F.relu(img_feature1))
        img_feature2 = self.dropout(F.relu(img_feature2))

        ts_feature = self.backbone(z)                                  # z: [bs x nvars x d_model x num_patch]
        bs, nvars, d_model, num_patch = ts_feature.shape
        flat_len = nvars * d_model * num_patch
        ts_feature = ts_feature.view(bs, flat_len)
        ts_feature = self.fc_ts(ts_feature)
        ts_feature = self.dropout(F.relu(ts_feature))
        
        combined_feature = torch.stack((ts_feature, img_feature1, img_feature2), dim=1)
        transformed_feature = self.transformer_encoder(combined_feature).view(batch_size, -1)

        output = self.fc_out(transformed_feature)       

        return output



class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0, dropout=0, act="gelu", store_attn=False, ##需要调整Dropout
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]

        return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

