from functools import partial

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
from ..layers.CustomPatch_height import CustomPatchEmbed
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..layers.pos_encoding import *
from ..layers.basics import *
from ..layers.attention import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer
          
class SFTIK(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self,  c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,
                 embed_dim = 768, pre_depth = 6, late_depth = 6, n_heads = 12, mlp_ratio=4., norm_layer=nn.LayerNorm, **kwargs):

        super().__init__()

        # Backbone
        self.patch_len = patch_len
        self.stride = stride
        self.context_window = context_window

        self.ts_patch_num = int((context_window - patch_len)/stride + 1)
        self.W_P = nn.Linear(c_in * patch_len, embed_dim)
        self.ts_pos_embed = nn.Parameter(torch.zeros(1, self.ts_patch_num, embed_dim))

        self.img_patch = CustomPatchEmbed(patch_width=16,in_chans = 1,embed_dim=embed_dim)
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.img_patch.num_patches, embed_dim))

        self.former_blocks = nn.ModuleList([
                Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(pre_depth)])
        
        self.single_blocks = nn.ModuleList([
                Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(pre_depth)])
        
        self.later_blocks = nn.ModuleList([
                Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(late_depth)])
        
        self.mix_patch_nums = self.img_patch.num_patches +  self.ts_patch_num

        # Output Head
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100)
        )

    def forward(self, pre_angle,pre_imu, image):

        img2 = torch.squeeze(image[:, 1:2, :, :, :], dim=1)  # Second_image
        img2_patches = self.img_patch(img2) + self.img_pos_embed

        for blk in self.single_blocks:
            img2_patches = blk(img2_patches)

        fuse_patches = img2_patches
        for blk in self.later_blocks:
            fuse_patches = blk(fuse_patches)
        pooled = torch.mean(fuse_patches, dim=1)
        
        output = self.ffn(pooled)

        return output

