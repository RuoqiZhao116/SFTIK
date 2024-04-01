import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..layers.pos_encoding import *
from ..layers.basics import *
from ..layers.attention import *
          
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

        self.img_patch = PatchEmbed(img_size = 224, patch_size = 16, in_chans = 1, embed_dim = embed_dim)
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
        
        self.mix_patch_nums = 2 * self.img_patch.num_patches +  self.ts_patch_num

        # Output Head
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100)
        )

    def forward(self, pre_angle,pre_imu, image):

        # IMU Time Series Embedding with position encoding
        thigh_angle = pre_angle[:,0:1,:]
        z = torch.cat((thigh_angle,pre_imu),axis=1)                                          # z: [bs x nvars x len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,2,1,3)                                                             # z: [bs x patch_num x nvars x patch_len]
        bs, patch_num, _, _ = z.size()
        z_flattened = z.reshape(bs, patch_num, -1)                                         # z: [bs x patch_num x nvars*patch_len]
        z = self.W_P(z_flattened)                                                          # z: [bs x patch_num x emd_dim]
        ts_patches = z + self.ts_pos_embed

        # Img Embedding with position encoding
        img1 = torch.squeeze(image[:, 0:1, :, :, :], dim=1)  # First_image   # [bs x channels x width x height]
        img2 = torch.squeeze(image[:, 1:2, :, :, :], dim=1)  # Second_image
        img1_patches = self.img_patch(img1) + self.img_pos_embed              # [bs x patch_num x emd_dim]
        img2_patches = self.img_patch(img2) + self.img_pos_embed

        # Early Fusion
        mix_patches = torch.cat([ts_patches, img1_patches], dim=1)
    
        for blk in self.former_blocks:
            mix_patches = blk(mix_patches)
        for blk in self.single_blocks:
            img2_patches = blk(img2_patches)

        fuse_patches = torch.cat([mix_patches,img2_patches], dim=1)
        for blk in self.later_blocks:
            fuse_patches = blk(fuse_patches)
        pooled = torch.mean(fuse_patches, dim=1)
        
        output = self.ffn(pooled)

        return output

