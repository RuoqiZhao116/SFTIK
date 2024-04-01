import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from .format import Format, nchw_to
from .helpers import to_2tuple
from .trace_utils import _assert

class CustomPatchEmbed(nn.Module):
    """ Custom 2D Image to Patch Embedding for vertical strips """
    def __init__(
            self,
            img_size: int = 224,
            patch_width: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)  # Ensure img_size is a tuple
        self.patch_size = (self.img_size[0], patch_width)  # Patch size (224,16)
        self.grid_size = (self.img_size[1] // patch_width, 1)  # Grid size in width, fixed height
        self.num_patches = self.grid_size[0]

        self.flatten = flatten
        self.output_fmt = Format(output_fmt) if output_fmt is not None else Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        # Convolution to create patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=(self.img_size[0],patch_width), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Checking image size
        if self.strict_img_size:
            _assert(H == self.img_size[0] and W == self.img_size[1], f"Input size doesn't match model ({self.img_size}).")
        elif not self.dynamic_img_pad:
            _assert(H == self.img_size[0], f"Input height ({H}) should match patch height ({self.img_size[0]}).")
            _assert(W % self.patch_size[0] == 0, f"Input width ({W}) should be divisible by patch width ({self.patch_size[0]}).")

        # Apply padding if necessary
        if self.dynamic_img_pad:
            pad_w = (self.patch_size[0] - W % self.patch_size[0]) % self.patch_size[0]
            x = F.pad(x, (0, pad_w, 0, 0))

        # Projecting to patches
        x = self.proj(x)
        
        # Flatten and transpose if necessary
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        
        # Apply normalization
        x = self.norm(x)
        return x
