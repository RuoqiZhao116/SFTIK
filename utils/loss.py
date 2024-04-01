import torch as tf
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(x, x_recon):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    return recon_loss