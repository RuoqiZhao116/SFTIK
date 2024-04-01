from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet34()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        num_ftrs = self.resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, 256)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x  
    
class LSTMTimeSeries(nn.Module):
    def __init__(self, nvars, hidden_size, num_layers):
        super(LSTMTimeSeries, self).__init__()
        self.nvars = nvars
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=nvars, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.image_enocder = ImageEncoder()
        # Fully connected layer
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100)
        )
    def forward(self, pre_angle, pre_imu,image):
        # Initialize hidden state and cell state
        angle = pre_angle[:,0:1,:]
        x =  torch.cat((angle,pre_imu),axis=1)
        x = x.permute(0,2,1)   
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        img1 = torch.squeeze(image[:, 0:1, :, :, :], dim=1)  # First_image   # [bs x channels x width x height]
        img2 = torch.squeeze(image[:, 1:2, :, :, :], dim=1)  # Second_image

        img_feature_1 = self.image_enocder(img1)
        img_feature_2 = self.image_enocder(img2)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        combined_features = torch.cat((out[:, -1, :], img_feature_1, img_feature_2), dim=1)

        # Decode the hidden state of the last time step
        out = self.ffn(combined_features)

        return out    

