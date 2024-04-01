from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
import torchvision.models as models

class ImageEncoder_MobileNet(nn.Module):
    def __init__(self):
        super(ImageEncoder_MobileNet, self).__init__()
        # Load the pre-trained MobileNet model
        self.mobilenet = models.mobilenet_v2(weights = None)

        # If your input images have 1 channel, modify the first convolution layer
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Remove the last classification layer
        self.features = self.mobilenet.features

        # Get the number of features from the last channel of MobileNet
        # Typically, for MobileNetV2, it's 1280, but it's a good practice to retrieve it programmatically
        num_ftrs = self.mobilenet.last_channel

        # Define a new fully connected layer for your specific task
        self.fc = nn.Linear(num_ftrs, 128)

    def forward(self, x):
        x = self.features(x)
        # Global average pooling
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(19*100, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 128)     # Second fully connected layer
        self.img_encoder = ImageEncoder_MobileNet()

        self.ffn = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 100)
        )

    def forward(self, pre_angle, pre_imu, image):
        # Initialize hidden state and cell state
        thigh_angle = pre_angle[:,0:1,:]
        x =  torch.cat((thigh_angle,pre_imu),axis=1)

        x = x.reshape(x.size(0), -1)

        # Apply the first fully connected layer and a ReLU
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        img1 = torch.squeeze(image[:, 0:1, :, :, :], dim=1)  # First_image   # [bs x channels x width x height]
        img2 = torch.squeeze(image[:, 1:2, :, :, :], dim=1)  # Second_image
        img_feature_1 = self.img_encoder(img1)
        img_feature_2 = self.img_encoder(img2)
        fuse_feature = torch.cat((img_feature_1, img_feature_2, x), dim=1)

        out = self.ffn(fuse_feature)

        return out 

