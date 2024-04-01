import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from PIL import Image

def scale_angle_to_unit_range(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    scaled_angle = (np.sin(angle_radians) + 1) / 2
    
    return scaled_angle

def depth_to_distance(depth, scale_factor):
    return depth * scale_factor

def depth_transform(image_path, scale_factor=1e-3, min_dist=0, max_dist=5):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_distance = depth_to_distance(img, scale_factor)
    img_clamped = np.clip(img_distance, min_dist, max_dist)
    img_scaled = img_clamped / max_dist
    img_resized = cv2.resize(img_scaled, (224, 224))
    return  torch.from_numpy(img_resized).unsqueeze(0)


class Terrain_Kinemics_Dataset(Dataset):
    '''
    This dataset provide the comprehensive info.
    '''
    def __init__(self, dataset_path,subject, run, side, modality):
        assert side in ['left', 'right'], "Side should be either 'left' or 'right'"
        assert modality in ['RGB_D', 'RGB', 'Depth'], "Modality should be 'RGB_D', 'RGB', or 'Depth'"

        self.base_path = os.path.join(dataset_path,subject, run)
        self.dataframe = pd.read_csv(os.path.join(self.base_path, f"meta_{side}.csv"))

        self.color_folder = os.path.join(self.base_path, "color")
        self.depth_folder = os.path.join(self.base_path, "depth")
        self.modality = modality

        self.color_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.subject = subject

    def __len__(self):
        return len(self.dataframe)

    def normalize_data_fixed_length(self, data, target_length=100):
        current_length = len(data)
        interpolated_time = np.linspace(0, current_length-1, target_length)
        return np.interp(interpolated_time, np.arange(current_length), data)
    
    def get_acc_data(self,flag,idx,df):
        prefixes = [f'{flag}_Left', f'{flag}_Right', f'{flag}_Back']
        suffixes = ['ACC_X', 'ACC_Y', 'ACC_Z']
        arrays = []
        for prefix in prefixes:
            for suffix in suffixes:
                column_name = f'{prefix}_{suffix}'
                # 从第一行提取数据，假设每个单元格是逗号分隔的字符串
                array = self.normalize_data_fixed_length(np.array(list(map(float, df.loc[idx, column_name].strip('[]').split(',')))))/16
                arrays.append(array)
        return np.stack(arrays, axis=0)
    
    def get_gyro_data(self,flag,idx,df):
        prefixes = [f'{flag}_Left', f'{flag}_Right', f'{flag}_Back']
        suffixes = ['GYRO_X', 'GYRO_Y', 'GYRO_Z']
        arrays = []
        for prefix in prefixes:
            for suffix in suffixes:
                column_name = f'{prefix}_{suffix}'
                # 从第一行提取数据，假设每个单元格是逗号分隔的字符串
                array = self.normalize_data_fixed_length(np.array(list(map(float, df.loc[idx, column_name].strip('[]').split(',')))))/2000
                arrays.append(array)
        return np.stack(arrays, axis=0)
    
    def __getitem__(self, idx):
        #获取不同模态的图像
        if self.modality == 'RGB_D':
            color_img1 = self.color_transform(Image.open(os.path.join(self.color_folder, self.dataframe.iloc[idx, 0])).convert('RGB'))
            color_img2 = self.color_transform(Image.open(os.path.join(self.color_folder, self.dataframe.iloc[idx, 1])).convert('RGB'))

            depth_img1 = depth_transform(os.path.join(self.depth_folder, self.dataframe.iloc[idx, 0]))
            depth_img2 = depth_transform(os.path.join(self.depth_folder, self.dataframe.iloc[idx, 1]))
            
            combined_img1 = torch.cat((color_img1, depth_img1), 0)
            combined_img2 = torch.cat((color_img2, depth_img2), 0)

            # Return both color and depth images
            stacked_images =  torch.stack([combined_img1, combined_img2], 0)

        elif self.modality == 'RGB':
            color_img1 = self.color_transform(Image.open(os.path.join(self.color_folder, self.dataframe.iloc[idx, 0])).convert('RGB'))
            color_img2 = self.color_transform(Image.open(os.path.join(self.color_folder, self.dataframe.iloc[idx, 1])).convert('RGB'))

            # Return both color and depth images
            stacked_images =  torch.stack([color_img1, color_img2], 0)
        
        elif self.modality == 'Depth':
            depth_img1 = depth_transform(os.path.join(self.depth_folder, self.dataframe.iloc[idx, 0]))
            depth_img2 = depth_transform(os.path.join(self.depth_folder, self.dataframe.iloc[idx, 1]))

            # Return both color and depth images
            stacked_images =  torch.stack([depth_img1, depth_img2], 0)

        label_value = self.dataframe.iloc[idx, 2]

        # Convert ANGLE_X data from comma-separated string to a numpy array
        pre_thigh_angle = self.normalize_data_fixed_length(scale_angle_to_unit_range(np.array(list(map(float, self.dataframe.loc[idx, 'Pre_Thigh_Angle'].strip('[]').split(','))))))
        pre_back_angle = self.normalize_data_fixed_length(scale_angle_to_unit_range(np.array(list(map(float, self.dataframe.loc[idx, 'Pre_Back_Angle'].strip('[]').split(','))))))
        pre_angle = np.stack((pre_thigh_angle, pre_back_angle), axis=0)

        cur_thigh_angle = self.normalize_data_fixed_length(scale_angle_to_unit_range(np.array(list(map(float, self.dataframe.loc[idx, 'Cur_Thigh_Angle'].strip('[]').split(','))))))
        cur_back_angle = self.normalize_data_fixed_length(scale_angle_to_unit_range(np.array(list(map(float, self.dataframe.loc[idx, 'Cur_Back_Angle'].strip('[]').split(','))))))
        cur_angle = np.stack((cur_thigh_angle, cur_back_angle), axis=0)

        pre_acc = self.get_acc_data('Pre',idx,self.dataframe)
        pre_gyro = self.get_gyro_data('Pre',idx,self.dataframe)
        pre_imu = np.concatenate((pre_acc,pre_gyro),axis=0)

        cur_acc = self.get_acc_data('Cur',idx,self.dataframe)
        cur_gyro = self.get_gyro_data('Cur',idx,self.dataframe)
        cur_imu = np.concatenate((cur_acc,cur_gyro),axis=0)

        sample = {'image': stacked_images, 'label': label_value,'pre_angle':pre_angle,'cur_angle':cur_angle,
                  'pre_imu':pre_imu,'cur_imu':cur_imu,'subject_id':self.subject}


        return sample