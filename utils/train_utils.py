import torch as tf
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import torch.optim as optim
import math
from .dataset import Terrain_Kinemics_Dataset

def cosine_anneling(epoch,T_max,eta_min):
    return eta_min + ( 1 - eta_min) * (1 + math.cos(math.pi*epoch / T_max)) / 2 

def cosine_growth(epoch,T_max,start_lr,end_lr):
    return start_lr + (end_lr - start_lr) * (1 - math.cos(math.pi * epoch / T_max))/2

def noam_lr_schedule(optimizer, d_model, warmup_steps=10):

    # Define lambda function for Noam learning rate schedule
    lmbda = lambda step: d_model**-0.5 * min(step**-0.5 if step > 0 else 1.0, step * warmup_steps**-1.5)

    # Use LambdaLR scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    return scheduler

def my_lr_schedule(optimizer, warmup_steps=20, steady_epoch = 150, start_ratio = 0.2,minima_ratio = 2e-3):

    # Define lambda function for Noam learning rate schedule
    lmbda = lambda epoch: cosine_growth(epoch,warmup_steps,start_ratio,1) if epoch < warmup_steps else cosine_anneling(epoch-warmup_steps,steady_epoch-warmup_steps,minima_ratio) if epoch < steady_epoch else minima_ratio

    # Use LambdaLR scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    return scheduler

def plot_max_mse_series(all_true, all_preds, labels_order):
    # Create the root directory
    root_dir = f"E:/Final_Project/logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for label in labels_order:
        if label not in all_true or label not in all_preds:
            print(f"Label {label} does not exist in the predictions or true values.")
            continue
        
        # Calculate MSE for each series and find the index of the max MSE
        mse_values = [mean_squared_error(all_true[label][i], all_preds[label][i]) for i in range(len(all_true[label]))]
        max_mse_idx = np.argmax(mse_values)
        
        # Extract the series with the max MSE
        true_series_max_mse = all_true[label][max_mse_idx]
        preds_series_max_mse = all_preds[label][max_mse_idx]
        
        # Plot the series
        plt.figure(figsize=(10, 6))
        plt.plot(true_series_max_mse, label='True Values', color=(70/255, 130/255, 180/255))
        plt.plot(preds_series_max_mse, label='Predictions', color=(223/255,127/255,88/255))
        plt.title(f"Label: {label} (Max MSE Series)")
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.ylim(-20,50)
        plt.legend()
        
        # Save the plot in the corresponding folder
        label_dir = os.path.join(root_dir)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        plt.savefig(os.path.join(label_dir, f"{label}_max_mse_series.png"))
        plt.close()  # Close the plot to free up memory


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.5, verbose = True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience  = patience
        self.min_lr    = min_lr
        self.factor    = factor
        self.verbose   = verbose
        
#         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer, 
#                                                                        steps     = self.patience, 
#                                                                        verbose   = self.verbose )
        self.lr_scheduler = tf.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode      = 'min',
                patience  = self.patience,
                factor    = self.factor,
                min_lr    = self.min_lr,
                verbose   = self.verbose 
            )
        
    def __call__(self, val_loss):
        self.lr_scheduler.step( val_loss )


class MultiEpochsDataLoader(tf.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def create_meta_dataloader(subjects, runs, sides, modality, dataset_path, 
                           test_subjects, val_subjects, 
                           train_batch_size=256, val_batch_size=32, n_workers=4):
    # 注意：我假设您的 Terrain_Kinemics_Dataset 需要 dataset_path, subject, run, side 和 modality 作为参数

    # 创建测试集数据加载器
    test_datasets = [Terrain_Kinemics_Dataset(dataset_path, subject, run, side, modality) 
                     for subject in test_subjects 
                     for run in runs 
                     for side in sides]
    test_dataset = ConcatDataset(test_datasets)
    test_loader = MultiEpochsDataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=n_workers)
    
    # 创建验证集数据加载器
    val_datasets = [Terrain_Kinemics_Dataset(dataset_path, subject, run, side, modality) 
                    for subject in val_subjects 
                    for run in runs 
                    for side in sides]
    val_dataset = ConcatDataset(val_datasets)
    val_loader = MultiEpochsDataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=n_workers)

    # 筛选出训练集中的主体
    train_subjects = [subject for subject in subjects if subject not in test_subjects and subject not in val_subjects]

    # 创建训练集数据加载器
    train_datasets = [Terrain_Kinemics_Dataset(dataset_path, subject, run, side, modality) 
                      for subject in train_subjects 
                      for run in runs 
                      for side in sides]
    train_dataset = ConcatDataset(train_datasets)
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=n_workers)

    return train_loader, val_loader, test_loader



def train_epoch(model, model_input, loss, optimizer, dataloader):
    
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Trainning", leave=True, disable=True):
        # Prepare the inputs based on model_input
        inputs = []
        for inp in model_input:
            tensor = batch[inp].float()
            if tf.cuda.is_available():
                tensor = tensor.cuda()
            inputs.append(tensor)

        optimizer.zero_grad()
        
        # Unpack the inputs based on their length and pass to the model
        x_recon = model(*inputs)

        x_recon = x_recon.squeeze(-1)
        cur_gait = batch['cur_angle'].float()
        if tf.cuda.is_available():
            cur_gait = cur_gait[:, 0:1, :].squeeze(1).cuda()
        
        loss_value = loss(cur_gait, x_recon)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item() * len(batch['cur_angle'])
        num_samples += len(batch['cur_angle'])

    avg_loss = total_loss / num_samples
    return avg_loss


def validation_epoch(model, model_input, loss, dataloader):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    num_samples = 0

    with tf.no_grad():  # No need to track gradients
        for batch in tqdm(dataloader, desc="Validation", leave=True, disable=True):
            # Prepare the inputs based on model_input
            inputs = []
            for inp in model_input:
                tensor = batch[inp].float()
                if tf.cuda.is_available():
                    tensor = tensor.cuda()
                inputs.append(tensor)

            # Unpack the inputs based on their length and pass to the model
            x_recon = model(*inputs)

            x_recon = x_recon.squeeze(-1)
            cur_gait = batch['cur_angle'].float()
            if tf.cuda.is_available():
                cur_gait = cur_gait[:, 0:1, :].squeeze(1).cuda()

            loss_value = loss(cur_gait, x_recon)
            total_loss += loss_value.item() * len(batch['cur_angle'])
            num_samples += len(batch['cur_angle'])

    avg_loss = total_loss / num_samples
    return avg_loss

def calculate_metrics_and_save(data, csv_path):
    results = []
    for (subject_id, label), data in data.items():
        rmses = []
        for true, pred in zip(data['true'], data['pred']):
            if len(true) > 0 and len(pred) > 0:
                rmse = np.sqrt(mean_squared_error(true, pred))
                rmses.append(rmse)
        print(f'Terrain {label},with rmse num {len(rmses)}')
        avg_rmse = np.nanmean(rmses) if rmses else np.nan

        correlations = [np.corrcoef(true, pred)[0, 1] for true, pred in zip(data['true'], data['pred']) if len(true) > 1 and len(pred) > 1]
        avg_pcc = np.mean(correlations)

        results.append({'Subject': subject_id, 'Label': label, 'RMSE': avg_rmse, 'Average PCC': avg_pcc})

    df = pd.DataFrame(results)

    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, mode='w', index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

    return df

def inverse_scale_angle_from_unit_range(scaled_value):
    angle_radians = np.arcsin(2 * scaled_value - 1)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def test_whole_model(model, model_input, dataloader, csv_path):
    model.eval()
    all_data = {}

    with tf.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True, disable=True):
            inputs = [batch[inp].float().cuda() if tf.cuda.is_available() else batch[inp].float() for inp in model_input]
            predictions = model(*inputs).squeeze(-1).clamp(0, 1).cpu().numpy()  # [batch_size, target_window]

            true_values = batch['cur_angle'].cpu().numpy()  # [batch_size, target_window]

            true_angles = inverse_scale_angle_from_unit_range(true_values[:, 0, :]) # 背部通道 [:, 1, :] 
            pre_angles = inverse_scale_angle_from_unit_range(predictions)
            subject_ids = batch['subject_id']
            labels = batch['label']

            # 为每个通道分别处理和存储数据
            for subject_id, label, true_angle, pred_angle in zip(subject_ids, labels, true_angles, pre_angles):
                key = (subject_id, label)
                
                # 大腿数据
                if key not in all_data:
                    all_data[key] = {'true': [], 'pred': []}
                all_data[key]['true'].append(true_angle)
                all_data[key]['pred'].append(pred_angle)



    df_thigh = calculate_metrics_and_save(all_data,f'{csv_path}_thigh.csv')

    return df_thigh