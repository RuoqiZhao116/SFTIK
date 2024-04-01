import torch
import numpy as np
from utils.loss import mse_loss
from utils.train_utils import create_meta_dataloader,train_epoch,validation_epoch,test_whole_model,noam_lr_schedule,LRScheduler,my_lr_schedule

import os
# import the model settings
from utils.network.SFTIK import SFTIK
from utils.network.Baseline.MobileNet_MLP import SimpleMLP
from utils.network.Baseline.ViT_PatchTST import PatchTST
from utils.network.Baseline.LSTM_ResNet import LSTMTimeSeries

import random
seed = 226
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(): 
    folds = [
    {'test': ['S08', 'S09'], 'validate': ['S02']},
    {'test': ['S04', 'S06'], 'validate': ['S03']},
    {'test': ['S02', 'S07'], 'validate': ['S09']},
    {'test': ['S03', 'S05'], 'validate': ['S04']},
    {'test': ['S01', 'S10'], 'validate': ['S05']},
    ]

    runs = ['01', '02', '03']
    sides = ['left','right']
    subjects = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10']
    # 'MobileNet_MLP'  'ResNet_LSTM' 'SFTIK' 'ViT_PatchTST'
    version = 'SFTIK'
    base_path = '/home/usr/SFTIK' # Change to your own path
    result_path = f"{base_path}/logs/{version}"

    for fold in folds:
        print(f'Start Testing On Subject {fold["test"]}',flush=True)
        test_subjects = fold['test']
        val_subjects = fold['validate']

        model_filename = os.path.join(f'{base_path}/weights/{version}/{version}' + '_val_' + ''.join(val_subjects) + '_test_' +  ''.join(test_subjects) + ".pth")

        #model = SFTIK(c_in = 19, context_window = 100, target_window = 100, patch_len = 10, stride = 10, embed_dim = 768, n_heads = 12, pre_depth = 6, late_depth = 6 )
        #model = LSTMTimeSeries(nvars = 19, hidden_size = 256, num_layers = 3)
        model = PatchTST(c_in = 19, context_window = 100, target_window = 100, patch_len = 10, stride = 10)
        #model = SimpleMLP()

        model.load_state_dict(torch.load(model_filename))
        
        _, _, test_loader = create_meta_dataloader(subjects=subjects, runs=runs, sides=sides, modality='Depth', dataset_path=f'{base_path}/dataset',
                                                                test_subjects=test_subjects, val_subjects=val_subjects, train_batch_size = 32, val_batch_size = 32)


        model_input = ['pre_angle','pre_imu','image']
        device = torch.device("cuda:0")
        model.to(device)
        
        test_whole_model(model=model, model_input=model_input, dataloader=test_loader,csv_path=result_path)

        del test_loader

        
if __name__ == '__main__':
    main()