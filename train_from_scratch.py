import torch
import numpy as np
from utils.loss import mse_loss
from utils.train_utils import create_meta_dataloader,train_epoch,validation_epoch,test_whole_model,my_lr_schedule
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from utils.network.SFTIK import SFTIK
# The baseline model could be imported from utils.network.Baseline.
import argparse
import random
seed = 226
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(): 
    # 5-fold Leave-one out cross validatoin (LOO-CV)
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
    version = 'SFTIK' # The model for training
    base_path = '/home/usr/SFTIK' # Change to your own path
    
    for fold in folds:
        print(f'Start Training{fold["test"]}',flush=True)
        # Extract test and validation subjects for the current fold
        test_subjects = fold['test']
        val_subjects = fold['validate']
    
        formatted_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = os.path.join(f'{base_path}/model',f"{version}_{formatted_time}_" + 'val_' + ''.join(val_subjects) + '_test_' +  ''.join(test_subjects) + ".pth")
        best_model_filename = os.path.join(f'{base_path}/model',f"{version}_{formatted_time}_best_" + 'val_' + ''.join(val_subjects) + '_test_' +  ''.join(test_subjects) + ".pth")

        writer = SummaryWriter()

        train_loader, val_loader, test_loader = create_meta_dataloader(subjects=subjects, runs=runs, sides=sides, modality='Depth', dataset_path = f'{base_path}/dataset',
                                                                test_subjects=test_subjects, val_subjects=val_subjects, train_batch_size = 32, val_batch_size = 32)

        model = SFTIK(c_in = 19, context_window = 100, target_window = 100, patch_len = 10, stride = 10, 
                         embed_dim = 768, n_heads = 12, pre_depth = 6, late_depth = 6)
        

        model_input = ['pre_angle','pre_imu','image']
        device = torch.device("cuda:0")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
        
        scheduler = my_lr_schedule(optimizer, warmup_steps = 50, steady_epoch = 180, start_ratio = 0.2,minima_ratio = 0.02)
        best_val_loss = float('inf')
        num_epochs = 200

        for epoch in range(num_epochs):
            print(f'{test_subjects} Current epoch {epoch}/{num_epochs}',flush=True)
            train_loss = train_epoch(model=model,model_input=model_input,loss=mse_loss,optimizer=optimizer,dataloader=train_loader)
            val_loss = validation_epoch(model=model,model_input=model_input,loss=mse_loss,dataloader=val_loader)

            torch.save(model.state_dict(), model_filename)

            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_filename)
            # Log the average losses to TensorBoard
            writer.add_scalar('Training Loss', train_loss, epoch)
            writer.add_scalar('Validation Loss', val_loss, epoch)

        writer.close()

        print("Test Performance.")
        model.load_state_dict(torch.load(best_model_filename))
        test_whole_model(model = model, model_input = model_input, dataloader= test_loader,csv_path = f'{base_path}/results/{version}')
        
        del train_loader,test_loader,val_loader
        
if __name__ == '__main__':
    main()