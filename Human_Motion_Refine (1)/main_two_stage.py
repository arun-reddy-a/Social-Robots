import numpy as np
import pickle
import os
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.gaussian_random_field import *
from utils.visualizations import *
from models.generator import *
from models.discriminator import *
from train_two_stage import train_GANO_HM_two_stage
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.dataloader import DD100
from torch.optim.lr_scheduler import CosineAnnealingLR


# put the following data loader coder in data/data_loader.py
# Todo
# 1. use relative espression to express the human body pose

if __name__ == "__main__":

    random.seed(9)

    ### loading all hyperparameters from config file
    # to run this code run  
    # $python main.py --config cnn

    with open('config/config_3.yaml', "r") as f:
        config = yaml.safe_load(f)

    # Loading datasets, the data is normalized and in angle-magnitude form.

    with open(config['train_data'], 'rb') as f:
        train_data = pickle.load(f) # (T, 242) 

    with open(config['test_data'], 'rb') as f:
        test_data = pickle.load(f) # (T, 242)

    for key, path in config['logging'].items():
    # Check if the path is a directory or a file
        if not os.path.exists(path):
            # Create directories for all paths, including those containing files
            os.makedirs(path if path.endswith('/') else os.path.dirname(path), exist_ok=True)
    
    print(type(config['lr']))

    train_dataset = DD100(train_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    len_val_data = int(config['val_data_percent']*len(test_data))

    val_data = random.sample(test_data, len_val_data)

    val_dataset = DD100(val_data)

    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = config['device']

    grf = Gaussian_RF(1,alpha=config['alpha'], tau=config['tau'], codim = config['codim'], device=config['device'] )

    D_transl = small_cDiscriminator_HM(config['transl_dim_info']+config['transl_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], pad=config['npad']).to(config['device']).float()
    D_pose = cDiscriminator_HM(config['pose_dim_info']+config['pose_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], pad=config['npad']).to(config['device']).float()

    G_transl = small_cGenerator_HM(config['codim']+config['transl_dim_info']+2*config['n_feat'], config['d_co_domain'], config['transl_dim_info'], config['n_feat'], config['modes'], pad=config['npad']).to(config['device']).float()
    G_pose = cGenerator_HM(config['codim']+config['pose_dim_info']+2*config['n_feat'], config['d_co_domain'], config['pose_dim_info'], config['n_feat'], config['modes'], pad=config['npad']).to(config['device']).float()


    print('G transl parameters ', count_parameters(G_transl)[0])
    print('G poses parameters ', count_parameters(G_pose)[0])

    print('D transl parameters ', count_parameters(D_transl)[0])
    print('D poses parameters ', count_parameters(D_pose)[0])

    num_epochs = 2000

    G_transl_optimizer = torch.optim.Adam(G_transl.parameters(), lr=config['lr']) 
    G_transl_scheduler = CosineAnnealingLR(G_transl_optimizer, T_max=num_epochs)

    G_pose_optimizer = torch.optim.Adam(G_pose.parameters(), lr=config['lr']) 
    G_pose_scheduler = CosineAnnealingLR(G_pose_optimizer, T_max=num_epochs)

    D_transl_optimizer = torch.optim.Adam(D_transl.parameters(), lr=config['lr'])
    D_transl_scheduler = CosineAnnealingLR(D_transl_optimizer, T_max=num_epochs)

    D_pose_optimizer = torch.optim.Adam(D_pose.parameters(), lr=config['lr'])
    D_pose_scheduler = CosineAnnealingLR(D_pose_optimizer, T_max=num_epochs)

    D_transl.train()
    D_pose.train()

    G_transl.train()
    G_pose.train()

    # instead of passing all the paramters separately just pass the params, access the parameters from the params object
    losses_D_transl, losses_G_transl, losses_W_transl, losses_MSE_transl, losses_V_MSE_transl, \
    losses_D_poses, losses_G_poses, losses_W_poses, losses_MSE_poses, losses_V_MSE_poses, \
    losses_MSE_full, losses_V_MSE_full = train_GANO_HM_two_stage(
        D_transl, D_pose, G_transl, G_pose, train_dataloader, val_dataloader, 
        config['epochs'], D_transl_optimizer, D_pose_optimizer, 
        G_transl_optimizer, G_pose_optimizer, config['λ_grad'], grf, 
        config['n_critic'], config['device'], config['batch_size'], 
        config['logging'], config['hm_weight'], config['fc_weight'], 
        config['gt_weight'], D_transl_scheduler, D_pose_scheduler, G_transl_scheduler, G_pose_scheduler,
        data_stats=None, scheduler=None
)

