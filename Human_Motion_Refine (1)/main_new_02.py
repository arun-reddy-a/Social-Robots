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
from train_two_stage_copy_copy_copy_copy_copy import train_GANO_HM_two_stage_copy_copy_copy_copy_copy
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.dataloader import DD100
from torch.optim.lr_scheduler import CosineAnnealingLR



# put the following data loader coder in data/data_loader.py
# Todo
# 1. use relative espression to express the human body pose

if __name__ == "__main__":

    ### loading all hyperparameters from config file
    # to run this code run  
    # $python main.py --config cnn

    with open('config/config_new_2.yaml', "r") as f:
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
    print('len of train data is ', len(train_data))
    print('len of test data is ', len(test_data))
    train_data = train_data[:1]
    test_data = train_data
    train_dataset = DD100(train_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    len_val_data = int(config['val_data_percent']*len(test_data))
    val_data = random.sample(test_data, len_val_data)

    val_dataset = DD100(val_data)
    print('len of val_dataset is ', len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    print(len(val_dataloader))

    device = config['device']

    grf = Gaussian_RF(1, config['res'], alpha=config['alpha'], tau=config['tau'], codim = config['codim'], device=config['device'] )

    D_move = small_cDiscriminator_HM(config['move_dim_info']+config['move_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    D_orient = small_cDiscriminator_HM(config['orient_dim_info']+config['orient_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    D_betas = small_cDiscriminator_HM(config['betas_dim_info']+config['betas_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    D_hands = test_cDiscriminator_HM(config['hands_dim_info']+config['hands_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    D_legs = mid_cDiscriminator_HM(config['legs_dim_info']+config['legs_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    D_torso = mid_cDiscriminator_HM(config['torso_dim_info']+config['torso_dim_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()

    G_move = small_cGenerator_HM(config['codim']+config['move_dim_info']+2*config['n_feat'], config['d_co_domain'], config['move_dim_info'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    G_orient = small_cGenerator_HM(config['codim']+config['orient_dim_info']+2*config['n_feat'], config['d_co_domain'], config['orient_dim_info'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    G_betas = small_cGenerator_HM(config['codim']+config['betas_dim_info']+2*config['n_feat'], config['d_co_domain'], config['betas_dim_info'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    G_hands = test_cGenerator_HM(config['codim']+config['hands_dim_info']+2*config['n_feat'], config['d_co_domain'], config['hands_dim_info'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    G_legs = mid_cGenerator_HM(config['codim']+config['legs_dim_info']+2*config['n_feat'], config['d_co_domain'], config['legs_dim_info'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()
    G_torso = mid_cGenerator_HM(config['codim']+config['torso_dim_info']+2*config['n_feat'], config['d_co_domain'], config['torso_dim_info'], config['n_feat'], config['modes'], config['domain_padding'],pad=config['npad']).to(config['device']).float()


    print('G move parameters ', count_parameters(G_move)[0])
    print('G orient parameters ', count_parameters(G_orient)[0])
    print('G betas parameters ', count_parameters(G_betas)[0])
    print('G hands parameters ', count_parameters(G_hands)[0])
    print('G legs parameters ', count_parameters(G_legs)[0])
    print('G torso parameters ', count_parameters(G_torso)[0])

    print('D move parameters ', count_parameters(D_move)[0])
    print('D orient parameters ', count_parameters(D_orient)[0])
    print('D betas parameters ', count_parameters(D_betas)[0])
    print('D hands parameters ', count_parameters(D_hands)[0])
    print('D legs parameters ', count_parameters(D_legs)[0])
    print('D torso parameters ', count_parameters(D_torso)[0])

    num_epochs = config['epochs']


    G_move_optimizer = torch.optim.Adam(G_move.parameters(), lr=config['lr'])
    G_move_scheduler = CosineAnnealingLR(G_move_optimizer, T_max=num_epochs)

    G_orient_optimizer = torch.optim.Adam(G_orient.parameters(), lr=config['lr'])
    G_orient_scheduler = CosineAnnealingLR(G_orient_optimizer, T_max=num_epochs)

    G_betas_optimizer = torch.optim.Adam(G_betas.parameters(), lr=config['lr'])
    G_betas_scheduler = CosineAnnealingLR(G_betas_optimizer, T_max=num_epochs)

    G_hands_optimizer = torch.optim.Adam(G_hands.parameters(), lr=config['lr'])
    G_hands_scheduler = CosineAnnealingLR(G_hands_optimizer, T_max=num_epochs)

    G_legs_optimizer = torch.optim.Adam(G_legs.parameters(), lr=config['lr'])
    G_legs_scheduler = CosineAnnealingLR(G_legs_optimizer, T_max=num_epochs)

    G_torso_optimizer = torch.optim.Adam(G_torso.parameters(), lr=config['lr'])
    G_torso_scheduler = CosineAnnealingLR(G_torso_optimizer, T_max=num_epochs)

    D_move_optimizer = torch.optim.Adam(D_move.parameters(), lr=config['lr'])
    D_move_scheduler = CosineAnnealingLR(D_move_optimizer, T_max=num_epochs)

    D_orient_optimizer = torch.optim.Adam(D_orient.parameters(), lr=config['lr'])
    D_orient_scheduler = CosineAnnealingLR(D_orient_optimizer, T_max=num_epochs)

    D_betas_optimizer = torch.optim.Adam(D_betas.parameters(), lr=config['lr'])
    D_betas_scheduler = CosineAnnealingLR(D_betas_optimizer, T_max=num_epochs)

    D_hands_optimizer = torch.optim.Adam(D_hands.parameters(), lr=config['lr'])
    D_hands_scheduler = CosineAnnealingLR(D_hands_optimizer, T_max=num_epochs)

    D_legs_optimizer = torch.optim.Adam(D_legs.parameters(), lr=config['lr'])
    D_legs_scheduler = CosineAnnealingLR(D_legs_optimizer, T_max=num_epochs)

    D_torso_optimizer = torch.optim.Adam(D_torso.parameters(), lr=config['lr'])
    D_torso_scheduler = CosineAnnealingLR(D_torso_optimizer, T_max=num_epochs)

    if bool(config['resume']):
        print('Resuming training')
        # Load the model
        logging = config['logging']
        i = config['resume_epoch']

        G_move.load_state_dict(torch.load(logging['G_move_save_path']+f"generator_move_epoch{i}.pt"))
        G_orient.load_state_dict(torch.load(logging['G_orient_save_path']+f"generator_orient_epoch{i}.pt"))
        G_betas.load_state_dict(torch.load(logging['G_betas_save_path']+f"generator_betas_epoch{i}.pt"))
        G_hands.load_state_dict(torch.load(logging['G_hands_save_path']+f"generator_hands_epoch{i}.pt"))
        G_legs.load_state_dict(torch.load(logging['G_legs_save_path']+f"generator_legs_epoch{i}.pt"))
        G_torso.load_state_dict(torch.load(logging['G_torso_save_path']+f"generator_torso_epoch{i}.pt"))

        D_move.load_state_dict(torch.load(logging['D_move_save_path']+f"discriminator_move_epoch{i}.pt"))
        D_orient.load_state_dict(torch.load(logging['D_orient_save_path']+f"discriminator_orient_epoch{i}.pt"))
        D_betas.load_state_dict(torch.load(logging['D_betas_save_path']+f"discriminator_betas_epoch{i}.pt"))
        D_hands.load_state_dict(torch.load(logging['D_hands_save_path']+f"discriminator_hands_epoch{i}.pt"))
        D_legs.load_state_dict(torch.load(logging['D_legs_save_path']+f"discriminator_legs_epoch{i}.pt"))
        D_torso.load_state_dict(torch.load(logging['D_torso_save_path']+f"discriminator_torso_epoch{i}.pt"))

        # Load the optimizer
        G_move_optimizer.load_state_dict(torch.load(logging['G_move_optimizer_save_path']+f"epoch{i}.pt"))
        G_orient_optimizer.load_state_dict(torch.load(logging['G_orient_optimizer_save_path']+f"epoch{i}.pt"))
        G_betas_optimizer.load_state_dict(torch.load(logging['G_betas_optimizer_save_path']+f"epoch{i}.pt"))
        G_hands_optimizer.load_state_dict(torch.load(logging['G_hands_optimizer_save_path']+f"epoch{i}.pt"))
        G_legs_optimizer.load_state_dict(torch.load(logging['G_legs_optimizer_save_path']+f"epoch{i}.pt"))
        G_torso_optimizer.load_state_dict(torch.load(logging['G_torso_optimizer_save_path']+f"epoch{i}.pt"))

        D_move_optimizer.load_state_dict(torch.load(logging['D_move_optimizer_save_path']+f"epoch{i}.pt"))
        D_orient_optimizer.load_state_dict(torch.load(logging['D_orient_optimizer_save_path']+f"epoch{i}.pt"))
        D_betas_optimizer.load_state_dict(torch.load(logging['D_betas_optimizer_save_path']+f"epoch{i}.pt"))
        D_hands_optimizer.load_state_dict(torch.load(logging['D_hands_optimizer_save_path']+f"epoch{i}.pt"))
        D_legs_optimizer.load_state_dict(torch.load(logging['D_legs_optimizer_save_path']+f"epoch{i}.pt"))
        D_torso_optimizer.load_state_dict(torch.load(logging['D_torso_optimizer_save_path']+f"epoch{i}.pt"))

        # Load the scheduler
        G_move_scheduler.load_state_dict(torch.load(logging['G_move_scheduler_save_path']+f"epoch{i}.pt"))
        G_orient_scheduler.load_state_dict(torch.load(logging['G_orient_scheduler_save_path']+f"epoch{i}.pt"))
        G_betas_scheduler.load_state_dict(torch.load(logging['G_betas_scheduler_save_path']+f"epoch{i}.pt"))
        G_hands_scheduler.load_state_dict(torch.load(logging['G_hands_scheduler_save_path']+f"epoch{i}.pt"))
        G_legs_scheduler.load_state_dict(torch.load(logging['G_legs_scheduler_save_path']+f"epoch{i}.pt"))
        G_torso_scheduler.load_state_dict(torch.load(logging['G_torso_scheduler_save_path']+f"epoch{i}.pt"))

        D_move_scheduler.load_state_dict(torch.load(logging['D_move_scheduler_save_path']+f"epoch{i}.pt"))
        D_orient_scheduler.load_state_dict(torch.load(logging['D_orient_scheduler_save_path']+f"epoch{i}.pt"))
        D_betas_scheduler.load_state_dict(torch.load(logging['D_betas_scheduler_save_path']+f"epoch{i}.pt"))
        D_hands_scheduler.load_state_dict(torch.load(logging['D_hands_scheduler_save_path']+f"epoch{i}.pt"))
        D_legs_scheduler.load_state_dict(torch.load(logging['D_legs_scheduler_save_path']+f"epoch{i}.pt"))
        D_torso_scheduler.load_state_dict(torch.load(logging['D_torso_scheduler_save_path']+f"epoch{i}.pt"))
        

    D_move.train()
    D_orient.train()
    D_betas.train()
    D_hands.train()
    D_legs.train()
    D_torso.train()

    G_move.train()
    G_orient.train()
    G_betas.train()
    G_hands.train()
    G_legs.train()
    G_torso.train()

    final_result = train_GANO_HM_two_stage_copy_copy_copy_copy_copy(D_move, D_orient, D_betas, D_hands, D_legs, D_torso,
                                                G_move, G_orient, G_betas, G_hands, G_legs, G_torso,
                                                train_dataloader, val_dataloader, config['epochs'], D_move_optimizer, D_orient_optimizer, D_betas_optimizer, D_hands_optimizer, D_legs_optimizer, D_torso_optimizer,
                                                G_move_optimizer, G_orient_optimizer, G_betas_optimizer, G_hands_optimizer, G_legs_optimizer, G_torso_optimizer,
                                                config['λ_grad'], grf, config['n_critic'], config['device'], config['batch_size'], config['logging'], config['hm_weight'],
                                                config['fc_weight'], config['gt_weight'], D_move_scheduler, D_orient_scheduler, D_betas_scheduler, D_hands_scheduler, D_legs_scheduler, D_torso_scheduler,
                                                G_move_scheduler, G_orient_scheduler, G_betas_scheduler, G_hands_scheduler, G_legs_scheduler, G_torso_scheduler, config['resume_epoch'], data_stats=None, scheduler=None)

    # save the final result
    with open(config['logging']['final_result_path']+'final_result.pkl', 'wb') as f:
        pickle.dump(final_result, f)

