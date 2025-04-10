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
from train import train_GANO_HM
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.dataloader import DD100


# put the following data loader coder in data/data_loader.py
# Todo
# 1. use relative espression to express the human body pose

if __name__ == "__main__":

    ### loading all hyperparameters from config file
    # to run this code run  
    # $python main.py --config cnn

    with open('config/config_2.yaml', "r") as f:
        config = yaml.safe_load(f)

    with open(config['train_data_path'], 'rb') as f:
        train_data = pickle.load(f)

    with open(config['test_data_path'], 'rb') as f:
        test_data = pickle.load(f)
    
    print(type(config['lr']))

    train_dataset = DD100(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    test_dataset = DD100(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

     
    len_val_data = int(config['val_data_percent']*len(test_data))
    val_data = random.sample(test_data, len_val_data)

    val_dataset = DD100(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    
    device = config['device']

    grf = Gaussian_RF(1,alpha=config['alpha'], tau=config['tau'], codim = config['codim'], device=config['device'] )

    D = cDiscriminator_HM(config['pose_info']+config['pose_info']+2*config['n_feat'], config['d_co_domain'], config['n_feat'], config['modes'], pad=config['npad']).to(config['device']).float()
    G = cGenerator_HM(config['codim']+config['pose_info']+2*config['n_feat'], config['d_co_domain'], config['pose_info'], config['n_feat'], config['modes'], pad=config['npad']).to(config['device']).float()

    x = torch.rand((10, 300, 120)).to(config['device'])
    c = torch.rand((10, 300, 187)).to(config['device'])
    gen_out = G(x, c)
    print(gen_out.shape)

    x = torch.rand((10, 300, 187)).to(config['device'])
    c = torch.rand((10, 300, 187)).to(config['device'])
    dis_out = D(x, c)
    print(dis_out.shape)

    gen_total, gen_trainable = count_parameters(G)
    dis_total, dis_trainable = count_parameters(D)

    print('gen_params',gen_total, gen_trainable)
    print('dis_params',dis_total, dis_trainable)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=config['lr']) 
    D_optimizer = torch.optim.Adam(D.parameters(), lr=config['lr'])
    D.train()
    G.train()

    # instead of passing all the paramters separately just pass the params, access the parameters from the params object
    losses_D, losses_G, losses_W, losses_MSE, loss_V_MSE  = train_GANO_HM(D, G, train_dataloader, val_dataloader, config['epochs'], D_optimizer, G_optimizer,\
        config['λ_grad'], grf, config['n_critic'], config['device'], config['batch_size'], config['logging'], data_stats = None, scheduler=None)
