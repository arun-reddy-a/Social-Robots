import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.layers.fno_block import FNOBlocks

from utils.gaussian_random_field import *
from layers.conditional_embedding import Condition_embedding
from neuralop.models import UNO

torch.manual_seed(0)
np.random.seed(0)


class cGenerator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, out_channels, n_feat, modes, domain_padding, pad = 0, factor = 1):
        super(cGenerator_HM, self).__init__()

        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.out_channels=out_channels
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain, hidden_channels=d_co_domain, out_channels=self.out_channels, lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, n_layers=6,
                        uno_out_channels=[int(1.34*factor*self.width), 
                        int(1.30*factor*self.width), 
                        int(1.25*factor*self.width), 
                        int(1.25*factor*self.width), 
                        int(1.25*factor*self.width), 
                        int(self.width)],
                        uno_n_modes=[[self.modes], [self.modes], [self.modes], [self.modes], [self.modes], [self.modes]],
                        norm = 'instance_norm',
                        increment_n_modes = [[50], [50], [50], [50], [50], [50]],
                        uno_scalings=[[1], [1], [1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)

    def forward(self, x,c):
        
        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        
        
        x = torch.cat((x, fourier_feats, c), dim=-1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)

        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_fourier_features(self, shape, device, n_feat):
        freq = np.linspace(0, 1, n_feat)
        batchsize, size_x = shape[0], shape[1]
        fourier_features = torch.zeros(batchsize, size_x, 2*n_feat)
        for i in range(n_feat):
            fourier_features[:, :, 2*i] = torch.tensor(2*np.pi*freq[i]*np.sin(np.linspace(0, 1, size_x)), dtype=torch.float)
            fourier_features[:, :, 2*i+1] = torch.tensor(2*np.pi*freq[i]*np.cos(np.linspace(0, 1, size_x)), dtype=torch.float)
        return fourier_features.to(device)


class small_cGenerator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, out_channels, n_feat, modes, domain_padding, pad = 0, factor = 1):
        super(small_cGenerator_HM, self).__init__()

        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain//2 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.out_channels=out_channels
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain, hidden_channels=d_co_domain//2, out_channels=self.out_channels, lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, n_layers=4,
                        uno_out_channels=[int((1/2*factor)*factor*self.width), 
                        int((1/2*factor)*factor*self.width), 
                        int((1/2*factor)*factor*self.width), 
                        int(self.width)],
                        uno_n_modes=[[self.modes], [self.modes], [self.modes], [self.modes]],
                        norm = 'instance_norm',
                        increment_n_modes = [[50], [50], [50], [50]],
                        uno_scalings=[[1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)

    def forward(self, x,c):
        
        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        
        
        x = torch.cat((x, fourier_feats, c), dim=-1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)

        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_fourier_features(self, shape, device, n_feat):
        freq = np.linspace(0, 1, n_feat)*20
        freq[0] = 1
        batchsize, size_x = shape[0], shape[1]
        fourier_features = torch.zeros(batchsize, size_x, 2*n_feat)
        for i in range(n_feat):
            fourier_features[:, :, 2*i] = torch.tensor(2*np.pi*freq[i]*np.sin(np.linspace(0, 1, size_x)*20), dtype=torch.float)
            fourier_features[:, :, 2*i+1] = torch.tensor(2*np.pi*freq[i]*np.cos(np.linspace(0, 1, size_x)*20), dtype=torch.float)
        return fourier_features.to(device)


class mid_cGenerator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, out_channels, n_feat, modes, domain_padding, pad = 0, factor = 1):
        super(mid_cGenerator_HM, self).__init__()

        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain//2 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.out_channels=out_channels
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain, hidden_channels=d_co_domain//2, out_channels=self.out_channels, lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, n_layers=5,
                        uno_out_channels=[int((1/2*factor)*factor*self.width), 
                        int((1/2*factor)*factor*self.width), 
                        int((1/2*factor)*factor*self.width),
                        int((1/2*factor)*factor*self.width), 
                        int(self.width)],
                        uno_n_modes=[[self.modes], [self.modes], [self.modes], [self.modes], [self.modes]],
                        norm = 'instance_norm',
                        increment_n_modes = [[50], [50], [50], [50], [50]],
                        uno_scalings=[[1], [1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)

    def forward(self, x,c):
        
        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        
        
        x = torch.cat((x, fourier_feats, c), dim=-1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)

        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_fourier_features(self, shape, device, n_feat):
        freq = np.linspace(0, 1, n_feat)*20
        freq[0] = 1
        batchsize, size_x = shape[0], shape[1]
        fourier_features = torch.zeros(batchsize, size_x, 2*n_feat)
        for i in range(n_feat):
            fourier_features[:, :, 2*i] = torch.tensor(2*np.pi*freq[i]*np.sin(np.linspace(0, 1, size_x)*20), dtype=torch.float)
            fourier_features[:, :, 2*i+1] = torch.tensor(2*np.pi*freq[i]*np.cos(np.linspace(0, 1, size_x)*20), dtype=torch.float)
        return fourier_features.to(device)

class test_cGenerator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, out_channels, n_feat, modes, domain_padding, pad = 0, factor = 1):
        super(test_cGenerator_HM, self).__init__()

        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.out_channels=out_channels
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain, hidden_channels=d_co_domain, out_channels=self.out_channels, lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, n_layers=6,
                        uno_out_channels=[int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(self.width)],
                        uno_n_modes=[[self.modes], [self.modes], [self.modes], [self.modes], [self.modes], [self.modes]],
                        norm = 'instance_norm',
                        increment_n_modes = [[50], [50], [50], [50], [50], [50]],
                        uno_scalings=[[1], [1], [1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)

    def forward(self, x,c):
        
        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        
        
        x = torch.cat((x, fourier_feats, c), dim=-1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)

        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_fourier_features(self, shape, device, n_feat):
        freq = np.linspace(0, 1, n_feat)
        batchsize, size_x = shape[0], shape[1]
        fourier_features = torch.zeros(batchsize, size_x, 2*n_feat)
        for i in range(n_feat):
            fourier_features[:, :, 2*i] = torch.tensor(2*np.pi*freq[i]*np.sin(np.linspace(0, 1, size_x)), dtype=torch.float)
            fourier_features[:, :, 2*i+1] = torch.tensor(2*np.pi*freq[i]*np.cos(np.linspace(0, 1, size_x)), dtype=torch.float)
        return fourier_features.to(device)