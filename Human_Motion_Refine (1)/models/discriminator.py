import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.layers.fno_block import FNOBlocks

from utils.gaussian_random_field import *
from layers.conditional_embedding import Condition_embedding
from models.knet import kernel
from neuralop.models import UNO

torch.manual_seed(0)
np.random.seed(0)

## Todo
# 1. We might wanna use UNO in the middle
# 2. use the Fourier Feature Embedding (available in the neuralop library)


class cDiscriminator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, n_feat,modes, domain_padding, kernel_dim=16, pad = 0, factor = 3/4):
        super(cDiscriminator_HM, self).__init__()

        self.kernel_dim = kernel_dim
        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain,
                        hidden_channels=d_co_domain,
                        out_channels=self.kernel_dim, 
                        lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, 
                        n_layers=6,
                        uno_out_channels=[int(1.34*factor*self.width), 
                        int(1.30*factor*self.width), 
                        int(1.25*factor*self.width), 
                        int(1.25*factor*self.width), 
                        int(1.25*factor*self.width), 
                        int(self.width)],
                        uno_n_modes=[[self.modes], 
                        [self.modes],
                        [self.modes], 
                        [self.modes], 
                        [self.modes], 
                        [self.modes]],
                        increment_n_modes=[[50], [50], [50], [50], [50], [50]],
                        norm='instance_norm',
                        uno_scalings=[[1], [1], [1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)
        
        # kernel for last functional operation

        self.knet = kernel(1, self.kernel_dim)


    def forward(self, x,c):

        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        x = torch.cat((x, fourier_feats, c), dim=-1) # (N, 300, 165+165+1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)


        res1 = x_out.shape[1]
        grid_ker = self.get_grid(x_out.shape, x_out.device)

        kx = self.knet(grid_ker)

        kx = kx.view(x.shape[0],-1, 1)

        x_out = x_out.contiguous().view(x.shape[0],-1, 1)

        x_out = torch.einsum('bik,bik->bk', kx, x_out)/(res1)

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


class small_cDiscriminator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, n_feat,modes, domain_padding, kernel_dim=16, pad = 0, factor = 0.5):
        super(small_cDiscriminator_HM, self).__init__()

        self.kernel_dim = kernel_dim # 16
        self.in_width = in_d_co_domain # 16
        self.width = d_co_domain # 150
        self.factor = factor # 0.5
        self.padding = pad # 5
        self.modes = modes # 300
        self.n_feat = n_feat # 5

        print(in_d_co_domain, self.kernel_dim, self.width)
        self.uno = UNO(
                        in_channels=in_d_co_domain,
                        out_channels=kernel_dim,
                        hidden_channels=d_co_domain,
                        lifting_channels=self.width,
                        projection_channels=self.width,
                        n_layers=2,
                        uno_out_channels=[self.width, self.width],
                        uno_n_modes=[[modes], [modes]],
                        norm='instance_norm',
                        increment_n_modes=[[50], [50]],
                        uno_scalings=[[1], [1]],
                        domain_padding=domain_padding,
                        skip='linear',  # Important to avoid soft-gating error
                        channel_mlp_skip='linear'
                    )
        
        # kernel for last functional operation
        print('uno model created')
        self.knet = kernel(1, self.kernel_dim)


    def forward(self, x,c):
        print('hiii')
        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        x = torch.cat((x, fourier_feats, c), dim=-1) # (N, 300, 165+165+1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)


        res1 = x_out.shape[1]
        grid_ker = self.get_grid(x_out.shape, x_out.device)

        kx = self.knet(grid_ker)

        kx = kx.view(x.shape[0],-1, 1)

        x_out = x_out.contiguous().view(x.shape[0],-1, 1)

        x_out = torch.einsum('bik,bik->bk', kx, x_out)/(res1)

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


class mid_cDiscriminator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, n_feat,modes, domain_padding, kernel_dim=16, pad = 0, factor = 3/4):
        super(mid_cDiscriminator_HM, self).__init__()

        self.kernel_dim = kernel_dim
        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain//2 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain,
                        hidden_channels=d_co_domain//2,
                        out_channels=self.kernel_dim, 
                        lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, 
                        n_layers=5,
                        uno_out_channels=[int((1/2*factor)*factor*self.width), 
                        int((1/2*factor)*factor*self.width),  
                        int((1/2*factor)*factor*self.width),
                        int((1/2*factor)*factor*self.width),  
                        int(self.width)],
                        uno_n_modes=[[self.modes], 
                        [self.modes],
                        [self.modes], 
                        [self.modes], 
                        [self.modes]],
                        norm = 'instance_norm',
                        increment_n_modes = [[50], [50], [50], [50], [50]],
                        uno_scalings=[[1], [1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)
        
        # kernel for last functional operation

        self.knet = kernel(1, self.kernel_dim)


    def forward(self, x,c):

        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        x = torch.cat((x, fourier_feats, c), dim=-1) # (N, 300, 165+165+1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)


        res1 = x_out.shape[1]
        grid_ker = self.get_grid(x_out.shape, x_out.device)

        kx = self.knet(grid_ker)

        kx = kx.view(x.shape[0],-1, 1)

        x_out = x_out.contiguous().view(x.shape[0],-1, 1)

        x_out = torch.einsum('bik,bik->bk', kx, x_out)/(res1)

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

class test_cDiscriminator_HM(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, n_feat,modes, domain_padding, kernel_dim=16, pad = 0, factor = 3/4):
        super(test_cDiscriminator_HM, self).__init__()

        self.kernel_dim = kernel_dim
        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain 
        self.factor = factor
        self.padding = pad
        self.modes = modes
        self.n_feat = n_feat


        self.uno = UNO(in_channels=in_d_co_domain,
                        hidden_channels=d_co_domain,
                        out_channels=self.kernel_dim, 
                        lifting_channels=d_co_domain//2,
                        projection_channels=2*self.width, 
                        n_layers=6,
                        uno_out_channels=[int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(0.45*factor*self.width), 
                        int(self.width)],
                        uno_n_modes=[[self.modes], 
                        [self.modes],
                        [self.modes], 
                        [self.modes], 
                        [self.modes], 
                        [self.modes]],
                        norm = 'instance_norm',
                        increment_n_modes = [[50], [50], [50], [50], [50], [50]],
                        uno_scalings=[[1], [1], [1], [1], [1], [1]], horizontal_skips_map={7:0, 6:1, 5:2},
                        domain_padding=domain_padding)
        
        # kernel for last functional operation

        self.knet = kernel(1, self.kernel_dim)


    def forward(self, x,c):

        fourier_feats = self.get_fourier_features(x.shape, x.device, self.n_feat)

        x = torch.cat((x, fourier_feats, c), dim=-1) # (N, 300, 165+165+1)

        x = x.permute(0,2,1)

        x_out = self.uno(x)

        x_out = x_out.permute(0, 2, 1)


        res1 = x_out.shape[1]
        grid_ker = self.get_grid(x_out.shape, x_out.device)

        kx = self.knet(grid_ker)

        kx = kx.view(x.shape[0],-1, 1)

        x_out = x_out.contiguous().view(x.shape[0],-1, 1)

        x_out = torch.einsum('bik,bik->bk', kx, x_out)/(res1)

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