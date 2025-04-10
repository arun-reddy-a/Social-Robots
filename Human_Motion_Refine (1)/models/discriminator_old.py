import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.layers.fno_block import FNOBlocks

from utils.gaussian_random_field import *
from layers.conditional_embedding import Condition_embedding
from models.knet import kernel

torch.manual_seed(0)
np.random.seed(0)


## Todo
# 1. We might wanna use UNO in the middle
# 2. use the Fourier Feature Embedding (available in the neuralop library)


class cDiscriminator_HM_OLD(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain,modes, kernel_dim=16, pad = 0, factor = 3/4):
        super(cDiscriminator_HM_OLD, self).__init__()

        self.kernel_dim = kernel_dim
        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain 
        self.factor = factor
        self.padding = pad
        self.modes = modes


        self.fc = nn.Linear(self.in_width, self.width//2)
        
        self.fc_1 = nn.Linear(self.width//2, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        #self.L0 = OperatorBlock_1D(self.width, 2*factor*self.width,48, 15)
        
        self.L0 = FNOBlocks(in_channels=self.width, out_channels=int(2*factor*self.width), n_modes=(self.modes))
        
        #self.L1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 32,9, Normalize = True)
        self.L1 = FNOBlocks(in_channels=int(2*factor*self.width), out_channels=int(4*factor*self.width),
                        n_modes=(self.modes), norm='instance_norm')

        #self.L2 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 32,9)
        self.L2 = FNOBlocks(in_channels=int(4*factor*self.width), out_channels=int(4*factor*self.width), n_modes=(self.modes))
        
        #self.L3 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 32,9, Normalize = True)
        self.L3 = FNOBlocks(in_channels=int(4*factor*self.width), out_channels=int(4*factor*self.width),
                                 n_modes=(self.modes), norm='instance_norm')
        
        #self.L4 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 32,9)
        self.L4 = FNOBlocks(in_channels=int(4*factor*self.width), out_channels=int(4*factor*self.width), n_modes=(self.modes))

        #self.L5 = OperatorBlock_1D(8*factor*self.width, 2*factor*self.width, 48,9, Normalize = True)
        self.L5 = FNOBlocks(in_channels=int(8*factor*self.width), out_channels=int(2*factor*self.width),
                             n_modes=(self.modes), norm='instance_norm')

        #self.L6 = OperatorBlock_1D(4*factor*self.width, self.width, 64,15) # will be reshaped
        self.L6 = FNOBlocks(in_channels=int(4*factor*self.width), out_channels=self.width, n_modes=(self.modes))

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, self.kernel_dim)
        
        # kernel for last functional operation

        self.knet = kernel(1, self.kernel_dim)


    def forward(self, x,c):
        print('the shape of input x and c is ', x.shape, c.shape)
        
        grid = self.get_grid(x.shape, x.device)
     
        x = torch.cat((x, grid, c), dim=-1)
         
        print('the shape after concatenation of x, grid, c is ', x.shape)

        
        # The Below is for lifting  
        x_fc = self.fc(x)

        x_fc = F.gelu(x_fc)
        x_fc_1 = self.fc_1(x_fc)
        x_fc_1 = F.gelu(x_fc_1)
        

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        # Until here
        
        print('the shape after lifting is ', x_fc0.shape)

        x_fc0 = x_fc0.permute(0, 2, 1)
        
        padding = int((x_fc0.shape[-1]/30)*self.padding)
        
        x_fc0 = F.pad(x_fc0, [0,padding])
        
        D1 = x_fc0.shape[-1]

        print('the shape after padding is and D1 is ', x_fc0.shape, D1)
        
        x_c0 = self.L0(x_fc0, output_shape = (int(D1*self.factor), ))

        print('the shape after 0th layer is ', x_c0.shape)
        
        x_c1 = self.L1(x_c0 , output_shape = (D1//2,))

        print('the shape after 1st layer is ', x_c1.shape)

        x_c2 = self.L2(x_c1, output_shape = (D1//2,))

        print('the shape after 2nd layer is ', x_c2.shape)
                
        x_c3 = self.L3(x_c2, output_shape = (D1//2,))

        print('the shape after 3rd layer is ', x_c3.shape)
        
        x_c4 = self.L4(x_c3, output_shape = (D1//2,))

        print('the shape after 4th layer is ', x_c4.shape)
        
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        
        x_c5 = self.L5(x_c4, output_shape = (int(D1*self.factor), ))

        print('the shape after 5th layer is ', x_c5.shape)
        
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        
        x_c6 = self.L6(x_c5, output_shape = (int(D1), ))

        print('the shape after 6th layer is ', x_c6.shape)
        
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)
        

        if self.padding!=0:
            x_c6 = x_c6[..., :-padding]

        x_c6 = x_c6.permute(0, 2, 1)

        print('the shape after fno layers is ', x_c6.shape)
        
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)

        print('the shape after downsampling is ', x_out.shape)
        
        res1 = x_out.shape[1]
        grid_ker = self.get_grid(x_out.shape, x_out.device)

        kx = self.knet(grid_ker)

        kx = kx.view(x.shape[0],-1, 1)
        print('here it is ', x_out.shape)
        x_out = x_out.view(x.shape[0],-1, 1)


        x_out = torch.einsum('bik,bik->bk', kx, x_out)/(res1)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    