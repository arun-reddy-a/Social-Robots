import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.layers.fno_block import FNOBlocks

from utils.gaussian_random_field import *
from layers.conditional_embedding import Condition_embedding

torch.manual_seed(0)
np.random.seed(0)


class cGenerator_HM_OLD(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain,modes, pad = 0, factor = 1):
        super(cGenerator_HM_OLD, self).__init__()

        self.in_width = in_d_co_domain # input channel
        self.width = d_co_domain 
        self.factor = factor
        self.padding = pad
        self.modes = modes


        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        #self.L0 = OperatorBlock_1D(self.width, 2*factor*self.width,48, 15)
       
        self.L0 = FNOBlocks(in_channels=self.width, out_channels=2*factor*self.width, n_modes=(self.modes))

        #self.L1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 20,10, Normalize = True)
        self.L1 = FNOBlocks(in_channels=2*factor*self.width, out_channels=4*factor*self.width,
                        n_modes=(self.modes), norm='instance_norm')

        
        #self.L2 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 20,10)
        self.L2 = FNOBlocks(in_channels=4*factor*self.width, out_channels=4*factor*self.width, n_modes=(self.modes))
        
        #self.L3 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 20,10, Normalize = True)
        self.L3 = FNOBlocks(in_channels=4*factor*self.width, out_channels=4*factor*self.width,
                                 n_modes=(self.modes), norm='instance_norm')
        
        #self.L4 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 20,10)
        self.L4 = FNOBlocks(in_channels=4*factor*self.width, out_channels=4*factor*self.width, n_modes=(self.modes))

        #self.L5 = OperatorBlock_1D(8*factor*self.width, 2*factor*self.width, 48,10, Normalize = True)
        self.L5 = FNOBlocks(in_channels=8*factor*self.width, out_channels=2*factor*self.width,
                             n_modes=(self.modes), norm='instance_norm')

        #self.L6 = OperatorBlock_1D(4*factor*self.width, self.width, 40,15) # will be reshaped
        self.L6 = FNOBlocks(in_channels=4*factor*self.width, out_channels=self.width, n_modes=(self.modes))

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 165)

    def forward(self, x,c):
        
        grid = self.get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid, c), dim=-1)
        
        
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 2, 1)
        
        padding = int((x_fc0.shape[-1]/30)*self.padding)
        
        
        x_fc0 = F.pad(x_fc0, [0,padding]) 
        
        
        D1 = x_fc0.shape[-1]
        
        x_c0 = self.L0(x_fc0, output_shape = (int(D1*self.factor), )) 
        
        x_c1 = self.L1(x_c0 , output_shape = (D1//2,)) 
        

        x_c2 = self.L2(x_c1 , output_shape = (D1//2,)) 
        
        x_c3 = self.L3(x_c2, output_shape = (D1//2,)) 
        
        x_c4 = self.L4(x_c3, output_shape = (D1//2,)) 
        
        x_c4 = torch.cat([x_c4, x_c1], dim=1) 
        
        x_c5 = self.L5(x_c4, output_shape = (int(D1*self.factor), ))
        
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        
        x_c6 = self.L6(x_c5, output_shape = (int(D1), )) 
        
        x_c6 = torch.cat([x_c6, x_fc0], dim=1) 
        
        if self.padding!=0:
            x_c6 = x_c6[..., :-padding]

        x_c6 = x_c6.permute(0, 2, 1) # shape now is (N, 300, 3*(165+120+1))
        
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)