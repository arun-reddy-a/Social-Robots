import torch
import numpy as np
import math
import torch.nn as nn
from functools import reduce
import torch.fft

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GaussianRF_1D(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
        k_max = size//2
        if dim == 1:
            if size %2 ==0:
                k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                            torch.arange(start=-k_max, end=0, step=1, device=device)), 0)
            else:
                k = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=device), \
                            torch.arange(start=-k_max, end=0, step=1, device=device)), 0)
            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)
            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers
            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)
            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)
            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)
    def sample(self, N, mul=1):
        coeff = torch.randn(N, *self.size, 2, device=self.device)*mul
        coeff[...,0] = self.sqrt_eig*coeff[...,0] #real
        coeff[...,1] = self.sqrt_eig*coeff[...,1] #imag 
        
        ##########torch 1.7###############
        #u = torch.ifft(coeff, self.dim, normalized=False)
        #u = u[...,0]
        ##################################
        
        #########torch latest#############
        coeff_new = torch.complex(coeff[...,0],coeff[...,1])
        #print(coeff_new.size())
        u = torch.fft.ifft(coeff_new, dim = (-1), norm=None)
        
        u = u.real
        
        
        return u

class Gaussian_RF(object):
    def __init__(self, dim, res, alpha=2, tau=3, sigma=None, codim=None, device=None):
        self.alpha = alpha
        self.dim = dim 
        self.tau = tau
        self.sigma = sigma
        self.device = device
        self.codim = codim
        self.res = res
        self.grf = GaussianRF_1D(self.dim, res, self.alpha, self.tau, device=self.device)
    
    def sample(self,N, res=None, mul = 1):
        ## grf = GaussianRF_1D(self.dim, res, self.alpha, self.tau, device=self.device)
        #x = grf.sample(N, mul = mul)
        l = []
        for i in range(self.codim):
            l.append(self.grf.sample(N, mul = mul)[...,None])
        
        return torch.cat(l, dim = 2 )

