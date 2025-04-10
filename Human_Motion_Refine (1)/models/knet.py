import torch
import torch.nn as nn

def kernel(in_chan=2, up_dim=32):
	"""
		Kernel network apply on grid
	"""
	layers = nn.Sequential(
				nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
				nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
			)
	return layers