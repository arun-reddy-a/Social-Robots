import torch
import torch.nn as nn

def Condition_embedding(in_chan=2, up_dim=6):

    layers = nn.Sequential(
                nn.Linear(in_chan, 2*up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(2*up_dim, up_dim, bias=True), torch.nn.GELU(),
            )
    return layers