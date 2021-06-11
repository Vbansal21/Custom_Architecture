"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.parameter import Parameter
#import matplotlib.pyplot as plt

#import operator
#from functools import reduce
#from functools import partial
#from timeit import default_timer
from .utilities3 import *
from copy import deepcopy as dcpy

torch.manual_seed(0)
np.random.seed(0)

from torch.utils.checkpoint import checkpoint
checkpointed = True

def ckpt(f,arg1,arg2=None,arg3=None,checkpointed = checkpointed):
    if checkpointed:
        if arg2 == None and arg3 == None:
            return checkpoint(f,arg1)
        elif arg3 == None:
            return checkpoint(f,arg1,arg2)
        else:
            return checkpoint(f,arg1,arg2,arg3)
    else:
        if arg2 == None and arg3 == None:
            return f(arg1)
        elif arg3 == None:
            return f(arg1,arg2)
        else:
            return f(arg1,arg2,arg3)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        if input.size(-1)<weights.size(-1):
            weights = weights[:,:,:input.size(-1)]
        if input.size(-2)<weights.size(-2):
            weights = weights[:,:input.size(-2),:]
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = ckpt(torch.fft.rfft,x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = ckpt(self.compl_mul1d,x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft,x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width,inp_dim=2,out_dim=1,ffd_dim=128,transpose_req=True,num_layers=4):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 'num_layers' layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.transpose_req = transpose_req

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(inp_dim, self.width)

        conv_block = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv_layers = nn.ModuleList([dcpy(conv_block) for _ in range(num_layers)])
        w_block = nn.Conv1d(self.width, self.width, 1)
        self.w_layers = nn.ModuleList([dcpy(w_block) for _ in range(num_layers)])
        self.num_layers = num_layers

        self.fc1 = nn.Linear(self.width, ffd_dim)
        self.fc2 = nn.Linear(ffd_dim, out_dim)

    def forward(self, x):
        
        if not self.transpose_req:
            x = self.fc0(x.transpose(-1,-2))
        else:
            x = self.fc0(x)

        x = x.transpose(-1,-2)

        for i in range(self.num_layers):
            x_ = ckpt(self.conv_layers[i],x) + ckpt(self.w_layers[i],x)
            x_ = F.relu(x_)
            
        x = x.transpose(-1,-2)
        x = self.fc1(x)
        x = F.relu(x)


        if not self.transpose_req:
            x = self.fc2(x).transpose(-1,-2)
        else:
            x = self.fc2(x)

        return x
