#from ..models.embedder import Embedder, PositionalEncoder
#import math
from turtle import forward
import torch
from einops import rearrange,repeat
from typing import Optional
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

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
            
class GatedConvolution(nn.Module):
    def __init__(self,d_model,patch_size=3,padding=1):
        super(GatedConvolution,self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model,kernel_size=patch_size,padding=padding,groups=d_model//2,bias=True)
        #init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out

class GLU(nn.Module):
    def __init__(self,d_model,num_layers,patch_size=3,padding=1):#Dauphin's m_input= n_input= d_model
        super(GLU,self).__init__()
        self.gated_convs = nn.ModuleList([GatedConvolution(d_model,patch_size,padding) for _ in range(num_layers)])
    
    def forward(self,x):
        for convolution in self.gated_convs:
            x = convolution(x)
        return x
        
class SeparableConv1D(nn.Module):
    """ Input: (batch_size, in_channel, length)
        Output: (batch_size, out_channel, length)
    """
    def __init__(self, in_channel, inner_channels, out_channel, kernel_size=1, padding=0):
        super().__init__()
        self.deep_wise = nn.Conv1d(in_channel, inner_channels, kernel_size=kernel_size, padding=padding, groups=min(in_channel,inner_channels))
        self.point_wise = nn.Conv1d(inner_channels, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.deep_wise(x)
        x = self.point_wise(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class ET_Encoder_Block(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4,attn = None,ffd = None,pkm=None):
        super(ET_Encoder_Block,self).__init__()
        if attn == None:
            self.attention = nn.MultiheadAttention(d_model, num_heads) 
        else:
            self.attention = attn
        self.layer_norms = nn.ModuleList([ScaleNorm(d_model) for _ in range(4)])
        if ffd == None:
            self.feed_forward = nn.Sequential(
                nn.Conv1d(in_channels=d_model,out_channels=d_model*ff_hidden,kernel_size=1,stride=1,padding=0),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=d_model,kernel_size=1,stride=1,padding=0),
            )
        else:
            self.feed_forward = ffd
        self.pkm = pkm
        self.glu = GLU(d_model,1)
        self.left_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model*ff_hidden,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=d_model,kernel_size=1,padding=0),
        )
        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model//2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model//2,out_channels=d_model,kernel_size=1,padding=0),
        )

        self.mid_layer_norm=ScaleNorm(d_model)
        self.sep_conv = SeparableConv1D(d_model,d_model//2,d_model,kernel_size=9,padding=4)

    def forward(self,x:Tensor,*args) -> Tensor:

        glued = ckpt(self.glu,x)+x
        #glued = ckpt(self.glu,self.layer_norms[0](x))+x
        glu_normed = self.layer_norms[1](glued)

        left_branch = ckpt(self.left_net,glu_normed.transpose(1,2)).transpose(1,2)
        right_branch = ckpt(self.right_net,glu_normed.transpose(1,2)).transpose(1,2)

        mid_result = left_branch+right_branch
        mid_result = ckpt(self.mid_layer_norm,mid_result)
        mid_result = ckpt(self.sep_conv,mid_result.transpose(1,2)).transpose(1,2)

        mid_result = mid_result + glued

        normed = self.layer_norms[2](mid_result)
        attended = ckpt(self.attention,normed) + mid_result

        normed = self.layer_norms[3](attended)
        if self.pkm == None:
            forwarded = ckpt(self.feed_forward,normed.transpose(1,2)).transpose(1,2) + attended
        else:
            forwarded = ckpt(self.feed_forward,normed.transpose(1,2)).transpose(1,2) + ckpt(self.pkm,normed) + attended
        return forwarded
        


class ET_Decoder_Block(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4,attn = None,ffd = None,pkm=None):
        super(ET_Decoder_Block,self).__init__()

        if attn == None:
            self.attention_self_1 = nn.MultiheadAttention(d_model, num_heads) 
            self.attention_self_2 = nn.MultiheadAttention(d_model, num_heads) 
            self.attention_cross_1 = nn.MultiheadAttention(d_model, num_heads) 
            self.attention_cross_2 = nn.MultiheadAttention(d_model, num_heads) 
        else:
            self.attention_self_1 = attn['self_1'] 
            self.attention_self_2 = attn['self_2'] 
            self.attention_cross_1 = attn['cross_1'] 
            self.attention_cross_2 = attn['cross_2'] 

        self.layer_norms = nn.ModuleList([ScaleNorm(d_model) for _ in range(5)])

        self.pkm = pkm

        if ffd == None:
            self.feed_forward = nn.Sequential(
                nn.Conv1d(in_channels=d_model,out_channels=d_model*ff_hidden,kernel_size=1,stride=1,padding=0),
                nn.SiLU(),
                nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=d_model,kernel_size=1,stride=1,padding=0),
            )
        else:
            self.feed_forward = ffd

        self.sep_norm=ScaleNorm(d_model)
        self.sep_conv_l = SeparableConv1D(d_model,d_model*2,d_model,kernel_size=11,padding=5)
        self.sep_conv_r = SeparableConv1D(d_model,d_model//2,d_model,kernel_size=7,padding=3)
        self.sep_mid = SeparableConv1D(d_model,d_model,d_model,kernel_size=7,padding=3)

    def forward(self,x:Tensor,context:Tensor) -> Tensor:

        normed_x = x #self.layer_norms[0](x)

        self_attn = ckpt(self.attention_self_1,normed_x)
        cross_attn = ckpt(self.attention_cross_1,normed_x,context)

        attended = self_attn+cross_attn
        attended_normed = self.layer_norms[1](attended)

        sep_l = ckpt(self.sep_conv_l,attended_normed.transpose(1,2)).transpose(1,2)
        sep_l = F.relu(sep_l)

        sep_r = ckpt(self.sep_conv_r,attended_normed.transpose(1,2)).transpose(1,2)

        sep_attended = sep_l + sep_r
        sep_normed = self.sep_norm(sep_attended)

        sep_normed = ckpt(self.sep_mid,sep_normed.transpose(1,2)).transpose(1,2)
        sep_attended = sep_normed + attended

        sep_attn_normed = self.layer_norms[2](sep_attended)

        self_attn = ckpt(self.attention_self_2,sep_attn_normed) + sep_attended

        self_attn_normed = self.layer_norms[3](self_attn)
        cross_attn = ckpt(self.attention_cross_2,self_attn_normed,context) + self_attn

        attn_normed = self.layer_norms[4](cross_attn)

        if self.pkm==None:
            forwarded = ckpt(self.feed_forward,attn_normed.transpose(1,2)).transpose(1,2) + cross_attn
        else:
            forwarded = ckpt(self.feed_forward,attn_normed.transpose(1,2)).transpose(1,2) + cross_attn + ckpt(self.pkm,attn_normed)

        return forwarded