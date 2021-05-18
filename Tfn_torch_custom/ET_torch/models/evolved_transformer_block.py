#from ..models.embedder import Embedder, PositionalEncoder
#import math
import torch
from einops import rearrange
from typing import Optional
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..models.gated_linear_unit import GLU

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

class EvolvedTransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4,attn = None,ffd = None,context = True,pkm=None):
        super(EvolvedTransformerBlock,self).__init__()
        self.context_pass = context
        if attn == None:
            self.attention = nn.MultiheadAttention(d_model, num_heads) 
        else:
            self.attention = attn
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        if ffd == None:
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model,ff_hidden*d_model),
                nn.ReLU(),
                nn.Linear(ff_hidden*d_model,d_model),
            )
        else:
            self.feed_forward = ffd
        self.pkm = pkm
        self.glu = GLU(d_model,1)
        self.left_net = nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU()
        )
        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model//2,kernel_size=3,padding=1),
            nn.ReLU()
        )

        self.mid_layer_norm=nn.LayerNorm(d_model*ff_hidden)
        self.sep_conv=nn.Sequential(
            nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=1,kernel_size=9,padding=4),
            nn.Conv1d(in_channels=1,out_channels=d_model,kernel_size=1)
        )

    def forward(self,x:Tensor,context: Optional[Tensor] = None) -> Tensor:

        if context != None:
            x = rearrange(torch.cat((x.unsqueeze(0),context.unsqueeze(0)),dim=0), 't b n d -> (t b) n d')
        glued = ckpt(self.glu,self.layer_norms[0](x))+x
        glu_normed = self.layer_norms[1](glued)

        left_branch = ckpt(self.left_net,glu_normed)
        right_branch = ckpt(self.right_net,glu_normed.transpose(1,2)).transpose(1,2)
        right_branch = F.pad(input=right_branch, pad=(0,left_branch.shape[2]-right_branch.shape[2],0,0,0,0), mode='constant', value=0)

        mid_result = left_branch+right_branch
        mid_result = ckpt(self.mid_layer_norm,mid_result)
        mid_result = ckpt(self.sep_conv,mid_result.transpose(1,2)).transpose(1,2)

        mid_result = mid_result + glued

        normed = self.layer_norms[2](mid_result)
        if context != None:
            normed = rearrange(normed,'(t b) n d -> t b n d',t=2)
            context = normed[1]
            normed = normed[0]
            mid_result = mid_result[0]
        else:
            context = normed
        if self.context_pass:
            attended = ckpt(self.attention,normed,context) + mid_result 
        else:
            attended = ckpt(self.attention,normed) + mid_result
        normed = self.layer_norms[3](attended)
        if self.pkm == None:
            forwarded = ckpt(self.feed_forward,normed) + attended
        else:
            forwarded = ckpt(self.feed_forward,normed)+ckpt(self.pkm,normed)+attended
        return forwarded