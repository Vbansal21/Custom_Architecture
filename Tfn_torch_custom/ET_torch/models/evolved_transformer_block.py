#from ..models.embedder import Embedder, PositionalEncoder
#import math
import torch
from typing import Optional
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..models.gated_linear_unit import GLU

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
            x = torch.cat((x.unsqueeze(1),context.unsqueeze(1)),dim=1)
        glued = self.glu(self.layer_norms[0](x))+x
        glu_normed = self.layer_norms[1](glued)

        left_branch = self.left_net(glu_normed)
        right_branch = self.right_net(glu_normed.transpose(1,2)).transpose(1,2)
        right_branch = F.pad(input=right_branch, pad=(0,left_branch.shape[2]-right_branch.shape[2],0,0,0,0), mode='constant', value=0)

        mid_result = left_branch+right_branch
        mid_result = self.mid_layer_norm(mid_result)
        mid_result = self.sep_conv(mid_result.transpose(1,2)).transpose(1,2)

        mid_result = mid_result + glued

        normed = self.layer_norms[2](mid_result)
        if context != None:
            context = normed[:,1]
            normed = normed[:,0]
            mid_result = mid_result[:,0]
        else:
            context = normed
        if self.context_pass:
            attended = self.attention(normed,context) + mid_result 
        else:
            attended = self.attention(normed) + mid_result
        normed = self.layer_norms[3](attended)
        if self.pkm == None:
            forwarded = self.feed_forward(attended) + normed
        else:
            forwarded = self.feed_forward(attended)+self.pkm(attended)+normed
        return forwarded