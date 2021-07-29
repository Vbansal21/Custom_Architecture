#from ..models.embedder import Embedder, PositionalEncoder
import math
import torch
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer_s import Attention
from copy import deepcopy as dcpy


from torch.utils.checkpoint import checkpoint
checkpointed = True

def ckpt(f,*arg,checkpointed = checkpointed):
    if checkpointed:
        return checkpoint(f,*arg)
    else:
        return f(*arg)
            
class GatedConvolution(nn.Module):
    def __init__(self,d_model,patch_size=3,padding=1,dim=-1):
        super(GatedConvolution,self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model,kernel_size=patch_size,padding=0,groups=1,bias=True)
        self.padding = padding*2
        self.dim = dim
        #init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(F.pad(x.transpose(-1,-2),(self.padding,0),value=0)).transpose(-1,-2)
        out, gate = convoluted.chunk(2, dim=self.dim)
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
    def __init__(self, in_channels, inner_channels, out_channel, kernel_size=1, padding=0):
        super().__init__()
        self.pre_point_wise = nn.Conv1d(in_channels, inner_channels, kernel_size=1,groups=1)
        self.deep_wise = nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=0, groups=inner_channels)
        self.post_point_wise = nn.Conv1d(inner_channels, out_channel, kernel_size=1,groups=1)
        self.padding = padding*2

    def forward(self, x):
        x = self.pre_point_wise(x)
        x = self.deep_wise(F.pad(x,(self.padding,0),value=0))
        x = self.post_point_wise(x)
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
    def __init__(self,d_model,num_heads=8,ff_hidden=4,dim_heads=None,num_mem_kv=None,num_prev_state=None,num_prev_mem=None,hop_attn=None,local_heads=0,rotary_pos_emb=True,causal=False,nystrom=True,attend_to_self=True,context=False,use_mask=False,attn = None,ffd = None,pkm=None):
        super(ET_Encoder_Block,self).__init__()
        if attn == None:
            self.attention = Attention(d_model,
                                    heads=num_head,
                                    dim_head=dim_heads,
                                    num_mem_kv=num_mem_kv,
                                    num_prev_state=num_prev_state,
                                    num_prev_mem=num_prev_mem,
                                    local_heads=local_heads,
                                    hop_attn=hop_attn,
                                    rotary_pos_emb=rotary_pos_emb,
                                    causal=causal,
                                    nystrom=nystrom,
                                    attend_to_self=attend_to_self,
                                    context=context,
                                    use_mask=use_mask,
                                )
        else:False
            self.attention = attn
        self.layer_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(5)])
        if ffd == None:
            self.feed_forward = nn.Sequential(
            Rearrange("... n d -> ... d n"),
                nn.Conv1d(in_channels=d_model,out_channels=d_model*ff_hidden,kernel_size=1,stride=1,padding=0,groups=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=d_model,kernel_size=1,stride=1,padding=0,groups=1),
            Rearrange("... d n -> ... n d")
            )
        else:
            self.feed_forward = ffd
        self.pkm = pkm
        self.glu = GLU(d_model,1)
        self.left_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model*ff_hidden,kernel_size=1,padding=0,groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=d_model,kernel_size=1,padding=0,groups=1),
        )
        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model//2,kernel_size=3,padding=1,groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model//2,out_channels=d_model,kernel_size=1,padding=0,groups=1),
        )
        self.sep_conv = SeparableConv1D(d_model,d_model//2,d_model,kernel_size=9,padding=4)

    def forward(self, x: Tensor, context: Tensor = None, src_mask: Tensor = None) -> Tensor:

        glued = ckpt(self.glu,self.layer_norms[0](x))+x
        glu_normed = self.layer_norms[1](glued)

        left_branch = ckpt(self.left_net,glu_normed.transpose(1,2)).transpose(1,2)
        right_branch = ckpt(self.right_net,glu_normed.transpose(1,2)).transpose(1,2)

        mid_result = left_branch+right_branch
        mid_result = ckpt(self.layer_norms[2],mid_result)
        mid_result = ckpt(self.sep_conv,mid_result.transpose(1,2)).transpose(1,2)

        mid_result = mid_result + glued

        normed = self.layer_norms[3](mid_result)
        attended = ckpt(self.attention,normed,context,src_mask)
        attended = attended  + mid_result

        normed = self.layer_norms[4](attended)
        
        forwarded = ckpt(self.feed_forward,normed) + attended

        return forwarded
        


class ET_Decoder_Block(nn.Module):
    def __init__(self,d_model,num_heads=8,ff_hidden=4,dim_heads=None,num_mem_kv=None,num_prev_state=None,num_prev_mem=None,hop_attn=None,local_heads=0,rotary_pos_emb=True,causal=False,nystrom=True,attend_to_self=True,context=True,use_mask=False,attn = None,ffd = None):
        super(ET_Decoder_Block,self).__init__()

        if attn == None:
            self.attention_self_1 = Attention(d_model,
                                        heads=num_head*2,
                                        dim_head=dim_heads//2,
                                        num_mem_kv=num_mem_kv,
                                        num_prev_state=num_prev_state,
                                        num_prev_mem=num_prev_mem,
                                        hop_attn=hop_attn,
                                        local_heads=local_heads,
                                        rotary_pos_emb=rotary_pos_emb,
                                        causal=causal,
                                        nystrom=nystrom,
                                        attend_to_self=attend_to_self,
                                        context=False,
                                        use_mask=use_mask)
            self.attention_self_2 = Attention(d_model,
                                        heads=num_head,
                                        dim_head=dim_heads,
                                        num_mem_kv=num_mem_kv,
                                        num_prev_state=num_prev_state,
                                        num_prev_mem=num_prev_mem,
                                        hop_attn=dcpy(hop_attn),
                                        local_heads=local_heads,
                                        rotary_pos_emb=rotary_pos_emb,
                                        causal=causal,
                                        nystrom=nystrom,
                                        attend_to_self=False,
                                        context=False,
                                        use_mask=use_mask)
            self.attention_cross_1 = Attention(d_model,
                                        heads=num_head,
                                        dim_head=dim_heads,
                                        num_mem_kv=num_mem_kv,
                                        num_prev_state=num_prev_state,
                                        num_prev_mem=num_prev_mem,
                                        hop_attn=dcpy(hop_attn),
                                        local_heads=0 if context else local_heads,
                                        rotary_pos_emb=rotary_pos_emb,
                                        nystrom=nystrom,
                                        attend_to_self=False,
                                        context=True,
                                        use_mask=bool(not context and use_mask))
            self.attention_cross_2 = Attention(d_model,
                                        heads=num_head,
                                        dim_head=dim_heads,
                                        num_mem_kv=num_mem_kv,
                                        num_prev_state=num_prev_state,
                                        num_prev_mem=num_prev_mem,
                                        hop_attn=dcpy(hop_attn),
                                        local_heads=0 if context else local_heads,
                                        rotary_pos_emb=rotary_pos_emb,
                                        nystrom=nystrom,
                                        attend_to_self=False,
                                        context=True,
                                        use_mask=bool(not context and use_mask))
        else:
            self.attention_self_1 = attn['self_1'] 
            self.attention_self_2 = attn['self_2'] 
            self.attention_cross_1 = attn['cross_1'] 
            self.attention_cross_2 = attn['cross_2'] 

        self.layer_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(6)])

        if ffd == None:
            self.feed_forward = nn.Sequential(
                Rearrange("... n d -> ... d n"),
                nn.Conv1d(in_channels=d_model,out_channels=d_model*ff_hidden,kernel_size=1,stride=1,padding=0,groups=1),
                nn.SiLU(),
                nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=d_model,kernel_size=1,stride=1,padding=0,groups=1),
                Rearrange("... d n -> ... n d")
            )
        else:
            self.feed_forward = ffd

        self.sep_conv_l = SeparableConv1D(d_model,d_model*2,d_model,kernel_size=11,padding=5)
        self.sep_conv_r = SeparableConv1D(d_model,d_model//2,d_model,kernel_size=7,padding=3)
        self.sep_mid = SeparableConv1D(d_model,d_model,d_model,kernel_size=7,padding=3)

    def forward(self, x: Tensor, context: Tensor = None, src_mask: Tensor = None) -> Tensor:
            
        if context == None:
            context = src

        normed_x = self.layer_norms[0](x)

        cross_attn = ckpt(self.attention_cross_1,normed_x,context)

        self_attn = ckpt(self.attention_self_1,normed_x,None,src_mask)

        attended = self_attn + cross_attn + x
        attended_normed = self.layer_norms[1](attended)

        sep_l = ckpt(self.sep_conv_l,attended_normed.transpose(1,2)).transpose(1,2)
        sep_l = F.relu(sep_l)

        sep_r = ckpt(self.sep_conv_r,attended_normed.transpose(1,2)).transpose(1,2)

        sep_attended = sep_l + sep_r
        sep_normed = self.layer_norms[2](sep_attended)

        sep_normed = ckpt(self.sep_mid,sep_normed.transpose(1,2)).transpose(1,2)
        sep_attended = sep_normed + attended

        sep_attn_normed = self.layer_norms[3](sep_attended)

        self_attn = ckpt(self.attention_self_2,sep_attn_normed,None,src_mask)

        self_attn = self_attn + sep_attended

        self_attn_normed = self.layer_norms[4](self_attn)

        cross_attn = ckpt(self.attention_cross_2,self_attn_normed,context)

        cross_attn = cross_attn + self_attn

        attn_normed = self.layer_norms[5](cross_attn)

        forwarded = ckpt(self.feed_forward,attn_normed) + cross_attn

        return forwarded