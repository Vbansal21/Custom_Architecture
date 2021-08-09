from json import decoder
import math, time, torch, random, warnings, copy, typing, torchnlp, numpy
#import profile
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from memory_profiler import profile

import copy
import typing
from typing import Tuple, Optional, Any, NoReturn, Union, List, Dict, Iterable

import numpy as np

from torch import Tensor
from torch.nn.modules.container import ModuleList, Module
from torch.nn.modules.dropout import Dropout

from einops import repeat,rearrange,reduce
from einops.layers.torch import Rearrange
from mogrifier import Mogrifier
from functools import partial

from .evolved_transformer_block import ET_Encoder_Block,ET_Decoder_Block, GLU
from .product_key_memory import PKM
from .hopfield_modules import HopfieldLayer
from .attention_layer_s import Attention,ProjectionUpdater,find_modules, RotaryEmbedding, Dynamic_Memory
from .conformer import ConformerConvModule

from .fourier_1d import FNO1d

from .g_mlp_gpt import gMLPGPT

from torchnlp.encoders.text import SubwordEncoder

from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
autocast = torch.cuda.amp.autocast

checkpointed = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

if torch.cuda.is_available():
    pass
    #torch.backends.cudnn.deterministic = True

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ckpt(f,*args,checkpointing = checkpointed):
    if checkpointing:
        return checkpoint(f,*args)
    else:
        return f(*args)

def _get_activation_fn(activation):
    if isinstance(activation,nn.Module):
        return activation

    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "silu":
        return nn.SiLU()
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation.lower() == "lrelu" or activation.lower() == "leakyrelu":
        return nn.LeakyReLU()

    raise RuntimeError("activation should be relu/gelu/silu/sigmoid/leakyrelu(lrelu), not {}".format(activation))

def list_subtract(l, r):
    return [el for el in l if el not in set(r)]

def fetch_rotary_parameters(module,module_to_find=nn.Linear):
    params = []
    for m in module.modules():
        if isinstance(m, module_to_find):
            for p in m.parameters():
                params.append(p)
    rest = list_subtract(module.parameters(), params)
    return params, rest

def Positional_Encoding(x: Tensor) -> Tensor :
    max_len = x.size(-2)
    d_model = x.size(-1)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).to(x.dtype) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).to(x.device)
    return (x + pe).contiguous()

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
                    'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)

class RMSNorm(Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class ScaleNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class Identity(Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x,*args):
        return x

class GEGLU(Module):
    def __init__(self, dim_in, dim_out,layers=1):
        super().__init__()
        self.proj = nn.Sequential(
            #GLU(dim_in,layers),
            nn.Linear(dim_in, dim_out * 2))
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)   

class GRUGating(nn.Module):
    def __init__(self, dim, fn=None, mogrify = False, norm = True):
        super().__init__()
        self.dim = dim
        self.fn = fn if exists(fn) else Identity()
        self.norm = RMSNorm(dim) if norm else Identity()
        self.gru = nn.GRUCell(dim, dim)
        self.mogrify = Mogrifier(dim, iters = 13, factorize_k = dim // 4) if mogrify else None

    def forward(self, x, y=None,*args):
        shape = x.shape
        dim = self.dim

        if not exists(y):
            y = x

        y = ckpt(self.fn,self.norm(y),*args)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = ckpt(self.gru,
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )
        out = gated_output.reshape(shape)

        return out

class FFd(Module):
    def __init__(self,dim,activation="lrelu",layers=1,kernal_size=1,mult=4,fn=None,):
        super(FFd,self).__init__()
        kernal_size = (kernal_size//2)*2 + 1
        self.l_1 = GEGLU(dim,dim*mult,layers) 
        self.padding = kernal_size - 1
        self.l_2 = nn.Sequential(
            #Rearrange("... (l x) y -> ... l x y",l=1),
            #nn.ZeroPad2d(((dim*mult*3)//2,(dim*mult*3)//2,self.padding,0)),
            #nn.Conv2d(1,1,(kernal_size,dim*mult*3 + 1),stride=(1,mult),padding=(0,0),groups=1),
            #Rearrange("... l x y -> ... (l x) y"),
            nn.Linear(dim*mult,dim),
            _get_activation_fn(activation),
        )

        self.fn = fn if isinstance(fn,nn.ModuleList) else ((nn.ModuleList([i for i in fn if i != None]) if isinstance(fn,Iterable) else nn.ModuleList((fn,))) if fn !=None else None)
        self.lin = (nn.Linear((len(fn)+1)*dim,dim) if isinstance(fn,nn.ModuleList) else nn.Linear(2*dim,dim)) if exists(self.fn) else None

    def forward(self,x,*args):
        out = ckpt(self.l_1,x)
        out = ckpt(self.l_2,out)
        if self.fn != None:
            for fn in self.fn:
                tmp = ckpt(fn,x)
                out = torch.cat((out,tmp),dim=-1)
            out = ckpt(self.lin,out)
        return out.contiguous()

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        if x.size(1) < self.max_seq_len:
            n = torch.arange(x.size(1), device = x.device)
            return repeat(self.emb(n),'n d -> b n d',b=x.size(0))
        else:
            warnings.warn("Consider Upgrading ABSOLUTEPOSITIONALEMBEDDING's 'max_seq_length' parameter (and possibly retrain it)")
            s = x.size(1)
            multiplier = 1/(2**0.5)
            n = []

            for i in range(s//self.max_seq_len):
                tmp = torch.arange(self.max_seq_len, device = x.device)
                n.append(repeat(self.emb(tmp),'n d -> b n d',b=x.size(0)) * multiplier)
                multiplier *= (2**0.5)
            else:
                tmp = torch.arange(s%self.max_seq_len, device = x.device)
                n.append(repeat(self.emb(tmp),'n d -> b n d',b=x.size(0)) * multiplier)

            tmp = torch.cat(n,dim=1)
            return tmp

class ACT_basic(nn.Module):
    def __init__(self,hidden_size,function,threshold=0.1,factor=3,max_hop=None,checkpointed=False):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.max_hop = max_hop
        self.p = nn.Linear(hidden_size,1)
        self.fn = function
        self.threshold = 1 - threshold
        self.factor = factor
        self.checkpointed = checkpointed

    def forward(self, state, time_enc=None, pos_enc=None, max_hop=None, encoder_output=None):
        # init_hdd
        b,l,d = state.size()
        dtype = state.dtype
        device = state.device
        ## [B, S]
        halting_probability = torch.zeros(b,l).to(device)
        ## [B, S]
        remainders = torch.zeros(b,l).to(device)
        ## [B, S]
        n_updates = torch.zeros(b,l).to(device)
        ## [B, S, HDD]
        previous_state = torch.zeros_like(state).to(device)
        step = 0
        # set time_enc, pos_enc and max_hop if None
        time_enc = default(time_enc,_gen_timing_signal(l,d)).to(dtype).to(device)
        max_hop = default(default(max_hop,self.max_hop),max(2,int((l**(1/self.factor) + default(encoder_output,state).size(-2)**(1/self.factor))/2) + 1))
        pos_enc = default(pos_enc,_gen_timing_signal(max_hop,d)).to(dtype).to(device)
        # initiating adaptive computation:
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :l, :].type(dtype).to(device)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,l,1).type(dtype).to(device)

            p = self.sigma(self.p(state)).squeeze(-1)

            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).type(dtype).to(device)

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).type(dtype).to(device) * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).type(dtype).to(device) * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if isinstance(self.fn,nn.Module):
                state = ckpt(self.fn,state,encoder_output,checkpointing=self.checkpointed)
            elif isinstance(self.fn,nn.ModuleList):
                for layer in self.fn:
                    state = ckpt(layer,state,encoder_output,checkpointing=self.checkpointed)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        return (previous_state, remainders,n_updates)
        #
        #    p_t = remainders + n_updates
        #    avg_p_t = torch.sum(torch.sum(p_t,dim=1)/p_t.size(1))/p_t.size(0)
        #    loss += default(act_loss_weight,0.001) * avg_p_t.item()

class TransformerBlock(Module):

    @profile
    def __init__(self,
                     d_model=256,
                     nhead=8, 
                     dim_ffd_mult=4, 
                     dropout=0.1, 
                     activation="sigmoid",
                     mem_kv=None,
                     pkm_keys=None,
                     decoder=False,
                     encoder_n_decoder=False,
                     hopfield=False,
                     fno_layers=4,
                     modes=None,
                     width=None,
                     rotary_pos_emb=True,
                     causal=False,
                     local_heads=0,
                     attn='h_attn',
                     attend_to_self=True,
                     mlp_layers=1,
                     use_mask=False,
                     context=True,
                     ET = True,
                     prev_state_kv = None,
                     num_prev_mem = None,
                ):
        super(TransformerBlock, self).__init__()
        
        self.dropout = nn.ModuleList([Dropout(dropout) for _ in range(9 if (encoder_n_decoder and decoder) else 7)])

        pkm_keys = default(pkm_keys,32)

        mem_kv = default(mem_kv,1024)

        modes = default(modes,nhead)
        width = default(width,nhead)

        pkm = PKM(d_model,heads=1,num_keys=pkm_keys,topk=min(pkm_keys,nhead))

        self.conformer = GRUGating(d_model,fn=ConformerConvModule(d_model,expansion_factor=dim_ffd_mult//2,causal=True,dropout=dropout),norm=False)

        self.fno = GRUGating(d_model,fn=FNO1d(modes,
                                    width,
                                    inp_dim=d_model,
                                    out_dim=d_model,
                                    ffd_dim=dim_ffd_mult*width,
                                    num_layers=fno_layers
                                ))

        if hopfield:
            self.hopfield = GRUGating(d_model,fn=HopfieldLayer(input_size=d_model,
                                                            update_steps_max=-1,
                                                            dropout=dropout))
        else:
            self.hopfield = Identity()
            
        self.mlp = GRUGating(d_model,fn=gMLPGPT(dim=d_model,depth=mlp_layers,heads=nhead,ff_mult=dim_ffd_mult//2,seq_len=2**16,window=d_model//2,attn_dim=d_model,prob_survival=1-dropout))

        ffd1 = FFd(dim=d_model,activation=activation,mult=dim_ffd_mult,fn=pkm)

        self.feed_forward = GRUGating(d_model,fn=ffd1)

        self.to_out = copy.deepcopy(self.feed_forward)

        if not decoder:
            attn_block = ET_Encoder_Block(d_model,
                                num_heads=nhead,
                                dim_heads=d_model//nhead,
                                num_mem_kv=mem_kv,
                                num_prev_state=prev_state_kv,
                                num_prev_mem=num_prev_mem,
                                local_heads=local_heads,
                                #hop_attn=hop_attn,
                                rotary_pos_emb=rotary_pos_emb,
                                causal=causal,
                                attn=attn,
                                attend_to_self=attend_to_self,
                                context=context,
                                use_mask=use_mask,
                                ff_hidden=dim_ffd_mult,
                                ) if ET else Attention(d_model,
                                            heads=nhead,
                                            dim_head=d_model//nhead,
                                            num_mem_kv=mem_kv,
                                            num_prev_state=prev_state_kv,
                                            num_prev_mem=num_prev_mem,
                                            local_heads=local_heads,
                                            #hop_attn=copy.deepcopy(hop_attn),
                                            rotary_pos_emb=rotary_pos_emb,
                                            causal=causal,
                                            attn=attn,
                                            attend_to_self=False,
                                            context=False,
                                            use_mask=use_mask,
                                        )
        else:
            self.self_inp_enc = (GRUGating(
                                            d_model,
                                            fn = ET_Encoder_Block(d_model,
                                                                num_heads=nhead,
                                                                dim_heads=d_model//nhead,
                                                                num_mem_kv=mem_kv,
                                                                num_prev_state=prev_state_kv,
                                                                num_prev_mem=num_prev_mem,
                                                                local_heads=local_heads,
                                                                #hop_attn=hop_attn,
                                                                rotary_pos_emb=rotary_pos_emb,
                                                                causal=causal,
                                                                attn=attn,
                                                                attend_to_self=attend_to_self,
                                                                context=context,
                                                                use_mask=use_mask,
                                                                ff_hidden=dim_ffd_mult,
                                                                ),
                                    norm=False) if ET else GRUGating(d_model,fn = Attention(d_model,
                                            heads=nhead,
                                            dim_head=d_model//nhead,
                                            num_mem_kv=mem_kv,
                                            num_prev_state=prev_state_kv,
                                            num_prev_mem=num_prev_mem,
                                            local_heads=local_heads,
                                            #hop_attn=copy.deepcopy(hop_attn),
                                            rotary_pos_emb=rotary_pos_emb,
                                            causal=causal,
                                            attn=attn,
                                            attend_to_self=False,
                                            context=False,
                                            use_mask=use_mask,
                                        ),norm=False)) if encoder_n_decoder else Identity()

            self.self_ctxt_enc = (GRUGating(
                                            d_model,
                                            fn = ET_Encoder_Block(d_model,
                                                                num_heads=nhead,
                                                                dim_heads=d_model//nhead,
                                                                num_mem_kv=mem_kv,
                                                                num_prev_state=prev_state_kv,
                                                                num_prev_mem=num_prev_mem,
                                                                local_heads=local_heads,
                                                                #hop_attn=hop_attn,
                                                                rotary_pos_emb=rotary_pos_emb,
                                                                causal=causal,
                                                                attn=attn,
                                                                attend_to_self=attend_to_self,
                                                                context=context,
                                                                use_mask=use_mask,
                                                                ff_hidden=dim_ffd_mult,
                                                                ),
                                    norm=False) if ET else GRUGating(d_model,fn = Attention(d_model,
                                            heads=nhead,
                                            dim_head=d_model//nhead,
                                            num_mem_kv=mem_kv,
                                            num_prev_state=prev_state_kv,
                                            num_prev_mem=num_prev_mem,
                                            local_heads=local_heads,
                                            #hop_attn=copy.deepcopy(hop_attn),
                                            rotary_pos_emb=rotary_pos_emb,
                                            causal=causal,
                                            attn=attn,
                                            attend_to_self=False,
                                            context=False,
                                            use_mask=use_mask,
                                        ),norm=False)) if encoder_n_decoder else Identity()

            attn_block = ET_Decoder_Block(d_model,
                                num_heads=nhead,
                                ff_hidden=dim_ffd_mult,
                                dim_heads=d_model//nhead,
                                num_mem_kv=mem_kv,
                                num_prev_state=prev_state_kv,
                                num_prev_mem=num_prev_mem,
                                #hop_attn=hop_attn,
                                local_heads=local_heads,
                                rotary_pos_emb=rotary_pos_emb,
                                causal=causal,
                                attn=attn,
                                attend_to_self=attend_to_self,
                                context=context,
                                use_mask=use_mask,
                                ) if ET else Attention(d_model,
                                                            heads=nhead,
                                                            dim_head=d_model//nhead,
                                                            num_mem_kv=mem_kv,
                                                            num_prev_state=prev_state_kv,
                                                            num_prev_mem=num_prev_mem,
                                                            #hop_attn=hop_attn,
                                                            local_heads=0,
                                                            rotary_pos_emb=rotary_pos_emb,
                                                            causal=causal,
                                                            attn=attn,
                                                            attend_to_self=attend_to_self,
                                                            context=context,
                                                            use_mask=bool(not context and use_mask),)
        
        self.attn = GRUGating(d_model,fn = attn_block,norm=False)

        self.decoder_exists = decoder


    def forward(self, src: Tensor,context: Optional[Tensor] = None, src_mask: Tensor = None) -> Tensor:

        output = src

        output = ckpt(self.feed_forward,output)
        output = self.dropout[0](output)

        output = ckpt(self.hopfield,output)
        output = self.dropout[1](output)

        if self.decoder_exists:
            ctxt_mask = src_mask if context is None else None
            context = output if context == None else context

            context = ckpt(self.feed_forward,context)
            context = self.dropout[0](context)
            context = ckpt(self.hopfield,context)
            context = self.dropout[1](context)

            output = self.dropout[7](ckpt(self.self_inp_enc,output,None,src_mask))
            context = self.dropout[8](ckpt(self.self_ctxt_enc,context,None,ctxt_mask))

        elif exists(context):
            context = ckpt(self.feed_forward,context)
            context = self.dropout[0](context)
            context = ckpt(self.hopfield,context)
            context = self.dropout[1](context)

        output = ckpt(self.attn,output,output,context,src_mask)
        output = self.dropout[2](output)

        output = ckpt(self.to_out,output)
        output = self.dropout[3](output)

        output = ckpt(self.fno,output)
        output = self.dropout[4](output)

        output = ckpt(self.conformer,output)
        output = self.dropout[5](output)

        output = ckpt(self.mlp,output)
        output = self.dropout[6](output)

        return output

class TransformerModule(ModuleList):

    #@profile
    def __init__(self, 
                    nhead, 
                    nhid, 
                    num_layers, 
                    d_model,
                    dropout=0.5,
                    enable_encoder=False,
                    deberta_layers=1,
                    repeated_deberta_layers=None,
                    max_len=2**17,
                    prev_state_len=8192,
                    fno_layers=4,
                    modes=None,
                    width=None,
                    full_block_repeat=False,
                    causal=False,
                    attn='h_attn',
                    local_heads=2,
                    attend_to_self=True,
                    prev_state_self_num=32,
                    mlp_layers=1,
                    encoder_n_decoder=True,
                    repeated_main_layers = None,
                    ET = True,
                    mem_kv = None
                    ):
        super(TransformerModule, self).__init__()

        # deprecated cross attention config saveing
        self.config = dict(d_model=d_model, nhead=nhead, dim_ffd_mult=nhid//d_model, dropout=dropout,decoder=True,mem_kv=mem_kv,hopfield=True ,fno_layers=fno_layers,modes=modes,width=width,causal=causal,attn=attn  ,local_heads=local_heads,attend_to_self=attend_to_self,mlp_layers=mlp_layers,ET=ET)

        self.full_block_repeat = full_block_repeat
        self.enable_encoder=enable_encoder
        self.repeated_deberta_layers = default(repeated_deberta_layers,deberta_layers)
        self.repeated_main_layers = default(repeated_main_layers,num_layers)

        if num_layers != 0:
            if not enable_encoder:
                block = TransformerBlock(d_model, nhead, nhid, dropout,mem_kv=mem_kv,hopfield=True ,fno_layers=fno_layers,modes=modes,width=width,causal=causal, attn=attn,attend_to_self=attend_to_self,mlp_layers=mlp_layers,context=False,ET=ET)
                self.decoder_self = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(num_layers-1)])
                self.decoder_cross = Identity()
                self.encoder = Identity()
            else:
                block = TransformerBlock(d_model, nhead, nhid, dropout,mem_kv=mem_kv,fno_layers=fno_layers,modes=modes,width=width,causal=causal, attn=attn,attend_to_self=attend_to_self,mlp_layers=mlp_layers,context=False,ET=ET)
                self.encoder = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(num_layers-1)])
                self.decoder_self = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
                block = TransformerBlock(d_model, nhead, nhid, dropout,mem_kv=mem_kv,decoder=True,hopfield=True ,fno_layers=fno_layers,modes=modes,width=width,causal=causal,attn=attn  ,local_heads=local_heads,attend_to_self=attend_to_self,mlp_layers=mlp_layers,ET=ET)
                self.decoder_cross = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(num_layers-1)])
        else:
            self.encoder = nn.ModuleList([Identity()])
            self.decoder_self = nn.ModuleList([Identity()])
            self.decoder_cross = nn.ModuleList([Identity()])
            
        self.absolutepositionalembedding = AbsolutePositionalEmbedding(d_model,max_len) if deberta_layers else None
        block = TransformerBlock(d_model, nhead, nhid, dropout,mem_kv=mem_kv,decoder=True,encoder_n_decoder=encoder_n_decoder,hopfield=True,fno_layers=fno_layers,modes=modes,width=width,causal=causal , attn=attn ,local_heads=local_heads,attend_to_self=attend_to_self,mlp_layers=mlp_layers,context=False,ET=ET) if deberta_layers else None
        self.deberta_layers = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(deberta_layers-1)]) if deberta_layers else None
        
        self.prev_state_exists = False

        if prev_state_len > 0 and prev_state_self_num > 0:
            self.prev_state_exists = True
            self.register_buffer(
                name='prev_state',
                tensor=torch.zeros((1,prev_state_len,d_model))
            )

            self.prev_state_update = Attention(d_model,
                                                heads=nhead,
                                                dim_head=d_model//nhead,
                                                num_mem_kv=0,
                                                rotary_pos_emb=True,
                                                attn=attn,)
        else:
            self.prev_state = None

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_deberta_layers = deberta_layers
        self.prev_state_self_num = prev_state_self_num
        
    def pretrained_layer_multiplier(self,num=1,deb_num=1):
        self.num_layers *= num
        def multiple_of_layers(layers,num_=1):
            l = [i for i in layers]
            for i in range(num_-1):
                for j in layers:
                    l.append(copy.deepcopy(j))
            return l
        if self.enable_encoder:
            self.encoder = nn.ModuleList(multiple_of_layers(self.encoder,num))
            self.decoder_self = nn.ModuleList(multiple_of_layers(self.decoder_self,num))
            self.decoder_cross = nn.ModuleList(multiple_of_layers(self.decoder_cross,num))
            self.deberta_layers = nn.ModuleList(multiple_of_layers(self.deberta_layers,deb_num))
        else:
            self.decoder_self = nn.ModuleList(multiple_of_layers(self.decoder_self,num))
            self.deberta_layers = nn.ModuleList(multiple_of_layers(self.deberta_layers,deb_num))
            
    def convert_decoder_only_to_encoder_decoder(self):
        #Deprecated
        self.enable_encoder = True
        self.encoder = copy.deepcopy(self.decoder_self)
        self.decoder_self = self.decoder_self
        if deberta_layers != None:
            self.decoder_cross = nn.ModuleList(copy.deepcopy([i for j,i in enumerate(self.deberta_layers) if j < self.num_layers]))
            for i in self.decoder_cross:
                tmp = i.self_inp_enc
                i.self_inp_enc = Identity()
                del(tmp)
                tmp = i.self_ctxt_enc
                i.self_ctxt_enc = Identity()
                del(tmp)
        else:
            block = TransformerBlock(*self.config) if self.num_layers != 0 else Identity()
            self.decoder_cross = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(self.num_layers-1)]) if self.num_layers != 0 else Identity()
            

    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:
        output = src
        ctxt = context

        src_mask = None#torch.triu(torch.ones((src.size(1),src.size(1))),diagonal=1)
            
        if self.full_block_repeat:
            for _ in range(self.repeated_main_layers+1):
                if self.enable_encoder:
                    for enc in self.encoder:
                        ctxt = ckpt(enc,ctxt)
                    for i in range(self.num_layers):
                        output = ckpt(self.decoder_self[i],output,None,src_mask)
                        output = ckpt(self.decoder_cross[i],output,ctxt,src_mask)
                else:
                    for dec in self.decoder_self:
                        output = ckpt(dec,output,None,src_mask)
        else:
            if self.enable_encoder:
                for enc in self.encoder:
                    for _ in range(self.repeated_main_layers+1):
                        ctxt = ckpt(enc,ctxt)
                for i in range(self.num_layers):
                    for _ in range(self.repeated_main_layers+1):
                        output = ckpt(self.decoder_self[i],output,None,src_mask)
                    for _ in range(self.repeated_main_layers+1):
                        output = ckpt(self.decoder_cross[i],output,ctxt,src_mask)
            else:
                for dec in self.decoder_self:
                    for _ in range(self.repeated_main_layers+1):
                        output = ckpt(dec,output,None,src_mask)
        
        if self.deberta_layers!=None:
            out = Positional_Encoding(self.absolutepositionalembedding(output))
            if self.full_block_repeat:
                for _ in range(self.repeated_deberta_layers+1):
                    for enc in self.deberta_layers:
                        out = ckpt(enc,out,output,src_mask)
                        if exists(context):
                            out = ckpt(enc,out,context)
                else:
                    if self.deberta_layers!=None:
                        output = out
            else:
                for enc in self.deberta_layers:
                    for _ in range(self.repeated_deberta_layers+1):
                        out = ckpt(enc,out,output,src_mask)
                        if exists(context):
                            out = ckpt(enc,out,context)
                else:
                    if self.deberta_layers!=None:
                        output = out

        if self.prev_state_exists:
            prev_state = repeat(self.prev_state,'1 n d -> b n d',b=output.size(0))

            prev_state = ckpt(self.prev_state_update,prev_state,output)
            if exists(ctxt):
                prev_state = ckpt(self.prev_state_update,prev_state,ctxt)

            for _ in range(self.prev_state_self_num):
                prev_state = ckpt(self.prev_state_update,prev_state)

            output = ckpt(self.prev_state_update,output,prev_state)

            prev_state = torch.sum(prev_state,dim=0,keepdim=True).reshape(self.prev_state.shape) / output.size(0)
            self.prev_state = prev_state.clone().detach()

        if context != None:
            return output,ctxt
        else:
            return output

class TransformerModel(Module):

    @profile
    def __init__(self, 
                    ninp: int, 
                    nhead: int, 
                    nhid: int, 
                    nlayers: int,
                    ntoken: Optional[int] = None, 
                    padding_idx: int = 0,
                    dropout: float = 0.5,
                    activation: str = 'Lrelu',
                    mem_token: int = 00,
                    context_mem_token: Optional[int] = None,
                    encoder_decoder: bool = False,
                    deberta_layers: int = 1,
                    repeated_deberta_layers: int = 2,
                    max_seq_len=2**17,
                    discriminator: bool = False,
                    seq_scale_down: int = 8,
                    auto_check_redraw: bool = True,
                    feature_redraw_interval: int = 256,
                    full_block_repeat: bool = False,
                    causal: bool = True,
                    prev_state_len: int = 8192,
                    prev_state_self_num=32,
                    attn: str = 'h_attn',
                    local_heads: int = 1,
                    attend_to_self=True,
                    fno_layers=4,
                    modes=None,
                    width=None,
                    mlp_layers=1,
                    encoder_n_decoder=True,
                    repeated_main_layers=None,
                    ET=True,
                    mem_kv=None,
                    device: torch.DeviceObjType = device
                ) -> NoReturn :
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.time_ = [0,0]

        self.encoder_decoder = encoder_decoder
        self.transformer_block = TransformerModule(nhead, 
                                                        nhid, 
                                                        nlayers, 
                                                        ninp,
                                                        dropout,
                                                        enable_encoder=encoder_decoder,
                                                        deberta_layers=deberta_layers,
                                                        repeated_deberta_layers=repeated_deberta_layers,
                                                        max_len=max_seq_len,
                                                        full_block_repeat=full_block_repeat,
                                                        causal=causal,
                                                        prev_state_len=prev_state_len,
                                                        attn=attn,
                                                        local_heads=local_heads,
                                                        attend_to_self=attend_to_self,
                                                        fno_layers=fno_layers,
                                                        modes=modes,
                                                        width=width,
                                                        prev_state_self_num=prev_state_self_num,
                                                        mlp_layers=mlp_layers,
                                                        encoder_n_decoder=encoder_n_decoder,
                                                        repeated_main_layers=repeated_main_layers,
                                                        ET=ET,
                                                        mem_kv=mem_kv,
                                                        )
        
        self.embedding_encoder = nn.Embedding(ntoken, ninp,padding_idx=padding_idx) if ntoken != None else Identity()

        
        self.ninp = ninp
        self.ntokens = ntoken
        
        self.decoder = nn.Sequential(
            nn.Linear(ninp,ntoken),
            #nn.Sigmoid(),
            ) if ntoken != None else Identity()

        self.decoder = nn.Sequential(
                nn.Linear(ninp,2),
                #nn.LeakyReLU(0.05),
            ) if discriminator else self.decoder

        self.ffd1 = FFd(dim=ninp,activation=activation)
        self.ffd2 = copy.deepcopy(self.ffd1)

        self.mem_exist = True if mem_token else False
        if self.mem_exist:
            if type(mem_token)==int:
                self.mem = nn.Parameter(torch.randn(mem_token,ninp))
            elif type(mem_token) == Tensor:
                assert mem_token.size(-1)==ninp
                self.mem = nn.Parameter(mem_token)
        else:
            self.mem = None
            
        self.seq_scale_down = seq_scale_down
        self.attn_len = (self.seq_scale_down*3 // 2)*2 + 1

        self.scale_down_conv = nn.Conv1d(ninp,ninp,self.seq_scale_down*3,self.seq_scale_down,padding=(self.seq_scale_down*3 - 1)//2,groups=1)

        modes = nhead if modes == None else modes
        width = nhead if width == None else width

        self.scale_up_conv = nn.ConvTranspose1d(ninp,ninp,self.seq_scale_down*3,self.seq_scale_down,groups=1)

        self.device = device

        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.proj_updater = ProjectionUpdater(self.transformer_block, feature_redraw_interval)

        self.tokenizer = None
        self.vocab = None
        self.optimizer = None
        self.scheduler = None
        self.scheduler_lambda = None

        self.prev_states = []
        self.max_prev_states = 1

        self.init_weights()

    def get_avg_inference_time(self) -> int:
        if self.time_[1] != 0:
            return self.time_[0] / self.time_[1]
        else:
            return 0

    def fix_projection_matrices_(self) -> NoReturn :
        self.proj_updater.feature_redraw_interval = None

    def defix_projection_matrices_(self) -> NoReturn :
        self.proj_updater.feature_redraw_interval = self.feature_redraw_interval

    def init_weights(self) -> NoReturn :
        for w in self.parameters():
            w.data.uniform_(-0.01,0.01)
            
    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())
            
    def multiply_pretrained_transformer_layers(self,num: int = 1,deb_num: int = 1) -> NoReturn :
        self.transformer_block.pretrained_layer_multiplier(num,deb_num)

    def convert_decoder_only_to_encoder_decoder(self) -> NoReturn:
        self.transformer_block.convert_decoder_only_to_encoder_decoder()
        if self.discriminator_enabled:
            self.discriminator.convert_decoder_only_to_encoder_decoder()

    def init_tokenizer(self,
                        sample:str = "the quick brown fox jumps over the lazy dog.THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG?!@#$%^&*()`~-_+=[{]}\\|\"':;/.>,<1234567890\t\n\f\r",
                        append_eos: Optional[bool] = False,
                        target_vocab_size: Optional[int] = 2**17,
                        min_occ: Optional[int] = 1,
                        max_occ: Optional[int] = 1000,
                        reserved_tokens: Optional[list] = [
                                                            '[pad]','[unk]',
                                                            '[sos]','[eos]',
                                                            '[copy]',
                                                            '[mask]',
                                                            '[segment_seperator]',
                                                            '[non_text_content]','[/non_text_content]'
                                                            ],
                        eos_index: Optional[int] = 3,
                        unk_index: Optional[int] = 1,
                        pad_idx: Optional[int] = 0,
                        return_tokenizer: Optional[bool] = False
                        ) -> Union[NoReturn,torchnlp.encoders.text.text_encoder.TextEncoder] :
        sample += " ".join(sample.split(""))
        self.tokenizer = SubwordEncoder(sample,
                                        append_eos=append_eos,
                                        target_vocab_size=target_vocab_size,
                                        min_occurrences=min_occ,
                                        max_occurrences=max_occ,
                                        reserved_tokens=reserved_tokens,
                                        eos_index=eos_index,
                                        unknown_index=unk_index,
                                        padding_index=pad_idx)
        self.vocab = self.tokenizer.vocab
        if return_tokenizer:
            return self.tokenizer

    def random_mask_shuffle_encoder(self,
                                    inp: Tensor,
                                    mask: bool = True,
                                    mask_percentage: float = 15.0,
                                    mask_together_nos: int = 3,
                                    mask_continuous_pos: float = -101.0,
                                    shuffle: bool = True,
                                    shuffle_percentage: float = 15,
                                    shuffle_together_nos: int = 3,
                                    shuffle_continuous_pos: float = -101
                                ) -> Tensor:
        inp_2: Tensor = inp.clone().detach()
        index_to_be_trained_on = []

        count: int = 0
        together_count: int = 0
        for j in range(inp.size(1)):
            if not shuffle:
                break
            rnd: float = -1
            if shuffle_continuous_pos < -100 or shuffle_continuous_pos > 100:
                rnd: float = random.randint(0,100000)/1000
            elif shuffle_continuous_pos >= -100 and shuffle_continuous_pos <= 100:
                shuffle_together_nos = shuffle_percentage * (inp.size(1)/100)
                if shuffle_continuous_pos < 0:
                    if (((j+1)/inp.size(1)) + (shuffle_percentage/100)) >= ((inp.size(1)+((shuffle_continuous_pos/100)*inp.size(1)))/inp.size(1)):
                        rnd: float = shuffle_percentage/2
                else:
                    if (j+1)/inp.size(1) >= shuffle_continuous_pos/100:
                        rnd: float = shuffle_percentage/2
            if (((rnd>=0 and rnd<shuffle_percentage) or (together_count<shuffle_together_nos and together_count!=0)) and shuffle and (((count+1)/inp.size(1))<=shuffle_percentage/100)):
                while True:
                    r = random.randint(0,inp.size(1)-1)
                    if r!=j:
                        break
                if j not in index_to_be_trained_on:
                    index_to_be_trained_on.append(j)
                if r not in index_to_be_trained_on:
                    index_to_be_trained_on.append(r)
                inp_2[:,j],inp_2[:,r] = inp[:,r],inp[:,j]
                count += 1
                together_count += 1
            elif together_count>=shuffle_together_nos:
                together_count = 0

        count: int = 0
        together_count: int = 0
        for j in range(inp.size(1)):
            rnd: float = -1
            if mask_continuous_pos < -100 or mask_continuous_pos > 100 or mask_continuous_pos==None:
                rnd: float = random.randint(0,100000)/1000
            elif mask_continuous_pos >= -100 and mask_continuous_pos <= 100:
                mask_together_nos = mask_percentage * (inp.size(1)/100)
                if mask_continuous_pos < 0:
                    if (((j+1)/inp.size(1)) + (mask_percentage/100)) >= ((inp.size(1)+((mask_continuous_pos/100)*inp.size(1)))/inp.size(1)):
                        rnd: float = mask_percentage/2
                else:
                    if ((j+1)/inp.size(1)) >= mask_continuous_pos/100:
                        rnd: float = mask_percentage/2
            if (((rnd>=0 and rnd<mask_percentage) or (together_count<mask_together_nos and together_count!=0)) and mask and (((count+1)/inp.size(1))<=mask_percentage/100)):
                for i in range(inp.size(0)):
                    inp_2[i,j] = 5
                if j not in index_to_be_trained_on:
                    index_to_be_trained_on.append(j)
                count += 1
                together_count += 1
            elif together_count>=mask_together_nos:
                together_count = 0
        for _ in range(inp_2.size(1)//20):
            rnd = random.randint(0,inp_2.size(1)-1)
            if rnd not in index_to_be_trained_on:
                index_to_be_trained_on.append(rnd)
        index_to_be_trained_on = list(set(index_to_be_trained_on))
        index_to_be_trained_on = list(set(index_to_be_trained_on.extend(rnd)))
        del(inp_2,inp)
        torch.cuda.empty_cache()
        return out,index_to_be_trained_on

    def encode_text(self,
                        *args: Union[str,Tensor],
                        append_pad_at_start: Union[bool,int] = False,
                        append_pad_at_end: Union[bool,int] = False,
                        padding: Union[int,bool] = 0,
                        pad_idx: int = 0,
                        append_sos: bool = True,
                        sos_idx: int = 2,
                        append_eos: bool = True,
                        eos_idx: int = 3,
                        concatenate_all: bool = False,
                        concatenate_dim: int = 1,
                        append_segment_seperator: bool = True,
                        segment_idx: int = 6,
                        mask_at_random: bool = True,
                        mask_percentage: float = 15.,
                        mask_together_nos: int = 3,
                        mask_continuous_pos: float = -101.,
                        shuffle_at_random: bool = True,
                        shuffle_percentage: float = 15.,
                        shuffle_together_nos: int = 3,
                        shuffle_continuous_pos: float = -101.
                    ) -> List[Tensor] :
        encoded_text = []
        trainable_index = []
        for txt in args:
            if type(txt) == str:
                tmp = self.tokenizer.encode(txt)
            else:
                assert type(tmp) == Tensor
            if mask_at_random or shuffle_at_random:
                tmp,index_to_be_trained_on =   self.random_mask_shuffle_encoder(tmp,
                                                            mask=mask_at_random,
                                                            mask_percentage=mask_percentage,
                                                            mask_together_nos=mask_together_nos,
                                                            mask_continuous_pos=mask_continuous_pos,
                                                            shuffle=shuffle_at_random,
                                                            shuffle_percentage=shuffle_percentage,
                                                            shuffle_together_nos=shuffle_together_nos,
                                                            shuffle_continuous_pos=shuffle_continuous_pos
                                                        )

            trainable_index.append(index_to_be_trained_on)

            if append_sos:
                tmp = torch.cat((torch.full((tmp.size(0),1),sos_idx,dtype=torch.long,device=device),tmp),dim=1).contiguous()

            if append_eos:
                tmp = torch.cat((tmp,torch.full((tmp.size(0),1),eos_idx,dtype=torch.long,device=device)),dim=1).contiguous()

            if append_pad_at_end or append_pad_at_end:
                if type(append_pad_at_start) == int:
                    tmp = torch.cat((torch.full((tmp.size(0),append_pad_at_start),pad_idx,dtype=torch.long,device=self.device),tmp),dim=1)
                if type(append_pad_at_end) == int:
                    tmp = torch.cat((tmp,torch.full((tmp.size(0),append_pad_at_end),pad_idx,dtype=torch.long,device=self.device)),dim=1)
                if type(append_pad_at_start) == bool and type(append_pad_at_end) == bool:
                    if padding%2==0:
                        pad_l = pad_r = padding//2
                    else:
                        pad_l = padding//2
                        pad_r = (padding//2) + 1
                    tmp = torch.cat((torch.full((tmp.size(0),pad_l),pad_idx,dtype=torch.long,device=self.device),tmp,torch.full((tmp.size(0),pad_r),pad_idx,dtype=torch.long,device=self.device)),dim=1)
                elif not (type(append_pad_at_end) == bool and type(append_pad_at_start) == bool):
                    tmp = torch.cat((torch.full((tmp.size(0),padding),pad_idx,dtype=torch.long,device=self.device),tmp),dim=1)

            if append_segment_seperator:
                tmp = torch.cat((tmp,torch.tensor([[segment_idx]])),dim=1)
            encoded_text.append(tmp)
        
        if concatenate_all:
            encoded_text = [torch.cat(encoded_text,dim=concatenate_dim)]
        return encoded_text,trainable_index

    def decode_text(self,
                        *args: Tensor,
                        to_text: bool = True
                        ) -> list:
        decoded_text = []
        for txt in args:
            if type(txt) != Tensor:
                txt = torch.tensor(txt)

            if txt.size(-1) == self.ntokens and len(txt.size()) == 3:
                txt = torch.argmax(txt,dim=-1)

            if to_text:
                if txt.size(0)>1:
                    tmp = []
                    for i in txt:
                        tmp.append(self.tokenizer.decode(i))
                    txt = tmp
                else:
                    txt = self.tokenizer.decode(txt)
            decoded_text.append(txt)
        return decoded_text

    def init_optimizer(self,
                            opt: Optional[torch.optim.Optimizer] = None,
                            lr: Union[None,float,dict] = None,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            lambdaLR: Optional[typing.Callable] = None,
                            return_opt_schd: bool = False
                        ) -> Union[NoReturn,Tuple[torch.optim.Optimizer,torch.optim.lr_scheduler._LRScheduler]]:
        if opt != None:
            self.optimizer = opt
        else:
            lr = 1 if lr==None else lr
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        if scheduler != None:
            self.scheduler = scheduler
        else:
            if (lambdaLR == None or lambdaLR_disc == None) and self.scheduler_lambda == None:

                def lambda_lr(step_):
                    a = 5000000
                    b = 1000
                    c = 0.0
                    step = 1
                    multiplier = (bptt/512)*batch_size

                    def sub_func(step):
                        return (((a/b * (multiplier*step) + 1) / ((multiplier*step)**2 + a)) + c)/((step*(multiplier/200))**0.1+1)

                    if step_<(1024/(multiplier**(math.pi*2/10))):
                        return sub_func(step_)
                    elif step_<(2048/(multiplier**(math.pi*2/10))):
                        return sub_func(step_) / 25
                    else:
                        return sub_func(step_) / 625

                lambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda_lr)
            else:
                lambdaLR = self.scheduler_lambda if type(self.scheduler_lambda)==list else self.scheduler_lambda[-1]

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,lr_lambda=lambdaLR)
            self.scheduler_lambda = lambda_LR

        if return_opt_schd:
            return self.optimizer,self.scheduler

    def training_step(self,
                    data,
                    targets,
                    loss_criterion,
                    mem_tokens=None,
                    opt=None,
                    grad_clip=0.5,
                    deepspeed_enabled=False,
                    autocast_enabled=False,
                    trainable_index=None,
                    mem_ctxt=None,
                    mini_batch_size=None,
                    batch=None,
                ):

        self.train()
        step_start_time = time.time()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

        if opt == None:
            optimizer = self.optimizer
        else:
            optimizer = opt

        torch.cuda.empty_cache()

        def step_optimizer(input_data=data,output_targets=targets):
            outputs = {}
            losses = {}
            labels = {}
            output,single_pass_mem,single_pass_mem_ctxt = self.forward(input_data,mem=mem_tokens,context_mem=mem_ctxt)
            torch.cuda.empty_cache()
            if trainable_index != None:
                trainable_output = torch.cat([output[:,i:i+1] for i in trainable_index],dim=1)
                trainable_output_targets = torch.cat([output_targets[:,i:i+1] for i in trainable_index],dim=1)
            else:
                trainable_output = output
                trainable_output_targets = output_targets
                
            loss = loss_criterion(trainable_output.permute(1,2,0).contiguous(), trainable_output_targets.permute(1,0).contiguous())
            
            loss.backward(retain_graph=True)
            torch.cuda.empty_cache()
            if mini_batch_size != None and batch != None and batch%mini_batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            outputs['output'] = output

            losses['loss'] = loss.item()
            return outputs,losses,labels,single_pass_mem,single_pass_mem_ctxt
        
        if deepspeed_enabled or autocast_enabled:
            with autocast():
                outputs,losses,labels,mem_,mem_ctxt_ = step_optimizer(data,targets)
        else:
            outputs,losses,labels,mem_,mem_ctxt_ = step_optimizer(data,targets)

        acc = ((torch.argmax(outputs['output'],dim=-1)) == targets).sum().item()/outputs['output'].size(1)
        loss = losses['loss']

        if mem_ != None:
            pass
            mem_ = mem_.clone().detach()
        if mem_ctxt_ != None:
            pass
            mem_ctxt_ = mem_ctxt_.clone().detach()

        return outputs,losses,loss,acc,(step_start_time-time.time()),mem_,mem_ctxt_
        
    def get_prev_state(self) -> List[Tensor]:
        prev_states = {0:self.transformer_block.prev_state}
        modules = find_modules(self.transformer_block,Attention)
        for i,attn in enumerate(modules):
            prev_states[i+1] = attn.prev_state
        return prev_states

    def set_prev_state(self,prev_state:List[Tensor]):
        self.transformer_block.prev_state = prev_state[0]
        modules = find_modules(self.transformer_block,Attention)
        for i,attn in enumerate(modules):
            attn.prev_state = prev_states[i+1]

    #@autocast()
    def forward(self,
                    src:Union[Tensor,Dict[str,Tensor]],
                    context: Optional[Tensor] = None,
                    mem: Union[Tensor,None,Dict[str,Tensor]] = None, 
                    context_mem: Optional[Tensor] = None,
                    return_mem: bool = True,
                    return_logits: bool = False,
                    seq_scale_down: bool = True,
                ) -> Union[Tensor,Tuple[Tensor,Optional[Tensor],Optional[Tensor]]]:

        start_time = time.time()

        if type(src)==dict:
            try:
                context = src["context"]
            except:
                pass
            src = src["src"]
        
        if type(mem)==dict:
            try:
                context_mem = mem["context_mem"]
            except:
                pass
            mem = mem["mem"]

        self.prev_states.append(self.get_prev_state())
        while len(self.prev_states) > self.max_prev_states:
            tmp = self.prev_states.pop(0)
            del(tmp)

        (b,s) = src.shape
        
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()

        output = self.embedding_encoder(src)
        output = output * math.sqrt(self.ninp)
        output = ckpt(self.ffd1,output)

        if self.encoder_decoder:
            context = self.embedding_encoder(context)
            context = context * math.sqrt(self.ninp)
            context = ckpt(self.ffd1,context)

        output = Positional_Encoding(output)
        if self.encoder_decoder:
            context = Positional_Encoding(context)

        if seq_scale_down:
            output = ckpt(self.scale_down_conv,output.transpose(-1,-2)).transpose(-1,-2)
            if self.encoder_decoder:
                context = ckpt(self.scale_down_conv,context.transpose(-1,-2)).transpose(-1,-2)
                
        mem = default(mem,self.mem)
        if exists(mem):
            if (mem.size(-1) == output.size(-1) and len(mem.shape) == 2): 
                output = torch.cat((Positional_Encoding(repeat(mem, 'n d -> b n d', b = output.size(0))),output),dim=-2)
        if self.encoder_decoder:
            context_mem = default(context_mem,self.mem)
            if exists(context_mem):
                if (context_mem.size(-1) == context.size(-1) and len(context_mem.shape) == 2): 
                    context = torch.cat((Positional_Encoding(repeat(context_mem, 'n d -> b n d', b = context.size(0))),context),dim=-2)

        output = self.transformer_block(output,context)
        if isinstance(output,tuple):
            context = output[1]
            output = output[0]

        if exists(mem):
            mem = output[:,:mem.size(0)]
            mem = torch.sum(mem,dim=0,keepdim=True).reshape(-1,output.size(-1)) / b
            mem = mem.detach()
            output = output[:,mem.size(0):]

        if exists(context):
            if exists(context_mem):
                context_mem = context[:,:context_mem.size(0)]
                context_mem = torch.sum(mem,dim=0,keepdim=True).reshape(-1,output.size(-1)) / b
                context_mem = context_mem.detach()
                context = context[:,context_mem.size(0):]
        
        if seq_scale_down:
            output = ckpt(self.scale_up_conv,output.transpose(-1,-2)).transpose(-1,-2)
            output = output[:,:-(self.seq_scale_down*3 -1)]

            if context != None:
                context = ckpt(self.scale_up_conv,context.transpose(-1,-2)).transpose(-1,-2)
                context = context[:,:-(self.seq_scale_down*3 -1)]
                
        output = ckpt(self.ffd2,output)
        out = ckpt(self.decoder,output)
        
        self.time_[0] += time.time()-start_time
        self.time_[1] += 1

        out = out if not return_logits else [out,output]

        if return_mem:
            return out, mem, context_mem
        else:
            return out


class TransformerX(Module):

    @profile
    def __init__(self, **config: dict) -> NoReturn :
        super(TransformerX, self).__init__()

        self.time_: Union[float,int] =                              [(0,1)]
        self.model_type: str =                                      config.get("Project Name", "Transformer-X")
        self.dim_hidden: int =                                      config.get('dim_hidden')
        self.vocab: int =                                           config.get('vocab',256)
        self.dim_ffd_mult: int =                                    config.get('dim_ffd_mult',4)
        self.num_heads: int =                                       config.get('num_heads',8)
        self.num_local_heads: int =                                 config.get('num_local_heads',2)
        self.dropout: Union[float,int] =                            config.get('dropout',math.pi/10)
        self.max_seq_len: int =                                     config.get('max_seq_len',2**17)
        self.activation: str =                                      config.get('activation','Lrelu')
        self.discriminator: bool =                                  config.get('discriminator',False)
        self.discriminator_classes: int =                           config.get('discriminator_classes',2)
        self.num_layers: int =                                      config.get('num_layers',12)
        self.num_max_hop: Optional[int] =                           config.get('num_max_hop',None)
        self.exists_embedding: bool =                               config.get('embedding_encoder',True)
        self.fno_layers: int =                                      config.get("fno_layers",4)
        self.modes: Optional[int] =                                 config.get("modes",None)
        self.width: Optional[int] =                                 config.get("width",None)
        self.exists_head: bool =                                    config.get('logits_head',True)
        self.final_logits_activation: Optional[str] =               config.get('final_logits_activation',None)
        self.seq_scale_down: int =                                  config.get('seq_scale_down',1)
        self.mem_parameters: int =                                  config.get('mem_parameters',128)
        self.num_mem_static: int =                                  config.get("num_mem_static",1024)
        self.num_mem_dyn: int =                                     config.get("num_mem_dyn",1024)
        self.max_prev_states: int =                                 config.get('max_prev_states_storage',3)
        self.ET: bool =                                             config.get('ET',True)
        self.attn: str =                                            config.get('attn','h_attn')
        self.feature_redraw_interval: int =                         config.get('feature_redraw_interval',256)
        self.auto_check_redraw: bool =                              config.get('auto_check_redraw',True)
        self.causal: bool =                                         config.get('causal',False)
        self.attend_to_self =                                       config.get("attend_to_self",True)
        self.mlp_layers: int =                                      config.get("mlp_layers",1)
        self.pkm_keys: Optional[int] =                              config.get("pkm_keys",None)
        self.enable_hopfield: bool =                                config.get("enable_hopfield",True)
        self.mem_kv: Optional[int] =                                config.get("mem_kv",None)
        self.prev_state_kv: Optional[int] =                         config.get("prev_state_kv",None)
        self.num_prev_mem: Optional[int] =                          config.get("num_prev_mem",None)
        self.init_value: Optional[float] =                          config.get("init_value",0.01)

        Tfn_part = partial(TransformerBlock,
                        d_model = self.dim_hidden,
                        nhead = self.num_heads,
                        dim_ffd_mult = self.dim_ffd_mult,
                        dropout = self.dropout,
                        activation = self.activation,
                        mem_kv = self.mem_kv,
                        fno_layers = self.fno_layers,
                        pkm_keys = self.pkm_keys,
                        hopfield = self.enable_hopfield,
                        modes = self.modes,
                        width = self.width,
                        causal = self.causal,
                        local_heads = self.num_local_heads,
                        attn = self.attn,
                        attend_to_self = self.attend_to_self,
                        mlp_layers = self.mlp_layers,
                        ET = self.ET,
                        prev_state_kv = self.prev_state_kv,
                        num_prev_mem = self.num_prev_mem,
                        context=False,
                        decoder=False,
                        )
        
        self.transformer_layers = nn.ModuleList([ACT_basic(self.dim_hidden,Tfn_part()) for _ in range(self.num_layers)])
        self.dynamic_memory = nn.ModuleList([Dynamic_Memory(dim=self.dim_hidden,num_mem_static=self.num_mem_static,num_mem_dyn=self.num_mem_dyn) for _ in range(self.num_layers)])
        
        # byte_list = [i for i in str.encode("utf-32")]
        # max(byte_list) == 255
        # min(byte_list) == 0
        # str_from_bytes = bytes([ try: int_byte_list[i:i+4] for i in range(int_byte_list,4)]).decode("utf-32")
        # str == str_from_bytes --> True
        self.embedding_encoder = nn.Sequential(
                                                nn.Embedding(self.vocab, self.dim_hidden),
                                                #nn.Linear(self.dim_hidden,self.dim_hidden),
                                                #Rearrange("... (x l) y -> ... x (l y)",l=4),
                                                #nn.Linear(self.dim_hidden*4,self.dim_hidden),
                                                ) if self.exists_embedding else Identity()
        
        if self.exists_head:
            if not self.discriminator:
                self.decoder = nn.Sequential(
                    #nn.Linear(self.dim_hidden,self.dim_hidden*4),
                    #Rearrange("... x (l y) -> ... (x l) y",l=4),
                    #nn.Linear(self.dim_hidden,self.dim_hidden),
                    nn.Linear(self.dim_hidden,self.vocab) if not self.discriminator else nn.Linear(self.dim_hidden,self.discriminator_classes),
                    _get_activation_fn(self.final_logits_activation) if exists(self.final_logits_activation) else Identity(),
                    )
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.dim_hidden,self.discriminator_classes),
                    _get_activation_fn(self.final_logits_activation) if exists(self.final_logits_activation) else Identity(),
                    )
        else:
            self.decoder = Identity()

        self.ffd1 = FFd(dim=self.dim_hidden,mult=self.dim_ffd_mult,activation=self.activation)
        self.ffd2 = FFd(dim=self.dim_hidden,mult=self.dim_ffd_mult,activation=self.activation)

        self.mem_exist = True if self.mem_parameters else False
        self.mem = nn.Parameter(torch.randn(self.mem_parameters,self.dim_hidden)) if self.mem_exist else None
            
        self.attn_len = (self.seq_scale_down*3 // 2)*2 + 1
        self.scale_down_conv = nn.Conv1d(self.dim_hidden,self.dim_hidden,self.seq_scale_down*3,self.seq_scale_down,groups=1) if self.seq_scale_down > 1 else None
        self.scale_up_conv = nn.ConvTranspose1d(self.dim_hidden,self.dim_hidden,self.seq_scale_down*3,self.seq_scale_down,groups=1) if self.seq_scale_down > 1 else None

        self.proj_updater = ProjectionUpdater(nn.ModuleList([self.transformer_layers,self.dynamic_memory]), self.feature_redraw_interval)

        self.prev_states = []

        self.init_weights(self.init_value)

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_avg_inference_time(self) -> float:
        sum_ = 0
        for (tp,ln) in self.time_:
            sum_ += tp/(ln/1024)
        return sum_/(len(self.time_)-1)

    def fix_projection_matrices_(self) -> NoReturn :
        self.proj_updater.feature_redraw_interval = None

    def defix_projection_matrices_(self,feature_redraw_interval=None) -> NoReturn :
        self.proj_updater.feature_redraw_interval = feature_redraw_interval if exists(feature_redraw_interval) else self.feature_redraw_interval

    def init_weights(self,abs_value=0.01) -> NoReturn :
        for w in self.parameters():
            w.data.uniform_(-abs_value,abs_value)
   
    def get_prev_state(self) -> List[Tensor]:
        prev_states = []
        modules = find_modules(self.transformer_layers,Attention)
        modules.extend(find_modules(self.dynamic_memory,Dynamic_Memory))
        for i,mod in enumerate(modules):
            prev_states.append(mod.hidden_state_get())
        return prev_states

    def set_prev_state(self,prev_state:List[Tensor]) -> NoReturn :
        modules = find_modules(self.transformer_layers,Attention)
        modules.extend(find_modules(self.dynamic_memory,Dynamic_Memory))
        for i,mod in enumerate(modules):
            mod.hidden_state_set(prev_states[i+1])

    def training_step(self,
                    data,
                    targets,
                    loss_criterion,
                    opt=None,
                    grad_clip=0.5,
                    deepspeed_enabled=False,
                    autocast_enabled=False,
                    trainable_index=None,
                    mini_batch_size=None,
                    batch=None,
                ):

        self.train()
        step_start_time = time.time()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

        optimizer = opt
        torch.cuda.empty_cache()

        def step_optimizer(input_data=data,output_targets=targets):
            outputs = {}
            losses = {}
            labels = {}
            output,misc_losses = self.forward(input_data)
            torch.cuda.empty_cache()
            if trainable_index != None:
                trainable_output = torch.cat([output[:,i:i+1] for i in trainable_index],dim=1)
                trainable_output_targets = torch.cat([output_targets[:,i:i+1] for i in trainable_index],dim=1)
            else:
                trainable_output = output
                trainable_output_targets = output_targets
                
            loss = loss_criterion(trainable_output.permute(1,2,0).contiguous(), trainable_output_targets.permute(1,0).contiguous())

            extra_loss = 0
            for r,nu,probs,m_l,m_l_loss in misc_losses:
                p_t = r + nu
                extra_loss += 0.001 * reduce(r + nu,'b (n o) -> o','mean',o=1).item()
                extra_loss += reduce(probs,'b (n o) -> o','mean',o=1).item() * 0.005 + m_l_loss * 0.05
            loss = loss + extra_loss
            loss.backward(retain_graph=True)
            torch.cuda.empty_cache()
            if mini_batch_size != None and batch != None and batch%mini_batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            outputs['output'] = output

            losses['loss'] = loss.item()
            return outputs,losses,labels
        
        if deepspeed_enabled or autocast_enabled:
            with autocast():
                outputs,losses,labels = step_optimizer(data,targets)
        else:
            outputs,losses,labels = step_optimizer(data,targets)

        acc = ((torch.argmax(outputs['output'],dim=-1)) == targets).sum().item()/outputs['output'].size(1)
        loss = losses['loss']

        return outputs,losses,loss,acc,(step_start_time-time.time())
        
    #@autocast()
    def forward(self,
                    src:Tensor,
                    return_logits: bool = False,
                    seq_scale_down: bool = True,
                ) -> Union[Tuple[Tensor,List[List]],Tuple[Tensor,Tensor,List[List]]]:

        start_time = time.time()

        seq_scale_down = False if not exists(self.scale_down_conv) else seq_scale_down

        self.prev_states.append(self.get_prev_state())
        while len(self.prev_states) > self.max_prev_states:
            tmp = self.prev_states.pop(0)
            del(tmp)

        (b,s) = src.shape
        
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()

        output = self.embedding_encoder(src)
        output = output * math.sqrt(self.dim_hidden)

        output = Positional_Encoding(output)
        output = ckpt(self.ffd1,output)

        if seq_scale_down:
            output = ckpt(self.scale_down_conv,F.pad(output.transpose(-1,-2),((self.seq_scale_down*3 - 1),0),value=0)).transpose(-1,-2)
                
        output = torch.cat((Positional_Encoding(repeat(self.mem, 'n d -> b n d', b = output.size(0))),output),dim=-2)

        misc_losses = []
        for attn,mem in zip(self.transformer_layers,self.dynamic_memory):
            (output,r,nu) = attn(output)
            (output,probs,m_l,m_l_loss) = mem(output)

            misc_losses.append([r,nu,probs,m_l,m_l_loss])

        output = output[:,self.mem.size(0):]

        if seq_scale_down:
            output = ckpt(self.scale_up_conv,output.transpose(-1,-2)).transpose(-1,-2)
            output = output[:,-s:]

        output = ckpt(self.ffd2,output)
        out = ckpt(self.decoder,output)
        
        self.time_.append((time.time()-start_time,out.size(-2)))

        out = out if not return_logits else [out,output]
        
        return out,misc_losses

class MultiModalTransformer(Module):

    @profile
    def __init__(self, **config: dict) -> NoReturn :
        super(MultiModalTransformer, self).__init__()

        self.time_: Union[float,int] =                              [0,0]
        self.model_type: str =                                      config.get("Project Name", "Multi Modal Transformer-X")
        self.dim_hidden: int =                                      config.get('dim_hidden')
        self.dim_ffd_mult: int =                                    config.get('dim_ffd_mult',4)
        self.num_heads: int =                                       config.get('num_heads',8)
        self.num_local_heads: int =                                 config.get('num_local_heads',2)
        self.dropout: Union[float,int] =                            config.get('dropout',math.pi/10)
        self.max_seq_len: int =                                     config.get('max_seq_len',2**17)
        self.activation: str =                                      config.get('activation','Lrelu')
        self.discriminator: bool =                                  config.get('discriminator',False)
        self.discriminator_classes: int =                           config.get('discriminator_classes',2)
        self.attend_in_patches: bool =                              config.get('attend_in_patches',True)
        self.project_to_new_sequence: bool =                        config.get('project_to_new_sequnce',True)
        self.trunk_width: Optional[int] =                           config.get('trunk_width',None)
        self.seperate_attender_exists: bool =                       config.get('seperate_attender_exists',True)
        self.decoder_type_attender: bool =                          config.get('decoder_type_attender',False)
        self.seperate_doer_exists: bool =                           config.get('seperate_doer_exists',True)
        self.decoder_type_doer: bool =                              config.get('decoder_type_doer',False)
        self.num_layers: int =                                      config.get('num_layers',(1,1,1))
        self.num_max_hop: Optional[int] =                           config.get('num_max_hop',(None,None,None))
        self.exists_embedding: bool =                               config.get('embedding_encoder',True)
        self.fno_layers: int =                                      config.get("fno_layers",4)
        self.modes: Optional[int] =                                 config.get("modes",None)
        self.width: Optional[int] =                                 config.get("width",None)
        self.exists_head: bool =                                    config.get('logits_head',True)
        self.final_logits_activation: Optional[str] =               config.get('final_logits_activation',None)
        self.seq_scale_down: int =                                  config.get('seq_scale_down',4)
        self.mem_parameters: int =                                  config.get('mem_parameters',128)
        self.prev_state_len: int =                                  config.get("prev_state_len",8192)
        self.prev_state_self_attend: int =                          config.get("prev_state_self_attend",32)
        self.max_prev_states: int =                                 config.get('max_prev_states_storage',3)
        self.ET: bool =                                             config.get('ET',True)
        self.attn: str =                                            config.get('attn','h_attn')
        self.feature_redraw_interval: int =                         config.get('feature_redraw_interval',256)
        self.auto_check_redraw: bool =                              config.get('auto_check_redraw',True)
        self.encoder_decoder: bool =                                config.get('encoder_decoder',False)
        self.causal: bool =                                         config.get('causal',False)
        self.attend_to_self =                                       config.get("attend_to_self",True)
        self.mlp_layers: int =                                      config.get("mlp_layers",1)
        self.pkm_keys: Optional[int] =                              config.get("pkm_keys",None)
        self.enable_hopfield: bool =                                config.get("enable_hopfield",True)
        self.encoder_in_decoder: bool =                             config.get("encoder_in_decoder",True)
        self.mem_kv: Optional[int] =                                config.get("mem_kv",None)
        self.prev_state_kv: Optional[int] =                         config.get("prev_state_kv",None)
        self.num_prev_mem: Optional[int] =                          config.get("num_prev_mem",None)
        self.init_value: Optional[float] =                          config.get("init_value",0.01)

        if not self.attend_in_patches:
            warnings.warn("attend_in_patches is set to False, for very large sequence lengths, the memory requirements will be too large.")

        Tfn_part = partial(TransformerBlock,
                        d_model = self.dim_hidden,
                        nhead = self.num_heads,
                        dim_ffd_mult = self.dim_ffd_mult,
                        dropout = self.dropout,
                        activation = self.activation,
                        mem_kv = self.mem_kv,
                        encoder_n_decoder = self.encoder_in_decoder,
                        fno_layers = self.fno_layers,
                        pkm_keys = self.pkm_keys,
                        hopfield = self.enable_hopfield,
                        modes = self.modes,
                        width = self.width,
                        causal = self.causal,
                        local_heads = self.num_local_heads,
                        attn = self.attn,
                        attend_to_self = self.attend_to_self,
                        mlp_layers = self.mlp_layers,
                        ET = self.ET,
                        prev_state_kv = self.prev_state_kv,
                        num_prev_mem = self.num_prev_mem,
                        )

        self.trunk_sequence = nn.Parameter(torch.zeros((1,self.trunk_width,self.dim_hidden))) if exist(self.trunk_width) else torch.zeros((1,self.max_seq_len,self.dim_hidden))
        self.trunk_pos_embedding = Positional_Encoding if exists(self.trunk_width) else AbsolutePositionalEmbedding(self.dim_hidden,self.max_seq_len)

        self.thinker = nn.ModuleList([ACT_basic(dim_hidden,Tfn_part(context=True,decoder=True)) for _ in range(num_layers[0])])

        self.attender = nn.ModuleList([ACT_basic(dim_hidden,Tfn_part(context=self.project_to_new_sequence,decoder=self.decoder_type_attender)) for _ in range(num_layers[0])])

        self.doer = nn.ModuleList([ACT_basic(dim_hidden,Tfn_part(context=self.project_to_new_sequence,decoder=self.decoder_type_doer)) for _ in range(num_layers[0])])
        

        # byte_list = [i for i in str.encode("utf-32")]
        # max(byte_list) == 255
        # min(byte_list) == 0
        # str_from_bytes = bytes([ try: int_byte_list[i:i+4] for i in range(int_byte_list,4)]).decode("utf-32")
        # str == str_from_bytes --> True
        self.embedding_encoder = nn.Sequential(
                                                nn.Embedding(256, self.dim_hidden),
                                                nn.Linear(self.dim_hidden,self.dim_hidden),
                                                Rearrange("... (x l) y -> ... x (l y)",l=4),
                                                nn.Linear(self.dim_hidden*4,self.dim_hidden),
                                                ) if self.exists_embedding else Identity()
        
        if self.exists_head:
            if not discriminator:
                self.decoder = nn.Sequential(
                    nn.Linear(self.dim_hidden,self.dim_hidden*4),
                    Rearrange("... x (l y) -> ... (x l) y",l=4),
                    nn.Linear(self.dim_hidden,self.dim_hidden),
                    nn.Linear(self.dim_hidden,256) if not self.discriminator else nn.Linear(self.dim_hidden,self.discriminator_classes),
                    _get_activation_fn(self.final_logits_activation) if exists(self.final_logits_activation) else Identity(),
                    )
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.dim_hidden,self.discriminator_classes),
                    _get_activation_fn(self.final_logits_activation) if exists(self.final_logits_activation) else Identity(),
                    )
        else:
            self.decoder = Identity()

        self.ffd1 = FFd(dim=self.dim_hidden,mult=self.dim_ffd_mult,activation=self.activation)
        self.ffd2 = FFd(dim=self.dim_hidden,mult=self.dim_ffd_mult,activation=self.activation)

        self.mem_exist = True if self.mem_parameters else False
        self.mem = nn.Parameter(torch.randn(self.mem_parameters,self.dim_hidden)) if self.mem_exist else None
            
        self.seq_scale_down = self.seq_scale_down
        self.attn_len = (self.seq_scale_down*3 // 2)*2 + 1

        self.scale_down_conv = nn.Conv1d(self.dim_hidden,self.dim_hidden,self.seq_scale_down*3,self.seq_scale_down,padding=(self.seq_scale_down*3 - 1)//2,groups=1)

        self.scale_up_conv = nn.ConvTranspose1d(self.dim_hidden,self.dim_hidden,self.seq_scale_down*3,self.seq_scale_down,groups=1)

        self.proj_updater = ProjectionUpdater(nn.ModuleList([self.attender,self.thinker,self.doer]), self.feature_redraw_interval)

        self.optimizer = None
        self.scheduler = None
        self.scheduler_lambda = None

        self.prev_states = []

        _ = self.forward(torch.randint(0,255,(1,self.max_seq_len//8),device=self.device))
        self.init_weights(self.init_value)

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_avg_inference_time(self) -> float:
        return self.time_[0] / self.time_[1]

    def fix_projection_matrices_(self) -> NoReturn :
        self.proj_updater.feature_redraw_interval = None

    def defix_projection_matrices_(self,feature_redraw_interval=None) -> NoReturn :
        self.proj_updater.feature_redraw_interval = feature_redraw_interval if exists(feature_redraw_interval) else self.feature_redraw_interval

    def init_weights(self,abs_value=0.01) -> NoReturn :
        for w in self.parameters():
            w.data.uniform_(-abs_value,abs_value)
   
    def get_prev_state(self) -> List[Tensor]:
        prev_states = []
        modules = find_modules(self,Attention)
        modules.append(find_modules(self,Dynamic_Memory))
        for i,mod in enumerate(modules):
            prev_states.append(mod.hidden_state_get())
        return prev_states

    def set_prev_state(self,prev_state:List[Tensor]) -> NoReturn :
        modules = find_modules(self,Attention)
        modules.append(find_modules(self,Dynamic_Memory))
        for i,mod in enumerate(modules):
            mod.hidden_state_set(prev_states[i+1])

    #@autocast()
    def forward(self,
                    src:Union[Tensor,Dict[str,Tensor]],
                    context: Optional[Tensor] = None,
                    mem: Union[Tensor,None,Dict[str,Tensor]] = None, 
                    context_mem: Optional[Tensor] = None,
                    return_mem: bool = True,
                    return_logits: bool = False,
                    seq_scale_down: bool = True,
                ) -> Union[Tensor,Tuple[Tensor,Optional[Tensor],Optional[Tensor]]]:

        start_time = time.time()

        if type(src)==dict:
            try:
                context = src["context"]
            except:
                pass
            src = src["src"]
        
        if type(mem)==dict:
            try:
                context_mem = mem["context_mem"]
            except:
                pass
            mem = mem["mem"]

        self.prev_states.append(self.get_prev_state())
        while len(self.prev_states) > self.max_prev_states:
            tmp = self.prev_states.pop(0)
            del(tmp)

        (b,s) = src.shape
        
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()

        output = self.embedding_encoder(src)
        output = output * math.sqrt(self.ninp)
        output = ckpt(self.ffd1,output)

        if self.encoder_decoder:
            context = self.embedding_encoder(context)
            context = context * math.sqrt(self.ninp)
            context = ckpt(self.ffd1,context)

        output = Positional_Encoding(output)
        if self.encoder_decoder:
            context = Positional_Encoding(context)

        if seq_scale_down:
            output = ckpt(self.scale_down_conv,output.transpose(-1,-2)).transpose(-1,-2)
            if self.encoder_decoder:
                context = ckpt(self.scale_down_conv,context.transpose(-1,-2)).transpose(-1,-2)
                
        mem = default(mem,self.mem)
        if exists(mem):
            if (mem.size(-1) == output.size(-1) and len(mem.shape) == 2): 
                output = torch.cat((Positional_Encoding(repeat(mem, 'n d -> b n d', b = output.size(0))),output),dim=-2)
        if self.encoder_decoder:
            context_mem = default(context_mem,self.mem)
            if exists(context_mem):
                if (context_mem.size(-1) == context.size(-1) and len(context_mem.shape) == 2): 
                    context = torch.cat((Positional_Encoding(repeat(context_mem, 'n d -> b n d', b = context.size(0))),context),dim=-2)

        output = self.transformer_block(output,context)
        if isinstance(output,tuple):
            context = output[1]
            output = output[0]

        if exists(mem):
            mem = output[:,:mem.size(0)]
            mem = torch.sum(mem,dim=0,keepdim=True).reshape(-1,output.size(-1)) / b
            mem = mem.detach()
            output = output[:,mem.size(0):]

        if exists(context):
            if exists(context_mem):
                context_mem = context[:,:context_mem.size(0)]
                context_mem = torch.sum(mem,dim=0,keepdim=True).reshape(-1,output.size(-1)) / b
                context_mem = context_mem.detach()
                context = context[:,context_mem.size(0):]
        
        if seq_scale_down:
            output = ckpt(self.scale_up_conv,output.transpose(-1,-2)).transpose(-1,-2)
            output = output[:,:-(self.seq_scale_down*3 -1)]

            if context != None:
                context = ckpt(self.scale_up_conv,context.transpose(-1,-2)).transpose(-1,-2)
                context = context[:,:-(self.seq_scale_down*3 -1)]
                
        output = ckpt(self.ffd2,output)
        out = ckpt(self.decoder,output)
        
        self.time_[0] += time.time()-start_time
        self.time_[1] += 1

        out = out if not return_logits else [out,output]

        if return_mem:
            return out, mem, context_mem
        else:
            return out




def Trainer(model,
                data,
                targets,
                loss_criterion,
                discriminator=None,
                total_acc=0.,
                total_acc_d=0.,
                total_loss=0.,
                total_loss_d=0.,
                mem_tokens=None,
                opt=None,
                opt_disc=None,
                grad_clip=4.0,
                deepspeed_enabled=False,
                autocast_enabled=False,
                trainable_index=None,
                mem_ctxt=None,
            ):

    model.train()
    step_start_time = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)

    if opt == None:
        optimizer = model.optimizer
    else:
        optimizer = opt

    if discriminator!=None:
        if opt_disc == None:
            optimizer_disc = discriminator.optimizer
        else:
            optimizer_disc = opt_disc
    else:
        optimizer_disc = None

    torch.cuda.empty_cache()

    def step_optimizer(input_data=data,output_targets=targets):
        outputs = {}
        losses = {}
        labels = {}
        if discriminator!=None:
            model.zero_grad()
            discriminator.zero_grad()
                
            real_label = torch.full(output_targets.size(),0,dtype=torch.long,device=device)
            real_label_gen = torch.full(input_data.size(),0,dtype=torch.long,device=device)
            fake_label = torch.full(input_data.size(),1,dtype=torch.long,device=device)

            labels['real_label'] = real_label
            labels['real_label_gen'] = real_label_gen
            labels['fake_label'] = fake_label

            #out_d_real = model.forward(output_targets.detach(),return_mem=False,discriminator=True,generator=False)
            out_d_real = discriminator(model.embedding_encoder(output_targets),return_mem=False)
            loss_d_real = loss_criterion(rearrange(out_d_real,'b n c -> n c b'), rearrange(real_label,'b n -> n b'))
            loss_d_real.backward()

            #out_gan = model.forward(input_data.detach(),mem=mem_tokens,return_mem=False,discriminator=True,context_mem=mem_ctxt)
            out_d_real = discriminator(model(input_data,return_logits=True,return_mem=False)[1],mem=mem_tokens,context_mem=mem_ctxt,return_mem=False)
            loss_d_fake = loss_criterion(rearrange(out_gan,'b n c -> n c b'), rearrange(fake_label,'b n -> n b'))
            loss_d_fake.backward()

            optimizer_disc.step()
            optimizer_disc.zero_grad()
            model.zero_grad()
            discriminator.zero_grad()

            #out_gan = model.forward(input_data.detach(),mem=mem_tokens,return_mem=False,discriminator=True,context_mem=mem_ctxt)
            out_gan = discriminator(model(input_data,return_logits=True,return_mem=False)[1],mem=mem_tokens,context_mem=mem_ctxt,return_mem=False)
            loss_gen = loss_criterion(rearrange(out_gan,'b n c -> n c b'), rearrange(real_label_gen,'b n -> n b'))
            loss_gen.backward()

            output,single_pass_mem,single_pass_mem_ctxt = model.forward(input_data,mem=mem_tokens,context_mem=mem_ctxt)
            if trainable_index != None:
                trainable_output = torch.cat([output[:,i:i+1] for i in trainable_index],dim=1)
                trainable_output_targets = torch.cat([output_targets[:,i:i+1] for i in trainable_index],dim=1)
            else:
                trainable_output = output
                trainable_output_targets = output_targets

            loss = loss_criterion(trainable_output.permute(1,2,0).contiguous(), trainable_output_targets.permute(1,0).contiguous()).to(model.device)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            discriminator.zero_grad()

            outputs['out_d_real'] = out_d_real
            outputs['out_gan'] = out_gan
            outputs['output'] = output

            losses['loss_d_real'] = loss_d_real.item()
            losses['loss_d_fake'] = loss_d_fake.item()
            losses['loss_gen'] = loss_gen.item()
            losses['loss'] = loss.item()
        else:
            model.zero_grad()
            output,single_pass_mem,single_pass_mem_ctxt = model.forward(input_data,mem=mem_tokens,context_mem=mem_ctxt)
            torch.cuda.empty_cache()
            if trainable_index != None:
                trainable_output = torch.cat([output[:,i:i+1] for i in trainable_index],dim=1)
                trainable_output_targets = torch.cat([output_targets[:,i:i+1] for i in trainable_index],dim=1)
            else:
                trainable_output = output
                trainable_output_targets = output_targets
                
            loss = loss_criterion(trainable_output.permute(1,2,0).contiguous(), trainable_output_targets.permute(1,0).contiguous())
            loss.backward()
            torch.cuda.empty_cache()

            optimizer.step()
            optimizer.zero_grad()
            outputs['output'] = output

            losses['loss'] = loss.item()
        return outputs,losses,labels,single_pass_mem,single_pass_mem_ctxt
    
    if deepspeed_enabled or autocast_enabled:
        with autocast():
            outputs,losses,labels,mem_,mem_ctxt_ = step_optimizer(data,targets)
    else:
        outputs,losses,labels,mem_,mem_ctxt_ = step_optimizer(data,targets)
    
    acc_gen = 0.0
    loss_g = 0.0
    loss_d = 0.0

    if model.discriminator_enabled:
        acc_gen = ((torch.argmax(outputs['output'],dim=-1)) == targets).sum().item()/outputs['output'].size(1)
        acc_d = ((torch.argmax(outputs['out_d_real'],dim=-1)) == labels['real_label']).sum().item()/outputs['out_d_real'].size(1)
        acc_d += ((torch.argmax(outputs['out_gan'],dim=-1)) == labels['fake_label']).sum().item()/outputs['out_gan'].size(1)
        acc = ((torch.argmax(outputs['out_gan'],dim=-1)) == labels['real_label_gen']).sum().item()/outputs['out_gan'].size(1)
        acc += acc_gen

        total_acc += acc/2
        total_acc_d += acc_d/2
        loss_g = (losses['loss'] + losses['loss_gen'])/2
        total_loss += loss_g
        loss_d = (losses['loss_d_fake'] + losses['loss_d_real'])/2
        total_loss_d += loss_d
    else:
        acc_gen = ((torch.argmax(outputs['output'],dim=-1)) == targets).sum().item()/outputs['output'].size(1)
        total_acc += acc_gen
        total_loss += losses['loss']
        loss_g = losses['loss']
        total_acc_d = 0
        total_loss_d = 0

    return outputs,losses,total_acc,total_acc_d,total_loss,total_loss_d,loss_g,loss_d,acc_gen,(step_start_time-time.time()),optimizer,optimizer_disc,mem_,mem_ctxt_
    