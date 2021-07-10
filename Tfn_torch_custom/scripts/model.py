from json import decoder
import math
import time
#import profile
import torch
from torch import tensor
import torch.nn as nn
import random
import torch.nn.functional as F
import time
import warnings

from memory_profiler import profile

import copy
import typing
from typing import Tuple, Optional, Any, NoReturn, Union, List, Dict, Iterable

from torch import Tensor
from torch.nn.modules.container import ModuleList, Module
from torch.nn.modules.dropout import Dropout

from einops import repeat,rearrange
from mogrifier import Mogrifier

from .evolved_transformer_block import ET_Encoder_Block,ET_Decoder_Block, GLU
from .product_key_memory import PKM
from .hopfield_modules import HopfieldLayer
from .attention_layer_s import Attention,ProjectionUpdater,find_modules
from .conformer import ConformerConvModule

from .fourier_1d import FNO1d

from .g_mlp_gpt import gMLPGPT

from torchnlp.encoders.text import SubwordEncoder
import torchnlp

from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
autocast = torch.cuda.amp.autocast

checkpointed = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def ckpt(f,*args,checkpointed = checkpointed):
    if checkpointed:
        return checkpoint(f,*args)
    else:
        return f(*args)
        
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()

    raise RuntimeError("activation should be relu/gelu/silu, not {}".format(activation))

def Positional_Encoding(x: Tensor) -> Tensor :
    max_len = x.size(1)
    d_model = x.size(2)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).to(x.device)
    return x + pe

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
            GLU(dim_in,layers),
            nn.Linear(dim_in, dim_out * 2),
            )

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)   

class nBRC(nn.Module):
    def __init__(self, dims, hidden_dims):
        super().__init__()
        self.Ua = nn.Linear(dims, hidden_dims)
        self.Wa = nn.Linear(dims, hidden_dims)
        self.Uc = nn.Linear(dims, hidden_dims)
        self.Wc = nn.Linear(dims, hidden_dims)
        self.U  = nn.Linear(dims, hidden_dims)

    def forward(self, x, h):
        l = lambda linear, tensor: F.linear(tensor, linear.weight.clone(), linear.bias.clone())

        a = 1 + torch.tanh(l(self.Ua, x) + l(self.Wa, h))
        c = torch.sigmoid(l(self.Uc, x) + l(self.Wc, h))
        return c * h + (1 - c) * torch.tanh(l(self.U, x) + a * h)

class GRUGating(nn.Module):
    def __init__(self, dim, fn=None, mogrify = True, norm = True, post_norm = False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.post_norm = post_norm
        if norm:
            self.norm = RMSNorm(dim)
        else:
            self.norm = None
        self.gru = nBRC(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k = dim // 4) if mogrify else None

    def forward(self, x, y=None,*args):
        shape = x.shape
        dim = self.dim

        if y==None:
            y = x

        if self.fn != None:
            if self.post_norm:
                y_ = ckpt(self.fn,y,*args)

                y = self.norm(y_) if self.norm != None else y_
            else:
                inp = self.norm(y) if self.norm != None else y
                y = ckpt(self.fn,inp,*args)

        else:
            y = self.norm(y) if self.norm!=None else y

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = ckpt(self.gru,
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )
        out = gated_output.reshape(shape)

        return out

class ET_ffd(Module):

    def __init__(self,dim,activation="sigmoid",layers=1,kernal_size=3,mult=4,fn=None,sequential=False):
        super(ET_ffd,self).__init__()
        self.l_1 = nn.Sequential(
            nn.Conv1d(dim,dim*mult,kernal_size,1,padding=kernal_size//2,groups=1),
            _get_activation_fn(activation),
        )
        self.l_2 = GEGLU(dim*mult,dim,layers)

        self.fn = fn if isinstance(fn,nn.ModuleList) else ((nn.ModuleList(fn) if isinstance(fn,Iterable) else nn.ModuleList([i for i in fn if i != None])) if fn !=None else None)
        self.seq = sequential

    def forward(self,x,*args):
        out = ckpt(self.l_1,x.transpose(1,2)).transpose(1,2).contiguous()
        out = ckpt(self.l_2,out)
        if self.fn != None:
            for fn in self.fn:
                if self.seq:
                    inp = out
                else:
                    inp = x
                out = out + ckpt(fn,inp)
        return out

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len
        self.init_()

    def init_(self):
        for w in self.parameters():
            w.data.uniform_(-1/4,1/4)

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
            assert tmp.size(1) == s
            return tmp


class TransformerBlock(Module):

    #@profile
    def __init__(self,
                     d_model=128,
                     nhead=4, 
                     dim_feedforward=512, 
                     dropout=0.1, 
                     activation="relu",
                     mem_kv=1024,
                     pkm_dims=None,
                     pkm_keys=64,
                     decoder=False,
                     encoder_n_decoder=False,
                     hopfield=False,
                     hop_dim=None,
                     fno_layers=4,
                     modes=None,
                     width=None,
                     rotary_pos_emb=True,
                     fixed_emb=False,
                     causal=False,
                     local_heads=0,
                     nystrom=False,
                     attend_to_self=True,
                     mlp_layers=1,
                     use_mask=True,
                     context=True,
                ):
        super(TransformerBlock, self).__init__()
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if pkm_dims==None:
            pkm_dims = d_model//1
        if hop_dim==None:
            hop_dim = d_model//1

        pkm_heads = max((nhead * pkm_dims) // d_model,1)
        hop_heads = max((nhead * hop_dim) // d_model,1)

        modes = nhead if modes == None else modes
        width = nhead if width == None else width

        pkm1 = nn.Sequential(
            nn.Linear(d_model,pkm_dims),
            PKM(pkm_dims,heads=pkm_heads,num_keys=pkm_keys),
            nn.Linear(pkm_dims,d_model),
            ) if pkm_dims!=0 else None

        conformer = ConformerConvModule(d_model,causal=True,dropout=dropout)

        fno = FNO1d(modes,
                        width,
                        inp_dim=d_model,
                        out_dim=d_model,
                        ffd_dim=dim_feedforward,
                        num_layers=fno_layers
                    )

        ffd1 = nn.Sequential(ET_ffd(dim=d_model,fn=[pkm1,conformer,fno]),Dropout(dropout))

        self.feed_forward = GRUGating(d_model,ffd1)

        self.to_out = copy.deepcopy(self.feed_forward)

        if hopfield:
            hop_attn = nn.Sequential(
                                    nn.Linear(d_model,hop_dim),
                                    HopfieldLayer(
                                                input_size=hop_dim,
                                                num_heads=hop_heads,
                                                pattern_size=2**7,
                                                dropout=dropout,
                                                quantity=2**7,
                                            ),
                                    nn.Linear(hop_dim,d_model)
                                    )
        else:
            hop_attn = None

        if not decoder:
            
            attn_block = ET_Encoder_Block(d_model,
                                num_heads=nhead,
                                attn=Attention(d_model,
                                                    heads=nhead,
                                                    dim_head=d_model//nhead,
                                                    num_mem_kv=mem_kv,
                                                    local_heads=local_heads,
                                                    hop_attn=hop_attn,
                                                    rotary_pos_emb=rotary_pos_emb,
                                                    fixed_emb=fixed_emb,
                                                    causal=causal,
                                                    nystrom=nystrom,
                                                    attend_to_self=attend_to_self,
                                                    context=context,
                                                    use_mask=use_mask,
                                                ),
                                ffd=copy.deepcopy(self.feed_forward),
                                )
        else:
                
            self.ctxt_ffd = copy.deepcopy(self.feed_forward)

            self.self_inp_enc = GRUGating(
                                            d_model,
                                            ET_Encoder_Block(d_model,
                                                        num_heads=nhead,
                                                        attn=Attention(d_model,
                                                                            heads=nhead,
                                                                            dim_head=d_model//nhead,
                                                                            num_mem_kv=mem_kv,
                                                                            local_heads=local_heads,
                                                                            hop_attn=copy.deepcopy(hop_attn),
                                                                            rotary_pos_emb=rotary_pos_emb,
                                                                            fixed_emb=fixed_emb,
                                                                            causal=causal,
                                                                            nystrom=nystrom,
                                                                            attend_to_self=False,
                                                                            context=False,
                                                                            use_mask=use_mask,
                                                                        ),
                                                        ffd=copy.deepcopy(self.feed_forward),
                                                        ),norm=False) if encoder_n_decoder else Identity()

            self.self_ctxt_enc = GRUGating(
                                            d_model,
                                            ET_Encoder_Block(d_model,
                                                        num_heads=nhead,
                                                        attn=Attention(d_model,
                                                                            heads=nhead,
                                                                            dim_head=d_model//nhead,
                                                                            num_mem_kv=mem_kv,
                                                                            local_heads=local_heads,
                                                                            hop_attn=copy.deepcopy(hop_attn),
                                                                            rotary_pos_emb=rotary_pos_emb,
                                                                            fixed_emb=fixed_emb,
                                                                            causal=causal,
                                                                            nystrom=nystrom,
                                                                            attend_to_self=False,
                                                                            context=False,
                                                                            use_mask=use_mask,
                                                                        ),
                                                        ffd=copy.deepcopy(self.feed_forward),
                                                        ),norm=False) if encoder_n_decoder else Identity()

            attn = {
                'self_1':Attention(d_model,
                                        heads=nhead*2,
                                        dim_head=d_model//(nhead*2),
                                        num_mem_kv=mem_kv,
                                        hop_attn=hop_attn,
                                        local_heads=local_heads,
                                        rotary_pos_emb=rotary_pos_emb,
                                        fixed_emb=fixed_emb,
                                        causal=causal,
                                        nystrom=nystrom,
                                        attend_to_self=attend_to_self,
                                        context=False,
                                        use_mask=use_mask,),
                'self_2':Attention(d_model,
                                        heads=nhead,
                                        dim_head=d_model//nhead,
                                        num_mem_kv=mem_kv,
                                        hop_attn=copy.deepcopy(hop_attn),
                                        local_heads=local_heads,
                                        rotary_pos_emb=rotary_pos_emb,
                                        fixed_emb=fixed_emb,
                                        causal=causal,
                                        nystrom=nystrom,
                                        attend_to_self=False,
                                        context=False,
                                        use_mask=use_mask,),
                'cross_1':Attention(d_model,
                                        heads=nhead,
                                        dim_head=d_model//nhead,
                                        num_mem_kv=mem_kv,
                                        hop_attn=copy.deepcopy(hop_attn),
                                        rotary_pos_emb=rotary_pos_emb,
                                        nystrom=nystrom,
                                        attend_to_self=False,
                                        context=context,
                                        use_mask=bool(not context and use_mask),
                                        ),
                'cross_2':Attention(d_model,
                                        heads=nhead,
                                        dim_head=d_model//nhead,
                                        num_mem_kv=mem_kv,
                                        hop_attn=copy.deepcopy(hop_attn),
                                        nystrom=nystrom,
                                        rotary_pos_emb=rotary_pos_emb,
                                        attend_to_self=False,
                                        context=context,
                                        use_mask=bool(not context and use_mask),
                                        )
            }

            attn_block = ET_Decoder_Block(d_model,
                                num_heads=nhead,
                                attn=attn,
                                ffd=copy.deepcopy(self.feed_forward)
                                )
        
        self.attn = GRUGating(d_model,attn_block,norm=False)

        self.mlp = GRUGating(d_model,gMLPGPT(dim=d_model,depth=mlp_layers,seq_len=2**16,window=d_model,attn_dim=d_model//4,prob_survival=1-dropout),norm=False)

        self.decoder = decoder


    def forward(self, src: Tensor,context: Optional[Tensor] = None, src_mask: Tensor = None) -> Tensor:

        output = src
        #output = self.norm1(src)
        #output = Positional_Encoding(output)

        output = ckpt(self.feed_forward,output)

        if self.decoder:
            ctxt_mask = src_mask if context is None else None
            context = output if context == None else context
            #context = self.norm2(context)
            #context = Positional_Encoding(context)
            context = ckpt(self.ctxt_ffd,context)

            output = ckpt(self.self_inp_enc,output,None,src_mask)
            context = ckpt(self.self_ctxt_enc,context,None,ctxt_mask)

        output = ckpt(self.attn,output,output,context,src_mask)

        output = self.dropout1(output)

        output = ckpt(self.mlp,output)

        output = self.dropout2(output)

        output = self.to_out(output)

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
                    repeated_deberta_layers=2,
                    max_len=2**17,
                    prev_state_len=8192,
                    hop_dim=None,
                    pkm_dims=None,
                    fno_layers=4,
                    modes=None,
                    width=None,
                    full_block_repeat=False,
                    causal=False,
                    nystrom=False,
                    local_heads=2,
                    attend_to_self=True,
                    prev_state_self_num=32,
                    attend_to_inp=True,
                    mlp_layers=1,
                    encoder_n_decoder=True,
                    ):
        super(TransformerModule, self).__init__()

        # deprecated cross attention config saveing
        self.config = dict(d_model=d_model, nhead=nhead, dim_feedforward=nhid, dropout=dropout,decoder=True,hopfield=True,hop_dim=hop_dim,fno_layers=fno_layers,modes=modes,width=width,causal=causal,nystrom=nystrom,pkm_dims=pkm_dims,local_heads=local_heads,attend_to_self=attend_to_self,mlp_layers=mlp_layers)

        self.full_block_repeat = full_block_repeat
        self.enable_encoder=enable_encoder
        self.repeated_deberta_layers = repeated_deberta_layers

        if num_layers != 0:
            if not enable_encoder:
                block = TransformerBlock(d_model, nhead, nhid, dropout,hopfield=True,hop_dim=hop_dim,fno_layers=fno_layers,modes=modes,width=width,causal=causal,pkm_dims=pkm_dims,nystrom=nystrom,attend_to_self=attend_to_self,mlp_layers=mlp_layers,context=False)
                self.decoder_self = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(num_layers-1)])
            else:
                block = TransformerBlock(d_model, nhead, nhid, dropout,fno_layers=fno_layers,modes=modes,width=width,causal=causal,pkm_dims=pkm_dims,nystrom=nystrom,attend_to_self=attend_to_self,mlp_layers=mlp_layers,context=False)
                self.encoder = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(num_layers-1)])
                self.decoder_self = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
                block = TransformerBlock(d_model, nhead, nhid, dropout,decoder=True,hopfield=True,hop_dim=hop_dim,fno_layers=fno_layers,modes=modes,width=width,causal=causal,nystrom=nystrom,pkm_dims=pkm_dims,local_heads=local_heads,attend_to_self=attend_to_self,mlp_layers=mlp_layers)
                self.decoder_cross = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(num_layers-1)])
        else:
            self.encoder = nn.ModuleList([Identity()])
            self.decoder_self = nn.ModuleList([Identity()])
            self.decoder_cross = nn.ModuleList([Identity()])
            
        self.absolutepositionalembedding = AbsolutePositionalEmbedding(d_model,max_len) if deberta_layers else None
        block = TransformerBlock(d_model, nhead, nhid, dropout,decoder=True,encoder_n_decoder=encoder_n_decoder,hopfield=True,fno_layers=fno_layers,modes=modes,width=width,causal=causal,pkm_dims=pkm_dims,nystrom=nystrom,hop_dim=hop_dim,local_heads=local_heads,attend_to_self=attend_to_self,mlp_layers=mlp_layers,context=False) if deberta_layers else None
        self.deberta_layers = nn.ModuleList([block]+[copy.deepcopy(block) for _ in range(deberta_layers-1)]) if deberta_layers else None
        self.deb_attn_0 = Attention(d_model,
                                            heads=nhead,
                                            dim_head=d_model//nhead,
                                            num_mem_kv=0,
                                            rotary_pos_emb=True,
                                            nystrom=nystrom,) if deberta_layers else None
        
        self.attend_to_inp = Attention(d_model,
                                            heads=nhead,
                                            dim_head=d_model//nhead,
                                            num_mem_kv=0,
                                            rotary_pos_emb=True,
                                            nystrom=nystrom,) if attend_to_inp else None

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
                                                nystrom=nystrom,)

            self.prev_state_attend = Attention(d_model,
                                                heads=nhead,
                                                dim_head=d_model//nhead,
                                                num_mem_kv=0,
                                                rotary_pos_emb=True,
                                                nystrom=nystrom,)

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

        src_mask = torch.triu(torch.ones((src.size(2),src.size(2))),diagonal=1)

        if self.prev_state_exists:
            prev_state = repeat(self.prev_state,'1 n d -> b n d',b=output.size(0))
            prev_state = ckpt(self.prev_state_update,prev_state,output/output.size(-1))
            if ctxt != None:
                prev_state = ckpt(self.prev_state_update,prev_state,ctxt/ctxt.size(-1))

            for _ in range(self.prev_state_self_num):
                prev_state = ckpt(self.prev_state_update,prev_state)
            
        if self.enable_encoder:

            for enc in self.encoder:
                ctxt = ckpt(enc,ctxt)
            for i in range(self.num_layers):
                output = ckpt(self.decoder_self[i],output,None,src_mask)
                output = ckpt(self.decoder_cross[i],output,ctxt,src_mask)
        else:
            for dec in self.decoder_self:
                output = ckpt(dec,output)

        if self.prev_state_exists:
            output = ckpt(self.prev_state_attend,output,prev_state)

        if self.deberta_layers!=None:
            out = Positional_Encoding(self.absolutepositionalembedding(output))
            out = ckpt(self.deb_attn_0,out,output)
            if self.full_block_repeat:
                for _ in range(self.repeated_deberta_layers+1):
                    for enc in self.deberta_layers:
                        out = ckpt(enc,out,output,src_mask)
                else:
                    if self.deberta_layers!=None:
                        output = out
            else:
                for enc in self.deberta_layers:
                    for _ in range(self.repeated_deberta_layers+1):
                        out = ckpt(enc,out,output,src_mask)
                else:
                    if self.deberta_layers!=None:
                        output = out

        if self.attend_to_inp != None:
            output = ckpt(self.attend_to_inp,output,src)
            if context != None:
                output = ckpt(self.attend_to_inp,output,context)

        if self.prev_state_exists:
            prev_state = ckpt(self.prev_state_update,prev_state,output)
            if ctxt != None:
                prev_state = ckpt(self.prev_state_update,prev_state,ctxt)

            self.prev_state = torch.sum(prev_state,dim=0,keepdim=True).reshape(self.prev_state.shape) / output.size(0)

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
                    activation: str = 'relu',
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
                    nystrom: bool = True,
                    local_heads: int = 1,
                    attend_to_self=True,
                    fno_layers=4,
                    modes=None,
                    width=None,
                    attend_to_inp=True,
                    mlp_layers=1,
                    encoder_n_decoder=True,
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
                                                        nystrom=nystrom,
                                                        local_heads=local_heads,
                                                        attend_to_self=attend_to_self,
                                                        fno_layers=fno_layers,
                                                        modes=modes,
                                                        width=width,
                                                        prev_state_self_num=prev_state_self_num,
                                                        attend_to_inp=attend_to_inp,
                                                        mlp_layers=mlp_layers,
                                                        encoder_n_decoder=encoder_n_decoder,
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

        self.ffd1 = ET_ffd(dim=ninp)
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
        attn_len = ((self.seq_scale_down // 2)*2 + 1)*3

        self.scale_down_conv = nn.Sequential(
            nn.Conv1d(ninp,ninp,attn_len,padding=attn_len//2,groups=1),
            nn.Conv1d(ninp,ninp,self.seq_scale_down*3,self.seq_scale_down,padding=self.seq_scale_down*3 - 1,groups=1),
            nn.Conv1d(ninp,ninp,3,padding=1),
        )

        modes = nhead if modes == None else modes
        width = nhead if width == None else width

        self.scale_up_conv = nn.Sequential(
            nn.Conv1d(ninp,ninp,3,padding=1),
            nn.ConvTranspose1d(ninp,ninp,self.seq_scale_down*3,self.seq_scale_down,groups=1),
            nn.Conv1d(ninp,ninp,attn_len,padding=attn_len//2,groups=1),
        )

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
                    grad_clip=0.125,
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
            #loss = loss * min(2,max(1,torch.logical_and(torch.argmax(trainable_output,dim=-1) != trainable_output_targets, torch.argmax(trainable_output,dim=-1) == torch.full(trainable_output_targets.shape,1192,device=trainable_output.device,dtype=trainable_output.dtype)).sum().item()*(10/trainable_output_targets.view(-1).size(-1))))
            
            loss.backward()
            torch.cuda.empty_cache()
            if mini_batch_size != None and batch != None:
                if batch%mini_batch_size == 0:
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

        src = self.embedding_encoder(src)
        src = src * math.sqrt(self.ninp)
        src = ckpt(self.ffd1,src)

        if self.encoder_decoder:
            context = self.embedding_encoder(context)
            context = context * math.sqrt(self.ninp)
            context = ckpt(self.ffd1,context)

        if seq_scale_down:
            src = ckpt(self.scale_down_conv,src.transpose(-1,-2)).transpose(-1,-2)
            if self.encoder_decoder:
                context = ckpt(self.scale_down_conv,context.transpose(-1,-2)).transpose(-1,-2)
                
        mem = default(mem,self.mem)
        if exists(mem):
            if (mem.size(-1) == src.size(-1) and len(mem.shape) == 2): 
                src = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
        if self.encoder_decoder:
            context_mem = default(context_mem,self.mem)
            if exists(context_mem):
                if (context_mem.size(-1) == context.size(-1) and len(context_mem.shape) == 2): 
                    context = torch.cat((repeat(context_mem, 'n d -> b n d', b = context.size(0)),context),dim=-2)


        output = Positional_Encoding(src).contiguous()
        if self.encoder_decoder:
            context = Positional_Encoding(context).contiguous()

        output = ckpt(self.transformer_block,output,context)
        if isinstance(output,tuple):
            context = output[1]
            output = output[0]

        if exists(mem):
            mem = output[:,:mem.size(0)]
            mem = torch.sum(mem,dim=0,keepdim=True).reshape(self.mem.shape) / b
            mem = mem.detach()
            output = output[:,mem.size(0):]

        if exists(context):
            if exists(context_mem):
                context_mem = context[:,:context_mem.size(0)]
                context_mem = torch.sum(mem,dim=0,keepdim=True).reshape(self.context_mem.shape) / b
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
    