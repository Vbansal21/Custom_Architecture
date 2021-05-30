from json import decoder
import math
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
from typing import Tuple, Optional, Any, NoReturn, Union, Literal, List

from torch import Tensor
from torch.nn.modules.container import ModuleList, Module
from torch.nn.modules.dropout import Dropout

from einops import repeat,rearrange
from mogrifier import Mogrifier

from .evolved_transformer_block import ET_Encoder_Block,ET_Decoder_Block
from .product_key_memory import PKM
from .hopfield_modules import HopfieldLayer,HopfieldPooling
from .hopfield_modules.transformer import HopfieldEncoderLayer
from .performer_pytorch import SelfAttention

from .fourier_1d import FNO1d

from .dynamic_conv import Dynamic_conv1d
from .involution.involution_naive import involution

from .g_mlp_gpt import gMLPGPT

from torchnlp.encoders.text import SubwordEncoder
import torchnlp

from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
autocast = torch.cuda.amp.autocast

checkpointed = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def ckpt(f,arg1,arg2=None,arg3=None,arg4=None,checkpointed = checkpointed):
    if checkpointed:
        if arg2 == None and arg3 == None and arg4 == None:
            return checkpoint(f,arg1)
        elif arg3 == None and arg4 == None:
            return checkpoint(f,arg1,arg2)
        elif arg4 == None:
            return checkpoint(f,arg1,arg2,arg3)
        else:
            return checkpoint(f,arg1,arg2,arg3,arg4)
    else:
        if arg2 == None and arg3 == None and arg4 == None:
            return f(arg1)
        elif arg3 == None and arg4 == None:
            return f(arg1,arg2)
        elif arg4 == None:
            return f(arg1,arg2,arg3)
        else:
            return f(arg1,arg2,arg3,arg4)

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def Positional_Encoding(x: Tensor,
                            device: Optional[torch.DeviceObjType] = device
                        ) -> Tensor :
    max_len = x.size(1)
    d_model = x.size(2)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).to(device)
    return x + pe[:]

class convolutional_embedding(Module):

    def __init__(self,
        channels,
        kernal_size=1,
        stride_size=1
        ):
        super(convolutional_embedding,self).__init__()
        self.involution_layer = involution(channels,kernal_size,stride_size)
        self.dyn_conv_1d = Dynamic_conv1d(channels,channels,kernal_size,stride_size)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self,x):
        out = x.transpose(-1,-2).unsqueeze(-1).contiguous()
        out = ckpt(self.involution_layer,out)
        out = rearrange(out,'b d n o -> b (n o) d').contiguous()
        out2 = ckpt(self.dyn_conv_1d,x.transpose(-1,-2)).transpose(-1,-2)
        return self.norm(out+out2) + x

class RMSNorm(Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class GEGLU(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

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
    def __init__(self, dim, fn=None, mogrify = False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.gru = nBRC(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k = dim // 4) if mogrify else None

    def forward(self, x, y=None,*args):
        shape = x.shape
        dim = self.dim

        if y==None:
            y = x

        y = self.norm(ckpt(self.fn,y,*args)) + y if self.fn!=None else y

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = ckpt(self.gru,
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )

        return gated_output.reshape(shape)

class ET_ffd(Module):

    def __init__(self,dim,activation="gelu"):
        super(ET_ffd,self).__init__()
        self.l_1 = nn.Sequential(
            GEGLU(dim,dim*4),
            _get_activation_fn(activation),
        )

        self.l_2 = nn.Sequential(
            nn.Conv1d(dim*4,dim*4,1,1),
            _get_activation_fn(activation),
            nn.Conv1d(dim*4,dim,1,1),
            _get_activation_fn(activation),
        )

    def forward(self,x):
        out = ckpt(self.l_1,x)
        out = out.transpose(1,2)
        out = ckpt(self.l_2,out)
        return out.transpose(1,2).contiguous()

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

            for i in range(0,s,self.max_seq_len):
                tmp = torch.arange(i,self.max_seq_len+i, device = x.device)
                n.append(repeat(self.emb(tmp),'n d -> b n d',b=x.size(0)) * multiplier)
                multiplier *= (2**0.5)
            else:
                tmp = torch.arange(i+self.max_seq_len,s-1, device = x.device)
                n.append(repeat(self.emb(tmp),'n d -> b n d',b=x.size(0)) * multiplier)

            tmp = torch.cat(n,dim=1)
            assert tmp.size(1) == s
            return tmp


class TransformerBlock(Module):

    def __init__(self,
                     d_model,
                     nhead, 
                     dim_feedforward=2048, 
                     dropout=0.5, 
                     dropout_hopfield=0.0,
                     activation="gelu",
                     mem_kv=64*16,
                     pkm_dim=None,
                     pkm_keys=64,
                     decoder=False,
                     hopfield=False
                ):
        super(TransformerBlock, self).__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.dropout1 = Dropout(dropout)

        self.decoder = decoder

        self.ffd1 = ET_ffd(dim=d_model)

        if pkm_dim==None:
            pkm_dim = d_model//4

        pkm_heads = (nhead * pkm_dim) // d_model

        self.pkm1 = nn.Sequential(
            nn.Linear(d_model,pkm_dim),
            _get_activation_fn(activation),
            PKM(pkm_dim,heads=pkm_heads,num_keys=pkm_keys,dim_head=d_model//nhead),
            _get_activation_fn(activation),
            nn.Linear(pkm_dim,d_model),
            _get_activation_fn(activation),
            )

        self.gate = GRUGating(d_model)
        
        self.fno = GRUGating(d_model,FNO1d(nhead,
                                                    nhead,
                                                    inp_dim=d_model,
                                                    out_dim=d_model,
                                                    ffd_dim=dim_feedforward,
                                                    num_layers=4
                                                ))

        if hopfield:
            hop_attn = nn.Sequential(
                                    nn.Linear(d_model,d_model//4),
                                    GRUGating(d_model//4,HopfieldEncoderLayer(HopfieldLayer(
                                                input_size=d_model//4,
                                                num_heads=nhead,
                                                pattern_size=2**8,
                                                dropout=dropout_hopfield,
                                                quantity=2**8
                                            ),
                                            dim_feedforward=dim_feedforward//4) ),
                                    nn.Linear(d_model//4,d_model)
                                    )
        else:
            hop_attn = None

        if not decoder:
            
            attn_block = ET_Encoder_Block(d_model,
                                num_heads=nhead,
                                attn=SelfAttention(d_model,
                                                    heads=nhead,
                                                    dim_head=d_model//nhead,
                                                    num_mem_kv=mem_kv,
                                                    hop_attn=hop_attn
                                                ),
                                pkm=copy.deepcopy(self.pkm1),
                                )
        else:
            attn = {
                'self_1':SelfAttention(d_model,
                                        heads=nhead*2,
                                        dim_head=d_model//(nhead*2),
                                        num_mem_kv=mem_kv,
                                        hop_attn=hop_attn,
                                        rotary_pos_emb=True),
                'self_2':SelfAttention(d_model,
                                        heads=nhead,
                                        dim_head=d_model//nhead,
                                        num_mem_kv=mem_kv*4,
                                        rotary_pos_emb=True),
                'cross_1':SelfAttention(d_model,
                                        heads=nhead,
                                        dim_head=d_model//nhead,
                                        num_mem_kv=mem_kv,
                                        hop_attn=copy.deepcopy(hop_attn),
                                        rotary_pos_emb=False),
                'cross_2':SelfAttention(d_model,
                                        heads=nhead,
                                        dim_head=d_model//nhead,
                                        num_mem_kv=mem_kv*4,
                                        rotary_pos_emb=False)
            }

            attn_block = ET_Decoder_Block(d_model,
                                num_heads=nhead,
                                attn=attn,
                                pkm=copy.deepcopy(self.pkm1)
                                )
        
        self.attn = GRUGating(d_model,attn_block)

        self.mlp = GRUGating(d_model,gMLPGPT(dim=d_model,depth=1,seq_len=2**16,window=d_model*4))

        self.conv_emb = convolutional_embedding(d_model)

        self.decoder = decoder

        if decoder:
            self.ffd2 = copy.deepcopy(self.ffd1)

            self.pkm2 = copy.deepcopy(self.pkm1)
        
            self_attn_context = ET_Encoder_Block(d_model,
                                num_heads=nhead,
                                attn=SelfAttention(d_model,
                                                    heads=nhead,
                                                    dim_head=d_model//nhead,
                                                    num_mem_kv=mem_kv,
                                                    rotary_pos_emb=True
                                                ),
                                pkm=copy.deepcopy(self.pkm2)
                                )

            self.self_attn_context = GRUGating(d_model,self_attn_context)
        


    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:

        output = ckpt(self.norm1,src)
        output = Positional_Encoding(output)
        output2 = ckpt(self.ffd1,output) + ckpt(self.pkm1,output) + ckpt(self.conv_emb,output)
        output = ckpt(self.gate,output,self.dropout1(output2))

        if self.decoder:
            context = self.norm2(context)
            context = Positional_Encoding(context)
            context_ = ckpt(self.ffd2,context) + ckpt(self.pkm2,context)
            context_ = ckpt(self.self_attn_context,context,context_)

        output = ckpt(self.fno,output)

        output = ckpt(self.attn,output,output,context)
        output = self.dropout2(output)
        output = ckpt(self.mlp,output,output)
        return output

class TransformerModule(ModuleList):

    def __init__(self, nhead, nhid, num_layers, d_model,dropout=0.5,enable_encoder=False,deberta_layers=1,repeated_deberta_layers=2,max_len=2**17,prev_state_len=8192):
        super(TransformerModule, self).__init__()

        self.enable_encoder=enable_encoder
        self.repeated_deberta_layers = repeated_deberta_layers

        if not enable_encoder:
            block = TransformerBlock(d_model, nhead, nhid, dropout,hopfield=True)
            self.decoder = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        else:
            block = TransformerBlock(d_model, nhead, nhid, dropout)
            self.encoder = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
            self.decoder_self = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
            block = TransformerBlock(d_model, nhead, nhid, dropout,decoder=True,hopfield=True)
            self.decoder_cross = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        
        self.absolutepositionalembedding = AbsolutePositionalEmbedding(d_model,max_len)
        block = TransformerBlock(d_model, nhead, nhid, dropout,decoder=True)
        self.deberta_layers = nn.ModuleList([copy.deepcopy(block) for _ in range(deberta_layers)]) if deberta_layers else None
        
        self.norm = RMSNorm(d_model,eps=1e-16)

        self.scale_output = nn.Parameter(torch.zeros(d_model))
        self.scale_abs_pos_emb = nn.Parameter(torch.ones(d_model))

        del(block)

        self.prev_state_attend_0 = TransformerBlock(d_model, nhead, nhid, dropout,decoder=True,hopfield=True)
        self.prev_state_attend_1 = TransformerBlock(d_model, nhead, nhid, dropout,decoder=True)

        self.register_buffer(
            name='prev_state',
            tensor=torch.zeros((1,prev_state_len,d_model))
        )
        self.hop_pool = nn.Sequential(
                nn.Linear(d_model,d_model//4),
                HopfieldPooling(
                            input_size=d_model//4,
                        ),
                nn.Linear(d_model//4,d_model),
                nn.SiLU()
                        )
        self.prev_state_update = nn.Sequential(
            FNO1d(nhead,nhead,inp_dim=d_model,out_dim=d_model,ffd_dim=nhid,transpose_req=False,num_layers=1),
            nn.Conv1d(d_model,d_model,kernel_size=11,stride=1,padding=5,groups=d_model),
            nn.Conv1d(d_model,d_model,kernel_size=2,stride=1)
            )

        d_model = d_model
        self.num_layers = num_layers
        
    def pretrained_layer_multiplier(self,num=1,deb_num=1):
        self.num_layers *= num
        if self.enable_encoder:
            self.encoder = nn.ModuleList([copy.deepcopy(i) for i in self.encoder] * num)
            self.decoder_self = nn.ModuleList([copy.deepcopy(i) for i in self.decoder_self] * num)
            self.decoder_cross = nn.ModuleList([copy.deepcopy(i) for i in self.decoder_cross] * num)
            self.deberta_layers = nn.ModuleList([copy.deepcopy(i) for i in self.deberta_layers] * deb_num)
        else:
            self.decoder = nn.ModuleList([copy.deepcopy(i) for i in self.decoder] * num)
            self.deberta_layers = nn.ModuleList([copy.deepcopy(i) for i in self.deberta_layers] * deb_num)
            
    def convert_decoder_only_to_encoder_decoder(self):
        self.enable_encoder = True
        self.encoder = copy.deepcopy(self.decoder)
        self.decoder_cross = copy.deepcopy(self.decoder)
        self.decoder_self = self.decoder


    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:
        output = src
        ctxt = context

        if self.enable_encoder:
            for enc in self.encoder:
                ctxt = ckpt(enc,ctxt)
            for i in range(self.num_layers):
                output = ckpt(self.decoder_self[i],output)
                output = ckpt(self.decoder_cross[i],output,ctxt)
        else:
            for dec in self.decoder:
                output = ckpt(dec,output)

        output = ckpt(self.prev_state_attend_0,output,repeat(self.prev_state,'1 n d -> b n d',b=output.size(0)))

        out = self.absolutepositionalembedding(output) if len(self.deberta_layers)!=0 else output
        out = (out*self.scale_abs_pos_emb) + (self.norm(output * self.scale_output)).clamp(min=-0.125,max=0.125)

        for _ in range(self.repeated_deberta_layers+1):
            for enc in self.deberta_layers:
                out = ckpt(enc,out,output)
        else:
            if self.deberta_layers!=None:
                output = out

        output = ckpt(self.prev_state_attend_1,output,repeat(self.prev_state,'1 n d -> b n d',b=output.size(0)))

        tmp = ckpt(self.hop_pool,output)
        tmp = torch.sum(tmp,dim=0,keepdim=True).view(1,1,-1)
        self.prev_state = torch.cat((self.prev_state,tmp),dim=1)
        self.prev_state = ckpt(self.prev_state_update, self.prev_state.transpose(-1,-2)).transpose(-1,-2)

        return output

class TransformerModel(Module):

    @profile
    def __init__(self, 
                    ntoken: int, 
                    ninp: int, 
                    nhead: int, 
                    nhid: int, 
                    nlayers: int,
                    padding_idx: int = 0,
                    dropout: float = 0.5,
                    activation: Literal['gelu','relu'] = 'gelu',
                    mem_token: int = 00,
                    context_mem_token: int = 00,
                    encoder_decoder: bool = False,
                    deberta_layers: int = 1,
                    repeated_deberta_layers: int = 2,
                    max_seq_len=2**17,
                    discriminator: bool = False,
                    seq_scale_down: int = 8,
                    device: torch.DeviceObjType = device
                ) -> NoReturn :
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder_decoder = encoder_decoder
        self.transformer_encoder = TransformerModule(nhead, nhid, nlayers, ninp,dropout,enable_encoder=encoder_decoder,deberta_layers=deberta_layers,repeated_deberta_layers=repeated_deberta_layers,max_len=max_seq_len)
        
        self.embedding_encoder = nn.Embedding(ntoken, ninp,padding_idx=padding_idx)

        
        self.ninp = ninp
        self.ntokens = ntoken
        
        self.decoder = nn.Linear(ninp,ntoken)

        self.ffd1 = nn.Sequential(
            nn.Linear(ninp,nhid),
            _get_activation_fn(activation),
            FNO1d(nhead,nhead,nhid,nhid,nhid,num_layers=1),
            _get_activation_fn(activation),
            nn.Linear(nhid,ninp),
            _get_activation_fn(activation),
            nn.Linear(ninp,nhid),
            _get_activation_fn(activation),
            FNO1d(nhead,nhead,nhid,nhid,nhid,num_layers=1),
            _get_activation_fn(activation),
            nn.Linear(nhid,ninp),
            _get_activation_fn(activation)
        )
        self.ffd2 = copy.deepcopy(self.ffd1)

        self.norm1 = RMSNorm(ninp)
        self.norm2 = RMSNorm(ninp)

        self.mem_exist = True if mem_token else False
        if self.mem_exist:
            if type(mem_token)==int:
                self.mem = nn.Parameter(torch.randn(mem_token,ninp))
            elif type(mem_token) == Tensor:
                assert mem_token.size(-1)==ninp
                self.mem = nn.Parameter(mem_token)

        
        if encoder_decoder:
            self.ffd3 = copy.deepcopy(self.ffd1)
            self.norm2 = RMSNorm(ninp)

            self.context_mem_exist = True if context_mem_token else False
            if self.mem_exist:
                if type(context_mem_token)==int:
                    self.context_mem = nn.Parameter(torch.randn(context_mem_token,ninp))
                elif type(context_mem_token) == Tensor:
                    assert context_mem_token.size(-1)==ninp
                    self.context_mem = nn.Parameter(context_mem_token)

        self.discriminator_enabled = discriminator
        if discriminator:
            self.discriminator = nn.Sequential(
                nn.Linear(ninp,nhid),
                _get_activation_fn(activation),
                nn.Linear(nhid,ninp),
                _get_activation_fn(activation),
                nn.Linear(ninp,nhid),
                _get_activation_fn(activation),
                nn.Linear(nhid,ninp),
                _get_activation_fn(activation),
                TransformerModule(nhead, nhid, nlayers, ninp,dropout,enable_encoder=encoder_decoder,deberta_layers=deberta_layers,repeated_deberta_layers=repeated_deberta_layers,max_len=max_seq_len),
                _get_activation_fn(activation),
                nn.Linear(ninp,nhid),
                _get_activation_fn(activation),
                nn.Linear(nhid,ninp),
                _get_activation_fn(activation),
                nn.Linear(ninp,nhid),
                _get_activation_fn(activation),
                nn.Linear(nhid,ninp),
                _get_activation_fn(activation),
                nn.Linear(ninp,2)
            )
        
        self.alt_mem = None
        self.alt_mem_with_primary_mem = False

        self.seq_scale_down = seq_scale_down

        self.scale_down_conv = nn.Sequential(
            nn.Conv1d(ninp,ninp,self.seq_scale_down*3,self.seq_scale_down),
        )
        self.scale_down_fno = GRUGating(ninp,FNO1d(nhead,
                                                    nhead,
                                                    inp_dim=ninp,
                                                    out_dim=ninp,
                                                    ffd_dim=nhid,
                                                    num_layers=8
                                                ))

        self.padding_for_conv_scale = nn.Parameter(torch.randn((ninp,(self.seq_scale_down*3)-1)))

        self.scale_up_conv = nn.Sequential(
            nn.ConvTranspose1d(ninp,ninp,self.seq_scale_down*3,self.seq_scale_down),
        )

        self.scale_up_fno = GRUGating(ninp,FNO1d(nhead,
                                                    nhead,
                                                    inp_dim=ninp,
                                                    out_dim=ninp,
                                                    ffd_dim=nhid,
                                                    num_layers=8
                                                ))

        #self.to(device)
        self.device = device
        self.init_weights()

    def init_weights(self) -> NoReturn :
        for w in self.parameters():
            w.data.uniform_(-1/16,1/16)
            
    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())
            
    def multiply_pretrained_transformer_layers(self,num: int = 1,deb_num: int = 1) -> NoReturn :
        self.transformer_encoder.pretrained_layer_multiplier(num,deb_num)

    def convert_decoder_only_to_encoder_decoder(self) -> NoReturn:
        self.transformer_encoder.convert_decoder_only_to_encoder_decoder()
        if self.discriminator_enabled:
            self.discriminator.convert_decoder_only_to_encoder_decoder()

    def alt_mem_tokens(self,mem: Tensor,alt_mem_with_primary_mem: Optional[bool] = True) -> NoReturn :
        self.alt_mem = mem
        self.alt_mem_with_primary_mem = alt_mem_with_primary_mem

    def init_tokenizer(self,
                        sample:str = "",
                        append_eos: Optional[bool] = False,
                        target_vocab_size: Optional[int] = 2**16,
                        min_occ: Optional[int] = 1,
                        max_occ: Optional[int] = 100,
                        reserved_tokens: Optional[list] = [
                                                            '<pad>',
                                                            '<unk>',
                                                            '<s>', '</s>',
                                                            '<copy>',
                                                            '<mask>',
                                                            '<segment_seperator>',
                                                            '<non_text_content>', '</non_text_content>'
                                                           ],
                        eos_index: Optional[int] = 3,
                        unk_index: Optional[int] = 1,
                        pad_idx: Optional[int] = 0,
                        return_tokenizer: Optional[bool] = False
                        ) -> Union[NoReturn,torchnlp.encoders.text.text_encoder.TextEncoder] :
        self.tokenizer = SubwordEncoder(sample,
                                        append_eos=append_eos,
                                        target_vocab_size=target_vocab_size,
                                        min_occurrences=min_occ,
                                        max_occurrences=max_occ,
                                        reserved_tokens=reserved_tokens,
                                        eos_index=eos_index,
                                        unknown_index=unk_index,
                                        padding_index=pad_idx)
        self.vocab = self.tokenizer.vocab_size
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
        for i in range(inp.size(0)):
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
                    inp_2[i,j],inp_2[i,r] = inp[i,r],inp[i,j]
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
                    inp_2[i,j] = 5
                    count += 1
                    together_count += 1
                elif together_count>=mask_together_nos:
                    together_count = 0
        return inp_2.clone().detach()

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
        for txt in args:
            if type(txt) == str:
                tmp = self.tokenizer.encode(txt)
            else:
                assert type(tmp) == Tensor
            if mask_at_random or shuffle_at_random:
                tmp =   self.random_mask_shuffle_encoder(tmp,
                                                            mask=mask_at_random,
                                                            mask_percentage=mask_percentage,
                                                            mask_together_nos=mask_together_nos,
                                                            mask_continuous_pos=mask_continuous_pos,
                                                            shuffle=shuffle_at_random,
                                                            shuffle_percentage=shuffle_percentage,
                                                            shuffle_together_nos=shuffle_together_nos,
                                                            shuffle_continuous_pos=shuffle_continuous_pos
                                                        )

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
        return encoded_text

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
                            opt: Optional[Literal['Torch_optimizer']] = None,
                            opt_disc: Optional[Literal['Torch_optimizer']] = None,
                            lr: Union[None,float,dict] = None,
                            lr_disc: Union[None,float,dict] = None,
                            scheduler: Optional[Literal['Torch_LRscheduler']] = None,
                            lambdaLR: Optional[Literal['lambda_function']] = None,
                            scheduler_disc: Optional[Literal['Torch_LRscheduler']] = None,
                            lambdaLR_disc: Optional[Literal['lambda_function']] = None,
                            return_opt_schd: bool = False
                        ) -> Union[NoReturn,Tuple[Literal['Torch_optimizer'],Literal['Torch_LRscheduler']]]:
        if opt != None:
            self.optimizer = opt
            if self.discriminator_enabled:
                self.optimizer_disc = opt_disc
        else:
            assert lr!=None
            if not self.discriminator_enabled:
                self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            else:
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
                for p in self.parameters():
                    p.requires_grad_(False)
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                if lr_disc == None:
                    lr_disc = lr
                self.optimizer_disc = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr_disc)
                for p in self.parameters():
                    p.requires_grad_(True)

        if scheduler != None:
            self.scheduler = scheduler
            if self.discriminator_enabled:
                self.scheduler_disc = scheduler_disc
        else:
            if lambdaLR == None or lambdaLR_disc == None:
                a = 5000000
                b = 500
                c = 0.0
                if lambdaLR==None:
                    lambdaLR = lambda step: (((a/b * step + 1) / (step**2 + a)) + c)/(step**0.1+1)
                if lambdaLR_disc==None and self.discriminator_enabled:
                    lambdaLR_disc = lambda step: (((a/b * step + 1) / (step**2 + a)) + c)/(step**0.1+1)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,lr_lambda=lambdaLR)
            if self.discriminator_enabled:
                self.scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_disc,lr_lambda=lambdaLR_disc)
        if return_opt_schd:
            if self.discriminator_enabled:
                return self.optimizer, self.optimizer_disc, self.scheduler, self.scheduler_disc
            else:
                return self.optimizer,self.scheduler

    def training_step(self,
                    data,
                    targets,
                    loss_criterion,
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
                ):

        self.train()
        step_start_time = time.time()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

        if opt == None:
            optimizer = self.optimizer
        else:
            optimizer = opt

        if self.discriminator_enabled:
            if opt_disc == None:
                optimizer_disc = self.optimizer_disc
            else:
                optimizer_disc = opt_disc
        else:
            optimizer_disc = None


        def step_optimizer(input_data=data,output_targets=targets):
            outputs = {}
            losses = {}
            labels = {}
            if self.discriminator_enabled:
                self.zero_grad()
                    
                real_label = torch.full(output_targets.size(),0,dtype=torch.long,device=device)
                real_label_gen = torch.full(input_data.size(),0,dtype=torch.long,device=device)
                fake_label = torch.full(input_data.size(),1,dtype=torch.long,device=device)

                labels['real_label'] = real_label
                labels['real_label_gen'] = real_label_gen
                labels['fake_label'] = fake_label

                out_d_real = self.forward(output_targets.detach(),return_mem=False,discriminator=True,generator=False)
                loss_d_real = loss_criterion(rearrange(out_d_real,'b n c -> n c b'), rearrange(real_label,'b n -> n b'))
                loss_d_real.backward()

                out_gan = self.forward(input_data.detach(),mem=mem_tokens,return_mem=False,discriminator=True)
                loss_d_fake = loss_criterion(rearrange(out_gan,'b n c -> n c b'), rearrange(fake_label,'b n -> n b'))
                loss_d_fake.backward(retain_graph=True)

                optimizer_disc.step()
                optimizer_disc.zero_grad()
                self.zero_grad()

                loss_gen = loss_criterion(rearrange(out_gan,'b n c -> n c b'), rearrange(real_label_gen,'b n -> n b'))
                loss_gen.backward()

                output,single_pass_mem = self.forward(input_data,mem=mem_tokens)
                loss = loss_criterion(rearrange(output,'b n c -> n c b'), rearrange(output_targets,'b n -> n b'))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                self.zero_grad()

                outputs['out_d_real'] = out_d_real
                outputs['out_gan'] = out_gan
                outputs['output'] = output

                losses['loss_d_real'] = loss_d_real.item()
                losses['loss_d_fake'] = loss_d_fake.item()
                losses['loss_gen'] = loss_gen.item()
                losses['loss'] = loss.item()
            else:
                self.zero_grad()
                output,single_pass_mem = self.forward(input_data,mem=mem_tokens)
                torch.cuda.empty_cache()
                loss = loss_criterion(output.permute(1,2,0).contiguous(), targets.permute(1,0).contiguous())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                outputs['output'] = output

                losses['loss'] = loss.item()
            return outputs,losses,labels,single_pass_mem
        
        if deepspeed_enabled or autocast_enabled:
            with autocast():
                outputs,losses,labels,mem_ = step_optimizer(data,targets)
        else:
            outputs,losses,labels,mem_ = step_optimizer(data,targets)
        
        acc_gen = 0.0
        loss_g = 0.0
        loss_d = 0.0

        if self.discriminator_enabled:
            acc = ((torch.argmax(outputs['output'],dim=-1)) == targets).sum().item()/outputs['output'].size(1)
            acc_d = ((torch.argmax(outputs['out_d_real'],dim=-1)) == labels['real_label']).sum().item()/outputs['out_d_real'].size(1)
            acc_d += ((torch.argmax(outputs['out_gan'],dim=-1)) == labels['fake_label']).sum().item()/outputs['out_gan'].size(1)
            acc_gen = ((torch.argmax(outputs['out_gan'],dim=-1)) == labels['real_label_gen']).sum().item()/outputs['out_gan'].size(1)
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

        return outputs,losses,total_acc,total_acc_d,total_loss,total_loss_d,loss_g,loss_d,acc_gen,(step_start_time-time.time()),optimizer,optimizer_disc,mem_


    #@autocast()
    def forward(self,
                    src:Tensor,
                    context: Optional[Tensor] = None,
                    mem: Optional[Tensor] = None, 
                    context_mem: Optional[Tensor] = None,
                    alt_mem_with_primary_key: Optional[bool] = None,
                    assign_to_alt_mem: bool = False,
                    return_mem: bool = True,
                    generator: bool = True,
                    discriminator: bool = False,
                ) -> Tuple[Tensor,Optional[Tensor],Optional[Tensor]]:
        
        b,s_ = src.size(0),src.size(1)

        #src.requires_grad_(True)
        src = self.embedding_encoder(src)
        src = src * math.sqrt(self.ninp)

        if generator or not self.discriminator_enabled:

            src = ckpt(self.scale_down_fno,src).transpose(-1,-2)

            if src.size(2)%self.seq_scale_down != 0:
                src = torch.cat((src,repeat(self.padding_for_conv_scale,'d n -> b d n',b=src.size(0)).to(self.device)),dim=2)

            src = ckpt(self.scale_down_conv,src).transpose(-1,-2)
            s = src.size(1)

            if self.encoder_decoder:
                context = ckpt(self.embedding_encoder,context)
                context = context * math.sqrt(self.ninp)
                context = ckpt(self.scale_down_fno,context).transpose(-1,-2)
                if context.size(2)%self.seq_scale_down!=0:
                    context = torch.cat((context,repeat(self.padding_for_conv_scale,'d n -> b d n',b=context.size(0)).to(self.device)),dim=2)

                context = ckpt(self.scale_down_conv,context).transpose(-1,-2)
                


            self.alt_mem_with_primary_mem = alt_mem_with_primary_key if type(alt_mem_with_primary_key) == bool else self.alt_mem_with_primary_mem

            if mem != None and self.mem_exist:
                if self.alt_mem_with_primary_mem and self.alt_mem != None:
                    output = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
                else:
                    output = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            elif self.mem_exist:
                if self.alt_mem_with_primary_mem and self.alt_mem != None:
                    output = torch.cat((repeat(self.mem, 'n d -> b n d', b = src.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
                elif not self.alt_mem_with_primary_mem and self.alt_mem != None:
                    output = torch.cat((repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
                else:
                    output = torch.cat((repeat(self.mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            else:
                output = src
            

            if self.encoder_decoder:
                if context_mem != None and self.context_mem_exist:
                    if self.alt_mem_with_primary_mem and self.alt_mem != None:
                        context = torch.cat((repeat(context_mem, 'n d -> b n d', b = context.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = context.size(0)),context),dim=-2)
                    else:
                        context = torch.cat((repeat(context_mem, 'n d -> b n d', b = context.size(0)),context),dim=-2)
                elif self.context_mem_exist:
                    if self.alt_mem_with_primary_mem and self.alt_mem != None:
                        context = torch.cat((repeat(self.context_mem, 'n d -> b n d', b = context.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = context.size(0)),context),dim=-2)
                    elif not self.alt_mem_with_primary_mem and self.alt_mem != None:
                        context = torch.cat((repeat(self.alt_mem, 'n d -> b n d', b = context.size(0)),context),dim=-2)
                    else:
                        context = torch.cat((repeat(self.context_mem, 'n d -> b n d', b = context.size(0)),context),dim=-2)


            #output = Positional_Encoding(output,device=self.device)
            if self.encoder_decoder:
                #context = Positional_Encoding(context,device=self.device)
                context = context.contiguous()

            output = output.contiguous()

            output = ckpt(self.ffd1,output)
            output = ckpt(self.norm1,output) + output

            if self.encoder_decoder:
                context = ckpt(self.scale_down_fno,context)
                context2 = ckpt(self.ffd3,context)
                context = ckpt(self.norm1, context2) + context

            output = ckpt(self.transformer_encoder,output,context)

            output = ckpt(self.ffd2,output)
            output = ckpt(self.norm2,output) + output


            for i in range(b):
                if i == 0:
                    mem = output[i,:output.size(1)-s]
                else:
                    mem += output[i,:output.size(1)-s]
            mem = mem if type(mem) == torch.tensor else None
            self.alt_mem = mem if assign_to_alt_mem else None

            output = output[:,output.size(1)-s:] if type(mem) != None or self.mem_exist else output

            output = ckpt(self.scale_up_fno,output)
            output = ckpt(self.scale_up_conv,output.transpose(-1,-2)).transpose(-1,-2)
            out = output[:,:s_]

            output = ckpt(self.decoder,out)

        if discriminator and not generator:
            out = src

        if discriminator:
            output = ckpt(self.discriminator,out,context)

        if return_mem:
            return output, mem
        else:
            return output
