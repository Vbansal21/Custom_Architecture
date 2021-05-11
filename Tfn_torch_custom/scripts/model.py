import math
import torch
from torch import tensor
import torch.nn as nn
import random
import torch.nn.functional as F

import copy
from typing import Tuple, Optional, Any

from torch import Tensor
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout

from typing import  Optional
from einops import repeat,rearrange

from ET_torch.models.evolved_transformer_block import EvolvedTransformerBlock
from product_key_memory import PKM
from hopfield_modules import Hopfield, HopfieldLayer, HopfieldPooling
from performer_torch import SelfAttention

from torchnlp.encoders.text import SubwordEncoder
import torchnlp

from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
autocast = torch.cuda.amp.autocast

torch.set_num_threads(12)
torch.set_num_interop_threads(12)

checkpointed = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ckpt(f,*args,checkpointed = checkpointed):
    if checkpointed:
        return checkpoint(f,*args)
    else:
        return f(*args)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)

class mem_norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim*2)
        #self.proj = nn.Linear(dim,dim*2)
        self.proj = HopfieldLayer(dim,output_size=dim*2)
        self.g = nn.Parameter(torch.zeros(1))
        self.norm = RMSNorm(dim)

    def forward(self, x, residual):
        residual_proj = torch.tanh(self.proj(residual))
        gated_output, gate = torch.tanh(self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual_proj, 'b n d -> (b n) d')
        ).reshape_as(residual_proj)).chunk(2, dim = -1)
        x = (gated_output + x) * F.gelu(gate) * self.g
        return self.norm(x)+residual

class TransformerBlock(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",context=False,mem_kv=8192):
        super(TransformerBlock, self).__init__()
        #self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)        
        #attn = SelfAttention(d_model,heads=nhead,dim_head=d_model//nhead)
        #hopfield = Hopfield(input_size=d_model,num_heads=nhead)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = Dropout(dropout)

        self.ffd1 = nn.Sequential(
            nn.Linear(d_model,dim_feedforward),
            _get_activation_fn(activation),
            nn.Linear(dim_feedforward,d_model),
            _get_activation_fn(activation),
        )

        self.geglu1 = nn.Sequential(
            GEGLU(d_model,d_model),
        )

        self.pkm1 = nn.Sequential(
            nn.Linear(d_model,d_model),
            _get_activation_fn(activation),
            PKM(d_model),
            _get_activation_fn(activation),
            nn.Linear(d_model,d_model),
            _get_activation_fn(activation),
            )
        
        
        self.attn = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=SelfAttention(d_model,
                                                causal=False,
                                                heads=nhead,
                                                dim_head=d_model//nhead,
                                                num_mem_kv=mem_kv,
                                            ),
                            ffd=nn.Sequential(
                                                nn.Linear(d_model,dim_feedforward),
                                                _get_activation_fn(activation),
                                                nn.Linear(dim_feedforward,d_model),
                                                _get_activation_fn(activation),
                                                GEGLU(d_model,d_model),
                                            ),
                            context=context,
                            pkm=copy.deepcopy(self.pkm)
                            )
        self.self_hop_src = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=Hopfield(
                                input_size=d_model,
                                num_heads=nhead
                                ),
                            ffd=copy.deepcopy(self.ffd),
                            context=False,
                            pkm=copy.deepcopy(self.pkm)
                            )
        

        self.gate = mem_norm(d_model)

        self.context_exist = context
        if context:
            self.ffd2 = nn.Sequential(
                nn.Linear(d_model,dim_feedforward),
                _get_activation_fn(activation),
                nn.Linear(dim_feedforward,d_model),
                _get_activation_fn(activation),
            )

            self.geglu2 = nn.Sequential(
                GEGLU(d_model,d_model),
            )

            self.pkm2 = nn.Sequential(
                nn.Linear(d_model,d_model),
                _get_activation_fn(activation),
                PKM(d_model),
                _get_activation_fn(activation),
                nn.Linear(d_model,d_model),
                _get_activation_fn(activation),
                )
        
            self.self_hop_context = EvolvedTransformerBlock(d_model,
                                num_heads=nhead,
                                attn=Hopfield(
                                    input_size=d_model,
                                    num_heads=nhead
                                    ),
                                ffd=copy.deepcopy(self.ffd),
                                context=False,
                                pkm=copy.deepcopy(self.pkm)
                                )
            self.context_gate = mem_norm(d_model)
        


    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:

        output = self.norm1(src)
        output = ckpt(self.ffd1,output) + ckpt(self.pkm1,output) + ckpt(self.geglu1,output)
        output = ckpt(self.self_hop_src,output) + output

        if self.context_exist:
            context = self.norm2(context)
            context = ckpt(self.ffd2,context) + ckpt(self.pkm2,context) + ckpt(self.geglu2,context)
            context_ = ckpt(self.self_hop_context,context)
            context = ckpt(self.context_gate,context,context_)
            del(context_)
            context = Positional_Encoding(context)

        output = Positional_Encoding(output)
        output = ckpt(self.attn,output,context)
        output = self.dropout(output)
        src = ckpt(self.gate,output,src)
        del(output)
        return src

class TransformerModule(nn.Module):
    __constants__ = ['norm']

    def __init__(self, nhead, nhid, num_layers, d_model,dropout=0.5,enable_encoder=False):
        super(TransformerModule, self).__init__()

        self.enable_encoder=enable_encoder

        if not enable_encoder:
            self.decoder = _get_clones(TransformerBlock(d_model, nhead, nhid, dropout), num_layers)
        else:
            self.encoder = _get_clones(TransformerBlock(d_model, nhead, nhid, dropout), num_layers)
            self.decoder_self = _get_clones(TransformerBlock(d_model, nhead, nhid, dropout), num_layers)
            self.decoder_cross = _get_clones(TransformerBlock(d_model, nhead, nhid, dropout,context=True), num_layers)
        d_model = d_model
        self.num_layers = num_layers
        
    def pretrained_layer_multiplier(self,num=1):
        self.num_layers *= num
        if self.enable_encoder:
            self.encoder = nn.ModuleList([copy.deepcopy(i) for i in self.encoder] * num)
            self.decoder_self = nn.ModuleList([copy.deepcopy(i) for i in self.decoder_self] * num)
            self.decoder_cross = nn.ModuleList([copy.deepcopy(i) for i in self.decoder_cross] * num)
        else:
            self.decoder = nn.ModuleList([copy.deepcopy(i) for i in self.decoder] * num)


    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:
        output = src

        if self.enable_encoder:
            for enc in self.encoder:
                context = ckpt(enc,context)
            for i in range(self.num_layers):
                output = ckpt(self.decoder_self[i],output)
                output = ckpt(self.decoder_cross[i],output,context)
        else:
            for dec in self.decoder:
                output = ckpt(dec,output)

        return output

class TransformerModel(nn.Module):

    def __init__(self, 
                    ntoken: int, 
                    ninp: int, 
                    nhead: int, 
                    nhid: int, 
                    nlayers: int,
                    padding_idx: Optional[int] = 0,
                    dropout: Optional[float] = 0.5,
                    activation: Optional[str] = 'gelu',
                    mem_token: Optional[int] = 00,
                    context_mem_token: Optional[int] = 00,
                    encoder_decoder: Optional[bool] = False,
                    device: Optional[torch.DeviceObjType] = device
                ) -> None :
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder_decoder = encoder_decoder
        self.transformer_encoder = TransformerModule(nhead, nhid, nlayers, ninp,dropout,enable_encoder=encoder_decoder)
        
        embedding_encoder = nn.Sequential(
            nn.Embedding(ntoken, ninp,padding_idx=padding_idx),
            _get_activation_fn(activation),
            nn.Linear(ninp,nhid*2),
            _get_activation_fn(activation),
            nn.Linear(nhid*2,ninp),
            _get_activation_fn(activation),
        )
        
        self.ninp = ninp
        self.ntokens = ntoken
        
        self.decoder = nn.Sequential(
            nn.Linear(ninp,nhid*2),
            _get_activation_fn(activation),
            nn.Linear(nhid*2,ninp),
            _get_activation_fn(activation),
            nn.Linear(ninp,ntoken)
        )

        self.ffd1 = nn.Sequential(
            nn.Linear(ninp,nhid*2),
            _get_activation_fn(activation),
            nn.Linear(nhid*2,ninp),
            _get_activation_fn(activation)
        )
        self.ffd2 = copy.deepcopy(self.ffd1)

        self.norm1 = RMSNorm(ninp)
        self.norm2 = RMSNorm(ninp)

        self.embedding_encoder = embedding_encoder

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
        
        self.alt_mem = None
        self.alt_mem_with_primary_mem = False

        self.to(device)
        self.device = device
        self.init_weights()

    def init_weights(self) -> None :
        for w in self.parameters():
            w.data.uniform_(-1/16,1/16)
            
    def multiply_pretrained_transformer_layers(self,num: Optional[int] = 1) -> None :
        self.transformer_encoder.pretrained_layer_multiplier(num)

    def alt_mem_tokens(self,mem: Tensor,alt_mem_with_primary_mem: Optional[bool] = True) -> None :
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
                        ) -> None or torchnlp.encoders.text.text_encoder.TextEncoder :
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
                                mask: Optional[bool] = True,
                                mask_percentage: Optional[float] = 15,
                                mask_together_nos: Optional[int] = 3,
                                mask_continuous_pos: Optional[float] = -101,
                                shuffle: Optional[bool] = True,
                                shuffle_percentage: Optional[float] = 15,
                                shuffle_together_nos: Optional[int] = 3,
                                shuffle_continuous_pos: Optional[float] = -101
                            ) -> Tensor:
        inp_2 = inp.clone().detach()
        for i in range(inp.size(0)):
            count = 0
            together_count = 0
            for j in range(inp.size(1)):
                rnd = -1
                if shuffle_continuous_pos < -100 or shuffle_continuous_pos > 100:
                    rnd = random.randint(0,100000)/1000
                elif shuffle_continuous_pos >= -100 and shuffle_continuous_pos <= 100:
                    if shuffle_continuous_pos < 0:
                        if (inp.size(1)-j+1)/inp.size(1) >= ((-1)*shuffle_continuous_pos):
                            rnd = 0
                    else:
                        if (j+1)/inp.size(1) >= shuffle_continuous_pos:
                            rnd = 0
                if (((rnd>=0 and rnd<shuffle_percentage) or together_count<shuffle_together_nos) and shuffle and (((count+1)/inp.size(1))<=shuffle_percentage)):
                    while True:
                        r = random.randint(0,inp.size(1)-1)
                        if r!=j:
                            break
                    inp_2[i,j],inp_2[i,r] = inp[i,r],inp[i,j]
                    count += 1
                    together_count += 1
                elif together_count>=shuffle_together_nos:
                    together_count = 0

            count = 0
            together_count = 0
            for j in range(inp.size(1)):
                rnd = -1
                if mask_continuous_pos < -100 or mask_continuous_pos > 100:
                    rnd = random.randint(0,100000)/1000
                elif mask_continuous_pos >= -100 and mask_continuous_pos <= 100:
                    if mask_continuous_pos < 0:
                        if (inp.size(1)-j+1)/inp.size(1) >= ((-1)*mask_continuous_pos):
                            rnd = 0
                    else:
                        if (j+1)/inp.size(1) >= mask_continuous_pos:
                            rnd = 0
                if (((rnd>=0 and rnd<mask_percentage) or together_count<mask_together_nos) and mask and (((count+1)/inp.size(1))<=mask_percentage)):
                    inp_2[i,j] = 5
                    count += 1
                    together_count += 1
                elif together_count>=mask_together_nos:
                    together_count = 0
        return inp_2

    def encode_text(self,
                        *args: str,
                        append_pad_at_start: Optional[bool] = True,
                        pad_idx: Optional[int] = 0,
                        concatenate_all: Optional[bool] = False,
                        concatenate_dim: Optional[int] = 1,
                        append_segment_seperator: Optional[bool] = True,
                        segment_idx: Optional[int] = 6,
                        mask_at_random: Optional[bool] = True,
                        mask_percentage: Optional[float] = 15,
                        mask_together_nos: Optional[int] = 3,
                        mask_continuous_pos: Optional[float] = -101,
                        shuffle_at_random: Optional[bool] = True,
                        shuffle_percentage: Optional[float] = 15,
                        shuffle_together_nos: Optional[int] = 3,
                        shuffle_continuous_pos: Optional[float] = -101
                    ) -> list:
        encoded_text = []
        for txt in args:
            tmp = self.tokenizer.encode(txt)
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
            if append_pad_at_start:
                tmp = torch.cat((torch.tensor([[pad_idx]]),tmp),dim=1)
            if append_segment_seperator:
                tmp = torch.cat((tmp,torch.tensor([[segment_idx]])),dim=1)
            encoded_text.append(tmp)
        
        if concatenate_all:
            encoded_text = [torch.cat(encoded_text,dim=concatenate_dim)]
        return encoded_text

    def decode_text(self,
                        *args: Tensor,
                        to_text: Optional[bool] = True
                        ) -> list:
        decoded_text = []
        for txt in args:
            if type(txt) != Tensor:
                txt = torch.tensor(txt)

            if txt.size(-1) == self.ntokens and len(txt.size()) == 3:
                txt = torch.argmax(txt,dim=-1)

            if to_text:
                if txt.size(0) > 1:
                    txt = self.tokenizer.batch_decode(txt)
                else:
                    txt = self.tokenizer.decode(txt)
            decoded_text.append(txt)

    @autocast()
    def forward(self,
                    src:Tensor,
                    context: Optional[Tensor] = None,
                    mem: Optional[Tensor] = None, 
                    context_mem: Optional[Tensor] = None,
                    alt_mem_with_primary_key: Optional[bool] = None,
                    assign_to_alt_mem: Optional[bool] = True,
                    return_mem: Optional[bool] = True,
                ) -> Tuple(Tensor,Optional[Tensor]):
        
        b,s = src.size(0),src.size(1)

        src = ckpt(self.embedding_encoder,src)
        src = src * math.sqrt(self.ninp)

        if self.encoder_decoder:
            context = ckpt(self.embedding_encoder,context)
            context = context * math.sqrt(self.ninp)


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


        output = Positional_Encoding(output,device=self.device)
        if self.encoder_decoder:
            context = Positional_Encoding(context,device=self.device)

        output2 = ckpt(self.ffd1,output)
        output = ckpt(self.norm1,output2) + output
        del(output2)

        if self.encoder_decoder:
            context2 = ckpt(self.ffd3,context)
            context = ckpt(self.norm1, context2) + context
            del(context2)

        output = ckpt(self.transformer_encoder,output,context)

        output2 = ckpt(self.ffd2,output)
        output = ckpt(self.norm2,output2) + output
        del(output2)


        for i in range(b):
            if i == 0:
                mem = output[i,:output.size(1)-s]
            else:
                mem += output[i,:output.size(1)-s]
        mem = mem if type(mem) == torch.tensor else None
        self.alt_mem = mem if assign_to_alt_mem else None

        output = output[:,output.size(1)-s:] if type(mem) != None or self.mem_exist else output


        output = ckpt(self.decoder,output)
        
        if return_mem:
            return output,mem
        else:
            return output


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