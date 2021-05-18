import math
import torch
from torch import tensor
import torch.nn as nn
import random
import torch.nn.functional as F

import copy
from typing import Tuple, Optional, Any, NoReturn, Union, Literal

from torch import Tensor
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout

from einops import repeat,rearrange

from ET_torch.models.evolved_transformer_block import EvolvedTransformerBlock
from product_key_memory import PKM
from hopfield_modules import Hopfield, HopfieldLayer, HopfieldPooling
from performer_torch import SelfAttention

from x_transformers.x_transformers import AbsolutePositionalEmbedding

from fourier_neural_operator.fourier_1d import FNO1d

from torchnlp.encoders.text import SubwordEncoder
import torchnlp

from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
autocast = torch.cuda.amp.autocast

torch.set_num_threads(12)
torch.set_num_interop_threads(12)

checkpointed = True
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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
        self.gru = nn.GRUCell(dim, dim)
        #self.proj = nn.Linear(dim,dim*2)
        self.proj = HopfieldLayer(dim,output_size=dim*2)
        self.g = nn.Parameter(torch.zeros(1))
        self.norm = RMSNorm(dim)

    def forward(self, x, residual):
        gru_out = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        ).reshape_as(x)
        gated_output,gate = self.proj(gru_out).chunk(2,dim=-1)
        x = (gated_output * self.g + x) * F.gelu(gate)
        return self.norm(x)+residual

class TransformerBlock(nn.Module):

    def __init__(self,
                     d_model,
                     nhead, 
                     dim_feedforward=2048, 
                     dropout=0.5, 
                     activation="gelu",
                     context=False,
                     mem_kv=8192,
                     deberta_mode=False
                ):
        super(TransformerBlock, self).__init__()
        #self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)        
        #attn = SelfAttention(d_model,heads=nhead,dim_head=d_model//nhead)
        #hopfield = Hopfield(input_size=d_model,num_heads=nhead)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = Dropout(dropout)

        self.deberta_mode = deberta_mode

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
            nn.Linear(d_model,d_model//2),
            _get_activation_fn(activation),
            PKM(d_model//2),
            _get_activation_fn(activation),
            nn.Linear(d_model//2,d_model),
            _get_activation_fn(activation),
            )
        
        if not deberta_mode:
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
                                pkm=copy.deepcopy(self.pkm1)
                                )
        else:
            self.attn = EvolvedTransformerBlock(d_model,
                                num_heads=nhead,
                                attn=SelfAttention(d_model,
                                                    causal=False,
                                                    heads=nhead,
                                                    dim_head=d_model//nhead,
                                                    num_mem_kv=mem_kv,
                                                    to_q=EvolvedTransformerBlock(
                                                        d_model,
                                                        nhead,
                                                        attn=FNO1d(nhead,
                                                                    d_model,
                                                                    inp_dim=d_model,
                                                                    out_dim=d_model,
                                                                    ffd_dim=dim_feedforward
                                                                ),
                                                        ffd=nn.Sequential(
                                                                            nn.Linear(d_model,dim_feedforward),
                                                                            _get_activation_fn(activation),
                                                                            nn.Linear(dim_feedforward,d_model),
                                                                            _get_activation_fn(activation),
                                                                            GEGLU(d_model,d_model),
                                                                        ),
                                                        context=False,
                                                    ),
                                                    to_k=EvolvedTransformerBlock(
                                                        d_model,
                                                        nhead,
                                                        attn=FNO1d(nhead,
                                                                    d_model,
                                                                    inp_dim=d_model,
                                                                    out_dim=d_model,
                                                                    ffd_dim=dim_feedforward
                                                                ),
                                                        ffd=nn.Sequential(
                                                                            nn.Linear(d_model,dim_feedforward),
                                                                            _get_activation_fn(activation),
                                                                            nn.Linear(dim_feedforward,d_model),
                                                                            _get_activation_fn(activation),
                                                                            GEGLU(d_model,d_model),
                                                                        ),
                                                        context=False,
                                                    ),
                                                    to_out=EvolvedTransformerBlock(
                                                        d_model,
                                                        nhead,
                                                        attn=FNO1d(nhead,
                                                                    d_model,
                                                                    inp_dim=d_model,
                                                                    out_dim=d_model,
                                                                    ffd_dim=dim_feedforward
                                                                ),
                                                        ffd=nn.Sequential(
                                                                            nn.Linear(d_model,dim_feedforward),
                                                                            _get_activation_fn(activation),
                                                                            nn.Linear(dim_feedforward,d_model),
                                                                            _get_activation_fn(activation),
                                                                            GEGLU(d_model,d_model),
                                                                        ),
                                                        context=False,
                                                        pkm=copy.deepcopy(self.pkm1)
                                                    )
                                                ),
                                ffd=nn.Sequential(
                                                    nn.Linear(d_model,dim_feedforward),
                                                    _get_activation_fn(activation),
                                                    nn.Linear(dim_feedforward,d_model),
                                                    _get_activation_fn(activation),
                                                    GEGLU(d_model,d_model),
                                                ),
                                context=context,
                                pkm=copy.deepcopy(self.pkm1)
                                )
        self.self_hop_src = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=nn.Sequential(
                                                nn.Linear(d_model,d_model//2),
                                                HopfieldLayer(
                                                            input_size=d_model//2,
                                                            num_heads=nhead,
                                                            dropout=dropout
                                                        ),
                                                nn.Linear(d_model//2,d_model)
                                              ),
                            ffd=copy.deepcopy(self.ffd1),
                            context=False,
                            pkm=copy.deepcopy(self.pkm1)
                            )
        

        self.gate1 = mem_norm(d_model)
        self.gate2 = mem_norm(d_model)

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
                                ffd=copy.deepcopy(self.ffd2),
                                context=False,
                                pkm=copy.deepcopy(self.pkm2)
                                )
            self.context_gate = mem_norm(d_model)
        


    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:

        output = self.norm1(src)
        output = ckpt(self.ffd1,output) + ckpt(self.pkm1,output) + ckpt(self.geglu1,output)
        output2 = ckpt(self.self_hop_src,output)
        output = ckpt(self.gate1,output2,src)
        del(output2)

        if self.context_exist:
            context = self.norm2(context)
            context = ckpt(self.ffd2,context) + ckpt(self.pkm2,context) + ckpt(self.geglu2,context)
            context_ = ckpt(self.self_hop_context,context)
            context = ckpt(self.context_gate,context,context_)
            del(context_)
            context = Positional_Encoding(context) if not self.deberta_mode else context

        output = Positional_Encoding(output) if not self.deberta_mode else output
        output = ckpt(self.attn,output,context)
        output = self.dropout(output)
        src = ckpt(self.gate2,output,src)
        del(output)
        return src

class TransformerModule(nn.Module):

    def __init__(self, nhead, nhid, num_layers, d_model,dropout=0.5,enable_encoder=False,deberta_layers=1,repeated_deberta_layers=2):
        super(TransformerModule, self).__init__()

        self.enable_encoder=enable_encoder
        self.repeated_deberta_layers = repeated_deberta_layers

        if not enable_encoder:
            self.decoder = nn.ModuleList([TransformerBlock(d_model, nhead, nhid, dropout) for _ in range(num_layers)])
        else:
            self.encoder = nn.ModuleList([TransformerBlock(d_model, nhead, nhid, dropout) for _ in range(num_layers)])
            self.decoder_self = nn.ModuleList([TransformerBlock(d_model, nhead, nhid, dropout) for _ in range(num_layers)])
            self.decoder_cross = nn.ModuleList([TransformerBlock(d_model, nhead, nhid, dropout,context=True) for _ in range(num_layers)])
        
        self.absolutepositionalembedding = AbsolutePositionalEmbedding(d_model,2*17)
        self.deberta_layers = nn.ModuleList([TransformerBlock(d_model, nhead, nhid, dropout,context=True,deberta_mode=True) for _ in range(deberta_layers)])

        d_model = d_model
        self.num_layers = num_layers
        
    def pretrained_layer_multiplier(self,num=1):
        self.num_layers *= num
        if self.enable_encoder:
            self.encoder = nn.ModuleList([copy.deepcopy(i) for i in self.encoder] * num)
            self.decoder_self = nn.ModuleList([copy.deepcopy(i) for i in self.decoder_self] * num)
            self.decoder_cross = nn.ModuleList([copy.deepcopy(i) for i in self.decoder_cross] * num)
            self.deberta_layers = nn.ModuleList([copy.deepcopy(i) for i in self.deberta_layers] * num)
        else:
            self.decoder = nn.ModuleList([copy.deepcopy(i) for i in self.decoder] * num)
            self.deberta_layers = nn.ModuleList([copy.deepcopy(i) for i in self.deberta_layers] * num)
            
    def convert_decoder_only_to_encoder_decoder(self):
        self.enable_encoder = True
        self.encoder = copy.deepcopy(self.decoder)
        self.decoder_cross = copy.deepcopy(self.decoder)
        self.decoder_self = self.decoder


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

        out = self.absolutepositionalembedding(output)

        for _ in range(self.repeated_deberta_layers):
            for enc in self.deberta_layers:
                out = ckpt(enc,out,output)

        return out

class TransformerModel(nn.Module):

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
                    device: torch.DeviceObjType = device
                ) -> NoReturn :
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

    def init_weights(self) -> NoReturn :
        for w in self.parameters():
            w.data.uniform_(-1/16,1/16)
            
    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())
            
    def multiply_pretrained_transformer_layers(self,num: Optional[int] = 1) -> NoReturn :
        self.transformer_encoder.pretrained_layer_multiplier(num)

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
                        if (inp.size(1)-j+1)/inp.size(1) >= ((-1)*shuffle_continuous_pos):
                            rnd: float = 0
                    else:
                        if (j+1)/inp.size(1) >= shuffle_continuous_pos:
                            rnd: float = 0
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

            count: int = 0
            together_count: int = 0
            for j in range(inp.size(1)):
                rnd: float = -1
                if mask_continuous_pos < -100 or mask_continuous_pos > 100:
                    rnd: float = random.randint(0,100000)/1000
                elif mask_continuous_pos >= -100 and mask_continuous_pos <= 100:
                    mask_together_nos = mask_percentage * (inp.size(1)/100)
                    if mask_continuous_pos < 0:
                        if (inp.size(1)-j+1)/inp.size(1) >= ((-1)*mask_continuous_pos):
                            rnd: float = 0
                    else:
                        if (j+1)/inp.size(1) >= mask_continuous_pos:
                            rnd: float = 0
                if (((rnd>=0 and rnd<mask_percentage) or together_count<mask_together_nos) and mask and (((count+1)/inp.size(1))<=mask_percentage)):
                    inp_2[i,j] = 5
                    count += 1
                    together_count += 1
                elif together_count>=mask_together_nos:
                    together_count = 0
        return inp_2

    def encode_text(self,
                        *args: Union[str,Tensor],
                        append_pad_at_start: Union[bool,int] = True,
                        append_pad_at_end: Union[bool,int] = True,
                        padding: Union[int,bool] = 8,
                        pad_to_make_len_a_multiple: bool = True,
                        pad_idx: int = 0,
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
                    ) -> Union[list,Tensor] :
        encoded_text = []
        for txt in args:
            if type(txt) == str:
                tmp = self.tokenizer.batch_encode(txt)
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
            #tmp = torch.cat((torch.tensor([[pad_idx]]),tmp),dim=1)

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
                if txt.size(0) > 1:
                    txt = self.tokenizer.batch_decode(txt)
                else:
                    txt = self.tokenizer.decode(txt)
            decoded_text.append(txt)

    #@autocast()
    def forward(self,
                    src:Tensor,
                    context: Optional[Tensor] = None,
                    mem: Optional[Tensor] = None, 
                    context_mem: Optional[Tensor] = None,
                    alt_mem_with_primary_key: Optional[bool] = None,
                    assign_to_alt_mem: Optional[bool] = True,
                    return_mem: Optional[bool] = True,
                ) -> Tuple[Union[Tensor,Optional[Tensor]]]:
        
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
            context = context.contiguous()

        output = output.contiguous()

        output2 = ckpt(self.ffd1,output)
        output = ckpt(self.norm1,output2) + output

        if self.encoder_decoder:
            context2 = ckpt(self.ffd3,context)
            context = ckpt(self.norm1, context2) + context

        output = ckpt(self.transformer_encoder,output,context)

        output2 = ckpt(self.ffd2,output)
        output = ckpt(self.norm2,output2) + output


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