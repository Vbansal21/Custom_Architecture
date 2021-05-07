from logging import exception
import math
import torch
from torch import tensor
#from torch._C import int64
import torch.nn as nn
import torch.nn.functional as F


import io
from torchtext.utils import download_from_url, extract_archive
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator
from torchnlp.encoders.text import SubwordEncoder

import copy
from typing import Tuple, Optional, Any

from torch import Tensor
#from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
#from torch.nn.modules.linear import Linear
#from torch.nn.modules.normalization import LayerNorm

from typing import  Optional
#from torch.nn.modules.linear import _LinearWithBias
#from torch.nn.init import constant_
#from torch.nn.parameter import Parameter
from einops import repeat

torch.set_num_threads(12)
torch.set_num_interop_threads(12)

from x_transformers.x_transformers import RMSNorm
from ET_torch.models.evolved_transformer_block import EvolvedTransformerBlock
#from x_transformers.x_transformers import GEGLU
from product_key_memory import PKM
from hopfield_modules import Hopfield
from performer_torch import SelfAttention

from torch.utils.checkpoint import checkpoint #as ckpt

def ckpt(f,*args,checkpointed = True):
    if checkpointed:
        return checkpoint(f,*args)
    else:
        return f(*args)

from pytorch_model_summary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())
#device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)
#scaler = torch.cuda.amp.GradScaler(init_scale=2**3)
#autocast = torch.cuda.amp.autocast

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)

from x_transformers.x_transformers import GRUGating

class TransformerEncoderLayer(nn.Module):
    r"""Advances in
    Neural Information Processing Systems
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        #self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)        
        #attn = SelfAttention(d_model,heads=nhead,dim_head=d_model//nhead)
        #hopfield = Hopfield(input_size=d_model,num_heads=nhead)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = Dropout(dropout)

        self.ffd = nn.Sequential(
            nn.Linear(d_model,dim_feedforward),
            nn.GELU(),
            GEGLU(dim_feedforward,dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward,d_model),
            nn.GELU(),
        )
        self.pkm = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
            PKM(d_model),
            nn.GELU(),
            nn.Linear(d_model,d_model),
            nn.GELU(),
            )
        
        
        self.self_attn1 = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=SelfAttention(d_model,
                                                causal=False,
                                                heads=nhead,
                                                dim_head=d_model//nhead,
                                                num_mem_kv=2048,
                                                to_q=copy.deepcopy(self.pkm)
                                            ),
                            ffd=copy.deepcopy(self.ffd),
                            context=False,
                            pkm=copy.deepcopy(self.pkm)
                            )
        self.self_attn2 = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=Hopfield(
                                input_size=d_model,
                                num_heads=nhead
                                ),
                            ffd=copy.deepcopy(self.ffd),
                            context=False,
                            pkm=copy.deepcopy(self.pkm)
                            )
        
        self.gate = GRUGating(d_model)


    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:

        output = self.norm1(src)
        output = ckpt(self.ffd,output) + ckpt(self.pkm,output)
        src2 = ckpt(self.self_attn1,output)
        src3 = ckpt(self.self_attn2,output)
        del(output)

        output = src2 + src3
        output = self.dropout(output)
        output = ckpt(self.gate,output,src)
        del(src2,src3)

        src = self.norm2(output) + src
        del(output)
        return src

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, d_model,num_parallel_layers = 3):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        d_model = d_model
        self.num_parallel_layers = num_parallel_layers
        self.linear1 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.num_parallel_layers)])
        self.linear2 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.num_parallel_layers)])
        #self.enc = encoder_layer
        self.num_layers = num_layers
        self.norm = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        if self.num_parallel_layers != 0:
            self.norm1 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = None
            self.norm3 = None

    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:
        output = src
        out = []

        if self.num_parallel_layers > 0:
            for layer in self.linear1:
               out.append(self.norm1(ckpt(layer,output))+output)
        else:
            out = [src]

        for enc in self.layers:
            tmp = []

            for i in out:
                out2 = self.norm2(ckpt(enc,i)) + i
                tmp.append(out2)

            out = tmp.copy()

        tmp = []
        if self.num_parallel_layers > 0:
            for i,layer in enumerate(self.linear2):
                tmp.append(self.norm3(ckpt(layer,out[i])) + out[i])
            out = tmp.copy()
            tmp = None
            for i in out:
                if tmp is None:
                    tmp = i
                else:
                    tmp += i
            output = tmp
        else:
            output = out[0]

        if self.norm != None:
            output = self.norm(output)
        return output

class greedy_input_processing(nn.Module):

    def __init__(self,
                    text_embedding: Optional[nn.Module] = None,
                    audio_encoder:  Optional[nn.Module] = None,
                    image_encoder:  Optional[nn.Module] = None,
                    voxel_encoder:  Optional[nn.Module] = None
                    ):
        super(greedy_input_processing,self).__init__()

        self.text_embedding = text_embedding
        self.audio_encoder  = audio_encoder
        self.image_encoder  = image_encoder
        self.voxel_encoder  = voxel_encoder

    def forward(self,*args) -> Tensor:
        
        output = None
        nxt_arg_start_pos = []
        for i in args:
            tmp = None
            assert type(i) == Tensor or type(i) == tensor
            if len(i.size()) == 2 and self.text_embedding != None:
                tmp = ckpt(self.text_embedding,i)
                nxt_arg_start_pos.append((tmp.size(1),'text'))
            elif len(i.size()) == 3 and self.audio_encoder != None:
                tmp = ckpt(self.audio_encoder,i)
                nxt_arg_start_pos.append((tmp.size(1),'audio'))
            elif len(i.size()) == 4 and self.image_encoder != None:
                tmp = ckpt(self.image_encoder,i)
                nxt_arg_start_pos.append((tmp.size(1),'image'))
            elif len(i.size()) == 5 and self.voxel_encoder != None:
                tmp = ckpt(self.voxel_encoder,i)
                nxt_arg_start_pos.append((tmp.size(1),'voxel'))
            else:
                raise("Currently not defined for the inputted value")
            
            if tmp == None:
                tmp = i

            if output is None:
                output = tmp
            else:
                output = torch.cat((output,tmp),dim=1)
            del(tmp)
        
        return output,nxt_arg_start_pos

class lm_head(nn.Module):

    def __init__(self,
                    text_decoder:  Optional[nn.Module] = None,
                    audio_decoder: Optional[nn.Module] = None,
                    image_decoder: Optional[nn.Module] = None,
                    voxel_decoder: Optional[nn.Module] = None
                    ):
        super(lm_head,self).__init__()

        self.text_decoder = text_decoder
        self.audio_decoder = audio_decoder
        self.image_decoder = image_decoder
        self.voxel_decoder = voxel_decoder

    def forward(self,src: Tensor,nxt_arg_start_pos: list):
        
        src_copy = src
        output = []
        for (key,value) in nxt_arg_start_pos:
            tmp = None
            if value=='text' and self.text_decoder != None:
                tmp = src_copy[:,:key] if src_copy.size(1) != key else src_copy
                src_copy = src_copy[:,key:] if src_copy.size(1) != key else src_copy
                tmp = ckpt(self.text_decoder,tmp)
            elif value=='audio' and self.audio_decoder != None:
                tmp = src_copy[:,:key] if src_copy.size(1) != key else src_copy
                src_copy = src_copy[:,key:] if src_copy.size(1) != key else src_copy
                tmp = ckpt(self.audio_decoder,tmp)
            elif value=='image' and self.image_decoder != None:
                tmp = src_copy[:,:key] if src_copy.size(1) != key else src_copy
                src_copy = src_copy[:,key:] if src_copy.size(1) != key else src_copy
                tmp = ckpt(self.image_decoder,tmp)
            elif value=='voxel' and self.voxel_decoder != None:
                tmp = src_copy[:,:key] if src_copy.size(1) != key else src_copy
                src_copy = src_copy[:,key:] if src_copy.size(1) != key else src_copy
                tmp = ckpt(self.voxel_decoder,tmp)
            else:
                raise("Currently not defined for the inputted value")
            output.append(tmp)
            del(tmp)
        output = tuple(output)
        return output


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers,num_parallel_layers = 0, dropout=0.5,activation='gelu',mem_token=00):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, ninp, num_parallel_layers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        
        embedding_encoder = nn.Sequential(
            nn.Embedding(ntoken, ninp),
            nn.GELU(),
            nn.Linear(ninp,nhid*2),
            nn.GELU(),
            nn.Linear(nhid*2,ninp),
            nn.GELU(),
        )
        
        self.ninp = ninp

        
        self.decoder = nn.Sequential(
            nn.Linear(ninp,nhid*2),
            nn.GELU(),
            nn.Linear(nhid*2,ninp),
            nn.GELU(),
            nn.Linear(ninp,ntoken)
        )

        self.ffd1 = nn.Sequential(
            nn.Linear(ninp,nhid*2),
            nn.GELU(),
            nn.Linear(nhid*2,ninp),
            nn.GELU()
        )
        self.ffd2 = nn.Sequential(
            nn.Linear(ninp,nhid*2),
            nn.GELU(),
            nn.Linear(nhid*2,ninp),
            nn.GELU()
        )
        self.norm1 = RMSNorm(ninp)
        self.norm2 = RMSNorm(ninp)

        self.normalised_args_cat = embedding_encoder#embedding_encoder#greedy_input_processing(text_embedding=embedding_encoder)
        #self.decoder = decoder #lm_head(text_decoder=decoder)

        self.mem_exist = True if mem_token else False
        if self.mem_exist:
            if type(mem_token)==int:
                self.mem = nn.Parameter(torch.randn(mem_token,ninp))
            elif type(mem_token) == Tensor:
                assert mem_token.size(-1)==ninp
                self.mem = nn.Parameter(mem_token)
        
        self.alt_mem = None
        self.alt_mem_with_primary_mem = False

        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            w.data.uniform_(-1/4,1/4)
        #self.decoder.weight.data.uniform_(-1/4,1/4)
        #self.decoder.bias.data.uniform_(-1/4,1/4)


    def alt_mem_tokens(self,mem: Tensor,alt_mem_with_primary_mem: Optional[bool] == True):
        self.alt_mem = mem
        self.alt_mem_with_primary_mem = alt_mem_with_primary_mem

    #@autocast()
    def forward(self, src:Tensor,mem: Optional[Tensor] = None,context: Optional[Tensor] = None,alt_mem_with_primary_key: Optional[bool] = None,assign_to_alt_mem: Optional[bool] = True) -> Tensor:
        
        orignal_src = src

        src = self.normalised_args_cat(src)#,nxt_arg_start_pos
        src = src * math.sqrt(self.ninp)


        self.alt_mem_with_primary_mem = alt_mem_with_primary_key if type(alt_mem_with_primary_key) == bool else self.alt_mem_with_primary_mem

        if mem != None and self.mem_exist:
            if self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            elif not self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            else:
                output = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)

        elif self.mem_exist:
            if self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(self.mem, 'n d -> b n d', b = src.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            elif not self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            else:
                output = torch.cat((repeat(self.mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)


        output = self.pos_encoder(src)

        output2 = self.ffd1(output)
        output = ckpt(self.norm1,output2) + output

        output = self.transformer_encoder(output)

        output2 = ckpt(self.ffd2,output)
        output = ckpt(self.norm2,output2) + output


        for i in range(output.size(0)):
            if i == 0:
                mem = [output[i,:output.size(1)-src.size(1)]]
            else:
                mem += [output[i,:output.size(1)-src.size(1)]]
        mem = torch.cat(mem,dim=1) if type(mem) == torch.tensor else None
        self.alt_mem = mem if assign_to_alt_mem else None

        output = output[:,output.size(1)-src.size(1):] if type(mem) == torch.tensor else output


        output = self.decoder(output)#,nxt_arg_start_pos)

        """if len(output) == 1:
            output = output[0]"""
        return output,mem


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1,max_len=2**17):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
#url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
#tokenizer = get_tokenizer('basic_english')
try:
    tokenizer = torch.load("models/tokenizer.tar")
except:
    tokenizer = SubwordEncoder(io.open(train_filepath, encoding="utf8"),target_vocab_size=2**17,reserved_tokens=[
    '<pad>','<unk>','<s>','</s>','<copy>','<mask>','<segment>','</segment>','<non_text_content>','</non_text_content>'
    ])
    torch.save(tokenizer,"models/tokenizer.tar")
vocab = tokenizer.vocab


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).contiguous()
    return data

batch_size = 1
eval_batch_size = batch_size

bptt = 512#*2*2
import random
def random_mask_encoder(inp):
    inp_2 = inp.clone().detach()
    for i in range(inp.size(0)):
        for j in range(inp.size(1)):
            rnd = random.randint(0,6)
            if rnd==1:
                inp_2[i,j] = 5
            elif rnd==0:
                inp_2[i,j] = inp[i,random.randint(0,inp.size(1)-1)]
    inp_2 = inp_2
    return inp_2


def prepare_batch(source):
    data = random_mask_encoder(source)
    target = source
    return torch.cat((data.unsqueeze(0),target.unsqueeze(0)),dim=0)

def get_batch(source,j):
    seq_len = min(bptt, source.size(2) - j)
    return source[0,:,j:j+seq_len].to(device),source[1,:,j:j+seq_len].reshape(-1).to(device)

ntokens = tokenizer.vocab_size
emsize = 2048//8
nhid = emsize * 2 
nlayers = 16
nhead = 16
num_parallel_layers = 3
dropout = 0.3

use_deepspeed = False

deepspeed_args = {
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.03,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 0.03,
          "warmup_num_steps": 2000
      }
  },
  "fp16": {
    "enabled": True,
    "loss_scale": 0.5,
    "initial_scale_power": 3,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 0
    },
  "gradient_clipping":0.5,
  "zero_optimization": {
    "stage": 3,
    "offload_param":{
        "device": "nvme",
        "nvme_path":"/mnt/nvme0n1p3/"
        },
    "offload_optimizer": {
        "device": "nvme",
        "nvme_path": "/mnt/nvme0n1p3/"
        },
    "stage3_gather_fp16_weights_on_model_save": True,
        #"overlap_comm": True,
        #"contiguous_gradients": True,
        "sub_group_size": 1e3,
        #"stage3_param_persistence_threshold": 1e8,
    },
    "flops_profiler": {
    "enabled": False,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": False
    },
    "activation_checkpointing": {
    "partition_activations": True,
    "cpu_checkpointing": True,
    "contiguous_memory_optimization": True,
    "number_checkpoints": nlayers*2+4,
    "synchronize_checkpoint_boundary": True,
    "profile": False
    }
    
}

#from performer_torch import PerformerLM

if use_deepspeed:
    import deepspeed

    with deepspeed.zero.Init(mem_efficient_linear=False,remote_device='nvme',config=deepspeed_args,enabled=False):
        #model = PerformerLM(num_tokens=ntokens,max_seq_len=2**17,dim=emsize,depth=nlayers,heads=nhead,causal=True,use_rezero=True,cross_attend=True)
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,num_parallel_layers, dropout)


model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,num_parallel_layers, dropout,mem_token=100).to(device)

#print(sum(p.numel() for p in model.parameters()))
import time
date_time = str(time.asctime().replace(" ","_")).replace(":","_")
path = "models"+"/model_"+str(emsize)+"_"+str(nlayers)+"_"+str(nhead)+"_"+str(num_parallel_layers)+".tar"

criterion = nn.CrossEntropyLoss()
lr = 0.03
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#optimizer = torch.optim.Adam(model.parameters(), lr= lr,betas=[0.8,0.99],eps=1e-8,weight_decay=3e-7)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
epoch = 0
best_val_loss = float("inf")

resume_batch = 0

train_eval_event = [date_time]

if use_deepspeed:
    model,optimizer,_,scheduler = deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler, config_params=deepspeed_args)

model.eval()
inp = torch.zeros((1,10),dtype=torch.long).to(device)
out = model(inp,assign_to_alt_mem=False)
print(torch.argmax((out.view(-1,ntokens)),dim=-1))
del(out)

best_model = model

try:
    try:
        checkpoint_ = torch.load(path, map_location=device)
    except:
        _,checkpoint_ = model.load_checkpoint(path,)

    epoch = checkpoint_['epoch']
    best_val_loss = checkpoint_['best_val_loss']
    vocab = checkpoint_['vocab']
    tokenizer = checkpoint_['tokenizer']
    """
    try:
        model.load_state_dict(checkpoint_['model_state_dict'],strict=False)
    except:
        try:
            model = checkpoint_['model']
        except Exception as e:
            print("Exception",e)
            """
    optimizer.load_state_dict(checkpoint_['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_['scheduler_state_dict'])

    try:
        resume_batch = checkpoint_['resume_batch']
        train_eval_event = checkpoint_['train_eval_events'] + [date_time]
    except Exception as e:
        print("Exception",e)
    try:
        best_model = checkpoint_['best_model']
    except:
        pass
    del(checkpoint_)
    torch.cuda.empty_cache()
except Exception as e:
    print("Exception",e)
    pass

def data_process(raw_text_iter):
  data = tokenizer.encode(raw_text_iter)
  return torch.tensor(data)

try:
    processed_train_data = torch.load("models/data/train")
    processed_test_data = torch.load("models/data/test")
    processed_val_data = torch.load("models/data/val")
except:

    train_data = data_process(io.open(train_filepath, encoding="utf8").read()).to(device)
    val_data = data_process(io.open(valid_filepath, encoding="utf8").read()).to(device)
    test_data = data_process(io.open(test_filepath, encoding="utf8").read()).to(device)

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)



    processed_train_data = prepare_batch(train_data)
    processed_test_data = prepare_batch(test_data)
    processed_val_data = prepare_batch(val_data)

    del(train_data,test_data,val_data)

    torch.save(processed_train_data,"models/data/train.tar")
    torch.save(processed_test_data,"models/data/test.tar")
    torch.save(processed_val_data,"models/data/val.tar")

torch.cuda.empty_cache()
#model.to(device)

print(summary(model, torch.zeros([5,10],dtype=torch.long).to(device)))


def inference(text,size=128,eval_model = best_model):
    model.eval()
    text_input = torch.cat((data_process(text).unsqueeze(0),torch.full(tuple([1,size]),5)),dim=1)
    #mask = eval_model.generate_square_subsequent_mask(text_input.size(1)).to(device)
    out,_= eval_model(text_input)
    out = torch.argmax(out.view(-1, ntokens),dim=-1)
    result = tokenizer.decode(out)
    return [tokenizer.decode(text_input.view(-1)),result]

def train(resume_batch=0,step_scheduler=1024,save_intermediate_intervel=4096,mini_batch_size=20):
    model.train() 
    total_loss = 0.
    start_time = time.time()
    single_pass_mem = None
    #model.alt_mem = None
    #model.alt_mem_tokens(None,False)
    acc = 0
    for batch, i in enumerate(range(0, processed_train_data.size(2), bptt)):
        if resume_batch != None:
            if batch < resume_batch:
                continue
        data, targets = get_batch(processed_train_data, i)
        #with autocast():
        output,single_pass_mem = model(data,mem = single_pass_mem)
        loss = criterion(output.view(-1, ntokens), targets)
        if use_deepspeed:
            model.backward(loss)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()/output.size(1)
        if batch % mini_batch_size == 0 or i == processed_train_data.size(2)-1:
            if use_deepspeed:
                model.step()
                model.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        del(data,targets)
        torch.cuda.empty_cache()

        total_loss += loss.item()
        log_interval = 20
        if batch % log_interval == 0 and batch > 0 and batch != resume_batch:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:04.5f} | ms/batch {:08.3f} | acc {:3.2f}% | '
                  'loss {:5.3f} | ppl {:10.3f}'.format(
                    epoch, batch, processed_train_data.size(2) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,acc*100/log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            acc = 0
            start_time = time.time()
        if batch % save_intermediate_intervel == 0 and batch > 0:
            
            torch.save(
            {
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'best_model': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':batch,
                'train_eval_events': train_eval_event
            },
            path
            )
            """
            ckpt_id = epoch*(processed_train_data.size(-1)//bptt) + batch
            model.save_checkpoint(path,ckpt_id,client_sd = {
                'epoch': epoch,
                'best_model': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':batch,
                'train_eval_events': train_eval_event
            })"""
        if step_scheduler != None:
            if (batch % step_scheduler == 0 and batch > 0) or (epoch >1 and batch == 0 and processed_train_data.size(2)//bptt < step_scheduler):
                scheduler.step()

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    total_acc = 0.
    single_pass_mem = None
    with torch.no_grad():
        for i in range(0, data_source.size(2), bptt):
            data, targets = get_batch(data_source, i)
            output,single_pass_mem = eval_model(data,mem = single_pass_mem)
            output_flat = output.view(-1, ntokens)
            total_loss += data.size(1) * criterion(output_flat, targets).item()
            total_acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()/output.size(1)
    return total_loss / (data_source.size(2)), total_acc / (data_source.size(2))

epochs = 30

while True:
    if epoch >= epochs:
        break
    epoch +=1
    epoch_start_time = time.time()
    train(resume_batch=resume_batch,mini_batch_size= 1)
    resume_batch = 0
    val_loss, val_acc = evaluate(model, processed_val_data)
    print('-' * 105)
    print('| end of epoch {:3d} | time: {:08.3f}s | valid acc {:3.2f}% | valid loss {:5.3f} | '
          'valid ppl {:10.3f}'.format(epoch, (time.time() - epoch_start_time),val_acc*100,
                                     val_loss, math.exp(val_loss)))
    print('-' * 105)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'best_model': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':0,
                'train_eval_events': train_eval_event
            },
            path
        )
        """
        ckpt_id = epoch*(processed_train_data.size(-1)//bptt) + batch
        model.save_checkpoint(path,ckpt_id,client_sd = {
                'epoch': epoch,
                'best_model': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':0,
                'train_eval_events': train_eval_event
            })"""
model = best_model

test_loss,test_acc = evaluate(best_model, processed_test_data)
print('=' * 105)
print('| End of training | test acc {:3.2f}% | test loss {:5.3f} | test ppl {:10.3f}'.format(test_acc*100,
    test_loss, math.exp(test_loss)))
print('=' * 105)

print(inference("Hello World!!! This is inference function on the currently trained model"))

while True:
    i = int(input("enter 1 for inference, 0 for exiting:"))
    if i == 0:
        break
    inp = input("input text, 1 string at a time, for inference:")
    print(inference(inp))