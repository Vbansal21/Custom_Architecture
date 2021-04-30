import deepspeed

from logging import exception
import math
import torch
from torch import tensor
#from torch._C import int64
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Tuple, Optional, Any

from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from typing import  Optional
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from einops import repeat

torch.set_num_threads(12)
torch.set_num_interop_threads(12)

from x_transformers.x_transformers import RMSNorm

from torch.utils.checkpoint import checkpoint #as ckpt

def ckpt(f,args,checkpointed = True):
    if checkpointed:
        return checkpoint(f,args)
    else:
        return f(args)

from pytorch_model_summary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)
#scaler = torch.cuda.amp.GradScaler()

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    r"""Advances in
    Neural Information Processing Systems
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        #self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        from performer_torch import SelfAttention
        #attn = SelfAttention(d_model,heads=nhead,dim_head=d_model//nhead)

        from hopfield_modules import Hopfield
        #hopfield = Hopfield(input_size=d_model,num_heads=nhead)

        from product_key_memory import PKM

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = Dropout(dropout)
        
        from x_transformers.x_transformers import GEGLU

        self.ffd = nn.Sequential(
            nn.Linear(d_model,dim_feedforward),
            _get_activation_fn(activation),
            nn.Linear(dim_feedforward,d_model),
            _get_activation_fn(activation),
            GEGLU(d_model,d_model),
            _get_activation_fn(activation)
        )
        self.pkm = nn.Sequential(
            Linear(d_model,d_model),
            _get_activation_fn(activation),
            PKM(d_model),
            _get_activation_fn(activation),
            Linear(d_model,d_model),
            _get_activation_fn(activation)
            )
        
        from ET_torch.models.evolved_transformer_block import EvolvedTransformerBlock
        self.self_attn1 = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=SelfAttention(d_model,
                                                causal=False,
                                                heads=nhead,
                                                dim_head=d_model//nhead
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

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:

        output = self.norm1(src)
        output = ckpt(self.ffd,output) + ckpt(self.pkm,output)
        src2 = ckpt(self.self_attn1,output)
        src3 = ckpt(self.self_attn2,output)
        del(output)

        output = self.dropout(src2 + src3)
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
        self.linear1 = nn.ModuleList([Linear(d_model, d_model).to(device) for _ in range(self.num_parallel_layers)])
        self.linear2 = nn.ModuleList([Linear(d_model, d_model).to(device) for _ in range(self.num_parallel_layers)])
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

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers,num_parallel_layers = 0, dropout=0.5,activation='gelu',mem_token=100):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, ninp, num_parallel_layers).to(device=device)
        #self.encoder = nn.Embedding(ntoken, ninp)

        embedding_encoder = nn.Sequential(
            nn.Embedding(ntoken, ninp),
            _get_activation_fn(activation),
            nn.Linear(ninp,nhid*2),
            _get_activation_fn(activation),
            nn.Linear(nhid*2,ninp),
            _get_activation_fn(activation)
        )

        self.ninp = ninp

        
        decoder = nn.Sequential(
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

        self.normalised_args_cat = greedy_input_processing(text_embedding=embedding_encoder)
        self.decoder = lm_head(text_decoder=decoder)

        self.mem_exist = True if mem_token else False
        if self.mem_exist:
            if type(mem_token)==int:
                self.mem = nn.Parameter(torch.randn(mem_token,ninp)).to(device)
            elif type(mem_token) == Tensor:
                assert mem_token.size(-1)==ninp
                self.mem = nn.Parameter(mem_token)
        
        self.alt_mem = None
        self.alt_mem_with_primary_mem = False

    def alt_mem_tokens(self,mem: Tensor,alt_mem_with_primary_mem: Optional[bool] == True):
        self.alt_mem = mem
        self.alt_mem_with_primary_mem = alt_mem_with_primary_mem

    def forward(self, src:Tensor,*args,context: Optional[Tensor] = None,mem: Optional[dict] = None,alt_mem_with_primary_key: Optional[bool] = None) -> Tensor:

        src,nxt_arg_start_pos = self.normalised_args_cat(src,*args)
        src = src * math.sqrt(self.ninp)


        if alt_mem_with_primary_key != None:
            self.alt_mem_with_primary_mem = alt_mem_with_primary_key

        if mem != None:
            if self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            elif not self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            else:
                output = torch.cat((repeat(mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)

        else:
            if self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(self.mem, 'n d -> b n d', b = src.size(0)),repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            elif not self.alt_mem_with_primary_mem and self.alt_mem != None:
                output = torch.cat((repeat(self.alt_mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)
            else:
                output = torch.cat((repeat(self.mem, 'n d -> b n d', b = src.size(0)),src),dim=-2)


        output = self.pos_encoder(src)

        output2 = ckpt(self.ffd1,output)
        output = ckpt(self.norm1,output2) + output

        output = self.transformer_encoder(output)

        output2 = ckpt(self.ffd2,output)
        output = ckpt(self.norm2,output2) + output


        for i in range(output.size(0)):
            if i == 0:
                mem = [output[i,:output.size(1)-src.size(1)]]
            else:
                mem += [output[i,:output.size(1)-src.size(1)]]
        mem = torch.cat(mem,dim=1)
        output = output[:,output.size(1)-src.size(1):]


        output = self.decoder(output,nxt_arg_start_pos)

        if len(output) == 1:
            output = output[0]

        return output,mem


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1,max_len=2**16):
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


import io
import torch
from torchtext.utils import download_from_url, extract_archive
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator
from torchnlp.encoders.text import SubwordEncoder


url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
#url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
#tokenizer = get_tokenizer('basic_english')
try:
    tokenizer = torch.load("models/tokenizer.tar")
except:
    tokenizer = SubwordEncoder(io.open(train_filepath, encoding="utf8"),target_vocab_size=50000,reserved_tokens=[
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

bptt = 512*2*2
import random
def random_mask_encoder(inp):
    inp_2 = inp.clone().detach()
    for i in range(inp.size(0)):
        for j in range(inp.size(1)):
            rnd = random.randint(0,6)
            if rnd==1:
                inp_2[i,j] = 5
            elif rnd==0:
                inp_2[i,j] = copy.deepcopy(inp[i,random.randint(0,inp.size(1)-1)])
    inp_2 = inp_2.clone().detach()
    return inp_2


def prepare_batch(source):
    data = random_mask_encoder(source)
    target = source
    return torch.cat((data.unsqueeze(0),target.unsqueeze(0)),dim=0).to(torch.device('cpu'))

def get_batch(source,j):
    seq_len = min(bptt, source.size(2) - j)
    return source[0,:,j:j+seq_len].to(device),source[1,:,j:j+seq_len].reshape(-1).to(device)

ntokens = tokenizer.vocab_size 
emsize = 512*2
nhid = emsize * 4 
nlayers = 8//8
nhead = 16
num_parallel_layers = 0
dropout = 0.3
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,num_parallel_layers, dropout).to(device=device)

print(sum(p.numel() for p in model.parameters()))
import time
date_time = str(time.asctime().replace(" ","_")).replace(":","_")
path = "models"+"/model_"+str(emsize)+"_"+str(nlayers)+"_"+str(nhead)+"_"+str(num_parallel_layers)+".tar"

criterion = nn.CrossEntropyLoss()
lr = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
epoch = 0
best_val_loss = float("inf")
best_model = model

resume_batch = 0

train_eval_event = [date_time]



deepspeed_args = {
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": True,
    "loss_scale": 0,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
    },
  "gradient_clipping":4.0,
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients" : True,
    "offload_param":{
        "device": "nvme",
        "nvme_path":"/dev/nvme0n1p3"
        },
    "offload_optimizer": {
       "device": "nvme",
       "nvme_path": "/dev/nvme0n1p3"
        },
    "elastic_checkpoint" : True,
    "stage3_gather_fp16_weights_on_model_save": True
    },
    "flops_profiler": {
    "enabled": True,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": True,
    },
    "activation_checkpointing": {
    "partition_activations": True,
    "cpu_checkpointing": True,
    "contiguous_memory_optimization": True,
    "number_checkpoints": nlayers,
    "synchronize_checkpoint_boundary": True,
    "profile": True
    }
    
}

model,optimizer,_,scheduler = deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config_params=deepspeed_args)

try:
    #checkpoint_ = torch.load(path, map_location=device)
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
    #optimizer.load_state_dict(checkpoint_['optimizer_state_dict'])
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
model.to(device)


model.eval()
model(torch.zeros([5,10],dtype=torch.long).to(device),torch.zeros([5,10],dtype=torch.long).to(device))

#print(summary(model, torch.zeros([5,10],dtype=torch.long).to(device)))


def inference(text,size=128,eval_model = best_model):
    model.eval()
    text_input = torch.cat((data_process(text).unsqueeze(0),torch.full(tuple([1,size]),5)),dim=1).to(device)
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
    acc = 0
    for batch, i in enumerate(range(0, processed_train_data.size(2), bptt)):
        if resume_batch != None:
            if batch < resume_batch:
                continue
        data, targets = get_batch(processed_train_data, i)
    
        output,single_pass_mem = model(data,mem=single_pass_mem)
        loss = criterion(output.view(-1, ntokens), targets)
        #loss.backward()
        model.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()/output.size(1)
        if batch % mini_batch_size == 0 or i == processed_train_data.size(2)-1:
            #optimizer.step()
            model.step()
            #optimizer.zero_grad()
        del(data,targets)
        torch.cuda.empty_cache()

        total_loss += loss.item()
        log_interval = 200
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
            """
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
            )"""
            ckpt_id = epoch*(processed_train_data.size(-1)//bptt) + batch
            model.save_checkpoint(path,ckpt_id,client_sd = {
                'epoch': epoch,
                'best_model': best_model,
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':batch,
                'train_eval_events': train_eval_event
            })
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


        """
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
        )"""
        batch = 0
        ckpt_id = epoch*(processed_train_data.size(-1)//bptt) + batch
        model.save_checkpoint(path,ckpt_id,client_sd = {
                'epoch': epoch,
                'best_model': best_model,
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':batch,
                'train_eval_events': train_eval_event
            })
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