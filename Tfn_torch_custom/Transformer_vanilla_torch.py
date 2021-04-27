from logging import exception
import math
import torch
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

torch.set_num_threads(10)
torch.set_num_interop_threads(10)

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
            Linear(d_model,d_model//2),
            _get_activation_fn(activation),
            PKM(d_model//2),
            _get_activation_fn(activation),
            Linear(d_model//2,d_model),
            _get_activation_fn(activation)
            )
        
        from ET_torch.models.evolved_transformer_block import EvolvedTransformerBlock
        self.self_attn1 = EvolvedTransformerBlock(d_model,
                            num_heads=nhead,
                            attn=SelfAttention(d_model,
                                                heads=nhead,
                                                dim_head=d_model//nhead
                                            ),
                            ffd=copy.deepcopy(self.ffd),
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

    def __init__(self, encoder_layer, num_layers, d_model,num_parallel_layers = 3, mem_size = 100):
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

        self.mem_exist = True if mem_size else False
        if self.mem_exist:
            self.mem = torch.randn(mem_size,d_model).to(device)

        if self.num_parallel_layers != 0:
            self.norm1 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = None
            self.norm3 = None

    def initialise_single_pass_mem(self,mem:Tensor):
        self.mem = mem
        self.init_mem_dict = False

    def forward(self, src: Tensor,context: Optional[Tensor] = None) -> Tensor:
        output = src
        out = []

        output = torch.cat((repeat(self.mem, 'n d -> b n d', b = output.size(0)),output),dim=-2)

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

        for i in range(output.size(0)):
            if i == 0:
                self.mem = output[i,:output.size(1)-src.size(1)]
            else:
                self.mem += output[i,:output.size(1)-src.size(1)]
        output = output[:,output.size(1)-src.size(1):]

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers,num_parallel_layers = 0, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, ninp, num_parallel_layers).to(device=device)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.nlayers = nlayers
        self.ffd1 = nn.Sequential(
            nn.Linear(ninp,nhid*2),
            nn.ReLU(),
            nn.Linear(nhid*2,ninp),
            nn.ReLU()
        )
        self.ffd2 = copy.deepcopy(self.ffd1)
        self.norm1 = RMSNorm(ninp)
        self.norm2 = RMSNorm(ninp)

        self.init_weights()

    def init_weights(self):
        initrange = 0.2
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src:Tensor,context: Optional[Tensor] = None,mem: Optional[dict] = None) -> Tensor:

        if mem != None:
            self.transformer_encoder.initialise_single_pass_mem(mem)
        src = ckpt(self.encoder,src) * math.sqrt(self.ninp)
        output = self.pos_encoder(src)
        output2 = ckpt(self.ffd1,output)
        output = ckpt(self.norm1,output2) + output
        output = self.transformer_encoder(output)
        mem = self.transformer_encoder.mem
        output2 = ckpt(self.ffd2,output)
        output = ckpt(self.norm2,output2) + output
        output = ckpt(self.decoder,output)
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
tokenizer = SubwordEncoder(io.open(train_filepath, encoding="utf8"),target_vocab_size=50000,reserved_tokens=[
    '<pad>','<unk>','</s>','<s>','<copy>','<mask>'
])
vocab = tokenizer.vocab


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).contiguous()
    return data.to(device)

batch_size = 1
eval_batch_size = batch_size

bptt = 400
import random
def random_mask_encoder(inp):
    inp_2 = torch.full(inp.size(),5).to(device=device)
    for i in range(inp.size(0)):
        for j in range(inp.size(1)):
            rnd = random.randint(0,2)
            if rnd==1:
                inp_2[i,j] = copy.deepcopy(inp[i,j])
            elif rnd==2:
                inp_2[i,j] = copy.deepcopy(inp[i,random.randint(0,inp.size(1)-1)])
    inp_2 = inp_2.clone().detach()
    #del(inp)
    inp_2.to(device)
    return inp_2


def get_batch(source, i):
    seq_len = min(bptt, source.size(1) - i)
    data = source[:,i:i+seq_len]
    data = random_mask_encoder(data)
    target = source[:,i:i+seq_len].reshape(-1)
    return data, target

ntokens = tokenizer.vocab_size 
emsize = 512
nhid = emsize * 4 
nlayers = 12
nhead = 16
num_parallel_layers = 0
dropout = 0.3
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,num_parallel_layers, dropout)
"""
from x_transformers import TransformerWrapper, Decoder
model = TransformerWrapper(
    num_tokens = tokenizer.vocab_size ,
    max_seq_len = max(bptt,2048),
    emb_dropout = dropout,
    num_memory_tokens = 50,
    attn_layers = Decoder(
        dim = emsize,
        depth = nlayers,
        heads = nhead,
        attn_dropout = dropout,
        ff_dropout = dropout,
        attn_num_mem_kv = 50,
        use_scalenorm = True,
        ff_glu = True,
        attn_talking_heads = True,
        macaron = True,
        rel_pos_bias = True,
        residual_attn = True
    )
).to(device)
"""
#print(sum(p.numel() for p in model.parameters()))
import time
date_time = str(time.asctime().replace(" ","_")).replace(":","_")
#path = "/models/"+date_time+"/model_"+str(emsize)+"_"+str(nlayers)+"_"+str(nhead)+"_"+str(num_parallel_layers)+".tar"
path = "models"+"/model_"+str(emsize)+"_"+str(nlayers)+"_"+str(nhead)+"_"+str(num_parallel_layers)+".tar"

criterion = nn.CrossEntropyLoss()
lr = 0.3 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
epoch = 0
best_val_loss = float("inf")
best_model = model

resume_batch = 0


try:
    #model.load_state_dict(torch.load(path), strict=False)
    checkpoint_ = torch.load(path, map_location=device)

    epoch = checkpoint_['epoch']
    best_val_loss = checkpoint_['best_val_loss']
    try:
        model.load_state_dict(checkpoint_['model_state_dict'],strict=False)
    except:
        try:
            model = checkpoint_['model']
        except:
            pass
    optimizer.load_state_dict(checkpoint_['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_['scheduler_state_dict'])
    vocab = checkpoint_['vocab']
    tokenizer = checkpoint_['tokenizer']
    try:
        resume_batch = checkpoint_['resume_batch']
    finally:
        pass
    best_model = model
    del(checkpoint_)
    torch.cuda.empty_cache()
except Exception as e:
    print("Exception",e)
    pass


def data_process(raw_text_iter):
  data = tokenizer.encode(raw_text_iter)
  return torch.tensor(data) #torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_data = data_process(io.open(train_filepath, encoding="utf8").read())
val_data = data_process(io.open(valid_filepath, encoding="utf8").read())
test_data = data_process(io.open(test_filepath, encoding="utf8").read())

train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

torch.cuda.empty_cache()


model.to(device)

print(summary(model, torch.zeros([5,10],dtype=torch.long).to(device)))


def inference(text,size=128,eval_model = best_model):
    model.eval()
    text_input = torch.cat((data_process(text),torch.full((size),5)),dim=0).unsqueeze(0).to(device)
    #mask = eval_model.generate_square_subsequent_mask(text_input.size(1)).to(device)
    out,_= eval_model(text_input)
    out = torch.argmax(out.view(-1, ntokens),dim=-1)
    result = tokenizer.decode(out)
    return [text,result]


def train(resume_batch=None,step_scheduler=1024,save_intermediate_intervel=4096,mini_batch_size=20):
    model.train() 
    total_loss = 0.
    start_time = time.time()
    mem_dict = None
    single_pass_mem = None
    acc = 0
    #src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    #optimizer.zero_grad()
    for batch, i in enumerate(range(0, train_data.size(1) - 1, bptt)):
        if resume_batch != None:
            if batch < resume_batch:
                continue
        data, targets = get_batch(train_data, i)
    
        output,single_pass_mem = model(data,mem=single_pass_mem)#, src_mask)
        #with torch.cuda.amp.autocast():
        loss = criterion(output.view(-1, ntokens), targets)
        #loss = torch.autograd.Variable(loss,requires_grad=True)
        #scaler.scale(loss).backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()/output.size(1)
        if batch % mini_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        #optimizer.step()
        #scaler.step(optimizer)
        #scaler.update()
    

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0 and batch != resume_batch:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:04.3f} | ms/batch {:08.3f} | acc {:5.3f} | '
                  'loss {:5.3f} | ppl {:10.3f}'.format(
                    epoch, batch, train_data.size(1) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,acc/log_interval,
                    cur_loss, math.exp(cur_loss)))
            print(inference("Hello World!!! This is inference function on the currently trained model"))
            total_loss = 0
            acc = 0
            start_time = time.time()
        if batch % save_intermediate_intervel == 0 and batch > 0:

            torch.save(
            {
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                #'model': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':batch
            },
            path
            )
        if step_scheduler != None:
            if (batch % step_scheduler == 0 and batch > 0):# or (epoch >1 and batch == 0):
                scheduler.step()

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    total_acc = 0.
    single_pass_mem = None
    #src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output,single_pass_mem = eval_model(data,mem = single_pass_mem)#, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += data.size(1) * criterion(output_flat, targets).item()
            total_acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()/output.size(0)
    return total_loss / (data_source.size(1) - 1), total_acc / (data_source.size(1) - 1)

epochs = 20

while True:
    if epoch >= epochs:
        break
    epoch +=1
    epoch_start_time = time.time()
    train(resume_batch=resume_batch,mini_batch_size= 1)
    resume_batch = 0
    val_loss, val_acc = evaluate(model, val_data)
    print('-' * 113)
    print('| end of epoch {:3d} | time: {:08.3f}s | valid acc {:5.3f} | valid loss {:5.3f} | '
          'valid ppl {:10.3f}'.format(epoch, (time.time() - epoch_start_time),val_acc,
                                     val_loss, math.exp(val_loss)))
    print('-' * 113)

    #scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        #best_model = best_model.to(torch.device("cpu"))
        #best_model_state = copy.deepcopy(best_model.state_dict())
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                #'model': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab': vocab,
                'tokenizer': tokenizer,
                'resume_batch':0
            },
            path
        )
#best_model = best_model.to(device)
model = best_model

test_loss,test_acc = evaluate(best_model, test_data)
print('=' * 113)
print('| End of training | test acc {:5.3f} | test loss {:5.3f} | test ppl {:10.3f}'.format(test_acc,
    test_loss, math.exp(test_loss)))
print('=' * 113)

print(inference("Hello World!!! This is inference function on the currently trained model"))

while True:
    i = int(input("enter 1 for inference, 0 for exiting:"))
    if i == 0:
        break
    inp = input("input text, 1 string at a time, for inference:")
    print(inference(inp))