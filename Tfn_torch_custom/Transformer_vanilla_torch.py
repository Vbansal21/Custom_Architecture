import io
import time
import copy
import math
import torch
import random,wandb
from torch import nn
import scripts.model
from torch import Tensor
from einops import rearrange
#from performer_torch import PerformerLM
from pytorch_model_summary import summary
from scripts.model import TransformerModel
from torchnlp.encoders.text import SubwordEncoder
from torchtext.utils import download_from_url, extract_archive
from typing import Tuple, Optional, Any, NoReturn, Union, Literal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)

#scaler = torch.cuda.amp.GradScaler(init_scale=2**3)
autocast = torch.cuda.amp.autocast

scripts.model.checkpointed = True
scripts.model.device = device

file = "wikitextv103"
if file == "wikitextv2":
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
elif file == "wikitextv103":
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
try:
    tokenizer = torch.load("models/tokenizer.tar")
except:
    tokenizer = SubwordEncoder(io.open(train_filepath, encoding="utf8"),target_vocab_size=2**16,reserved_tokens=[
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

ntokens = tokenizer.vocab_size
emsize = 2048//4
nhid = emsize * 4
nlayers = 3
deberta_layers = 6
repeated_deberta_layers = 1
nhead = 16
dropout = 0.3
mem_tokens = 512
bptt = (1024+mem_tokens) - mem_tokens
max_seq_len = 2**16

use_deepspeed = False

deepspeed_args = {
  "train_batch_size": batch_size,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": True,
    "loss_scale": 0.5,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
    },
  "gradient_clipping":0.5,
  "zero_optimization": {
    "stage": 3,
    'allgather_partitions': True,
    "allgather_bucket_size":1,
    "reduce_bucket_size":1,
    "offload_param":{
        "device": "nvme",
        "nvme_path":"/home/vbansal21/",
        "buffer_count":5+nlayers,
        "buffer_size":1,
        "max_in_cpu":1e7
        },
    "offload_optimizer": {
        "device": "nvme",
        "nvme_path": "/home/vbansal21/",
        "buffer_count":5+nlayers,
        #"fast_init":True
        },
    "stage3_gather_fp16_weights_on_model_save": True,
    #"stage3_max_live_parameters":1e8,
    #"stage3_max_reuse_distance":1e8,
    "stage3_prefetch_bucket_size":1e6,
        "overlap_comm": True,
        #"contiguous_gradients": True,
        "sub_group_size": 1,
        "stage3_param_persistence_threshold": 1e7,
    },
    "wall_clock_breakdown":True,
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
    "contiguous_memory_optimization": False,
    "number_checkpoints": nlayers*2+4,
    "synchronize_checkpoint_boundary": True,
    "profile": False
    }
}


if use_deepspeed:
    import deepspeed

    with deepspeed.zero.Init(mem_efficient_linear=True,remote_device='nvme',config=deepspeed_args,enabled=True):
        #model = PerformerLM(num_tokens=ntokens,max_seq_len=2**17,dim=emsize,depth=nlayers,heads=nhead,causal=True,use_rezero=True,cross_attend=True)
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout=dropout,deberta_layers=deberta_layers,repeated_deberta_layers=repeated_deberta_layers,mem_token=mem_tokens,max_seq_len=max_seq_len).half()
else:
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout=dropout,mem_token=mem_tokens,deberta_layers=deberta_layers,repeated_deberta_layers=repeated_deberta_layers,max_seq_len=max_seq_len,device=device).to(device)

def random_mask_shuffle_encoder(
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
                    if (((j+1)/inp.size(1)) + (shuffle_percentage/100)) >= ((-1)*shuffle_continuous_pos/100):
                        rnd: float = 0
                else:
                    if (j+1)/inp.size(1) >= shuffle_continuous_pos:
                        rnd: float = -1
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
            if mask_continuous_pos < -100 or mask_continuous_pos > 100:
                rnd: float = random.randint(0,100000)/1000
            elif mask_continuous_pos >= -100 and mask_continuous_pos <= 100:
                mask_together_nos = mask_percentage * (inp.size(1)/100)
                if mask_continuous_pos < 0:
                    if (((j+1)/inp.size(1)) + (mask_percentage/100)) >= ((-1)*mask_continuous_pos/100):
                        rnd: float = 0
                else:
                    if (j+1)/inp.size(1) >= mask_continuous_pos:
                        rnd: float = 0
            if (((rnd>=0 and rnd<mask_percentage) or (together_count<mask_together_nos and together_count!=0)) and mask and (((count+1)/inp.size(1))<=mask_percentage/100)):
                inp_2[i,j] = 5
                count += 1
                together_count += 1
            elif together_count>=mask_together_nos:
                together_count = 0
    return inp_2

def random_mask_encoder(inp):
    inp_2 = inp.clone().detach()
    for i in range(inp.size(0)):
        for j in range(inp.size(1)):
            rnd = random.randint(0,6)
            if rnd==1:
                inp_2[i,j] = 5
            elif rnd==0:
                inp_2[i,j] = inp[i,random.randint(0,inp.size(1)-1)]
    return inp_2

def prepare_batch(source):
    data = source.clone().detach()
    target = source
    return torch.cat((data.unsqueeze(0).to(torch.device('cpu')),target.unsqueeze(0).to(torch.device('cpu'))),dim=0)

def get_batch(source,j,bptt=bptt):
    seq_len = min(bptt -1 -1, source.size(2) - j -1 -1 -1)
    data = random_mask_shuffle_encoder(source[0,:,j:j+seq_len-1],mask_percentage=20,mask_together_nos=10,mask_continuous_pos=-21,shuffle_percentage=10).to(device)
    data = torch.cat((torch.full((data.size(0),1),2,dtype=torch.long,device=device),data,torch.full((data.size(0),2),3,dtype=torch.long,device=device)),dim=1).contiguous()
    targets = source[1,:,j:j+seq_len].to(device)
    targets = torch.cat((targets,torch.full((targets.size(0),1),3,dtype=torch.long,device=device),torch.full((targets.size(0),1),3,dtype=torch.long,device=device)),dim=1).contiguous()
    return data,targets

#print(sum(p.numel() for p in model.parameters()))
date_time = str(time.asctime().replace(" ","_")).replace(":","_")
path = "models"+"/model_"+str(emsize)+"_"+str(nlayers)+"_"+str(nhead)+".tar"

criterion = nn.CrossEntropyLoss()
lr = 0.0175
if not use_deepspeed:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.8,0.999),weight_decay=3e-7,eps=1e-8)
a = 5000000
b = 500
c = 0.0
step = 1
lambda_1 = lambda step: (((a/b * step + 1) / (step**2 + a)) + c)/(step**0.1+1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda_1)
load_scheduler = True
epoch = 0
best_val_loss = float("inf")

resume_batch = 0
epochs = 30

train_eval_event = [date_time]

if use_deepspeed:
    model,optimizer,_,scheduler = deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler, config_params=deepspeed_args)

model.eval()
inp = torch.randint(0,ntokens-1,(1,bptt),dtype=torch.long,device=device)
if use_deepspeed:
    with autocast():
        out,mem = model(inp,assign_to_alt_mem=False)
else:
    out,mem = model(inp,assign_to_alt_mem=False)
print(torch.argmax((out.view(-1,ntokens)),dim=-1))
del(out,mem,inp)

best_model = None

project_name = "Tfn_X"

wandb.init(project=project_name,config={
    "ntokens":ntokens,
    "d_model":emsize,
    "ffd":nhid,
    "layers":nlayers,
    "heads":nhead,
    "deberta_layers":deberta_layers,
    "repeated_deberta_layers":repeated_deberta_layers,
    "dropout":dropout,
    "memory_tokens":mem_tokens,
    "total_epochs":epochs,
    "Sequence_length":bptt,
    "max_seq_len":max_seq_len,
    "seq_scale_down":8
})


try:
    try:
        checkpoint_ = torch.load(path, map_location=device)
    except:
        _,checkpoint_ = model.load_checkpoint(path,)

    epoch = checkpoint_['epoch']
    best_val_loss = checkpoint_['best_val_loss']
    vocab = checkpoint_['vocab']
    tokenizer = checkpoint_['tokenizer']
    
    try:
        model.load_state_dict(checkpoint_['model_state_dict'],strict=False)
    except:
        try:
            model = checkpoint_['model']
        except Exception as e:
            print("Exception",e)
            
    optimizer.load_state_dict(checkpoint_['optimizer_state_dict'])
    step = checkpoint_['step_number']
    if load_scheduler:
        scheduler.load_state_dict(checkpoint_['scheduler_state_dict'])
    else:
        lambda_1 = lambda step: (((a/b * step + 1) / (step**2 + a)) + c)/(step**0.1+1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda_1)

    try:
        resume_batch = checkpoint_['resume_batch']
        train_eval_event = checkpoint_['train_eval_events'] + [date_time]
    except Exception as e:
        print("Exception",e)
    try:
        #best_model = checkpoint_['best_model'].to(torch.device('cpu'))
        pass
    except Exception as e:
        print("Exception",e)
        pass
    del(checkpoint_)
    torch.cuda.empty_cache()
except Exception as e:
    print("Exception",e)
    pass

if best_model==None:
    best_model=model

def data_process(raw_text_iter):
  data = tokenizer.encode(raw_text_iter)
  return torch.tensor(data)

try:
    processed_train_data = torch.load("models/data/"+file+"_train.tar")
    processed_test_data = torch.load("models/data/"+file+"_test.tar")
    processed_val_data = torch.load("models/data/"+file+"_val.tar")
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

    torch.save(processed_train_data,"models/data/"+file+"_train.tar")
    torch.save(processed_test_data,"models/data/"+file+"_test.tar")
    torch.save(processed_val_data,"models/data/"+file+"_val.tar")

torch.cuda.empty_cache()
model.to(device)

print(summary(model, torch.zeros([1,bptt],dtype=torch.long).to(device)))

# TODO: Setup 'curses' module to print colored text for inference output
#import curses
def inference(text,size=128,eval_model = best_model,reccurent_mem=None):
    model.eval()
    text_input = torch.cat((torch.full(tuple([1,1]),2),data_process(text).unsqueeze(0),torch.full(tuple([1,size]),5),torch.full(tuple([1,1]),3)),dim=1).to(device )
    #mask = eval_model.generate_square_subsequent_mask(text_input.size(1)).to(device)
    if use_deepspeed:
        with autocast():
            out,mem = eval_model(text_input,mem=reccurent_mem,assign_to_alt_mem=False)
    else:
        out,mem = eval_model(text_input,mem=reccurent_mem,assign_to_alt_mem=False)
    out = torch.argmax(out.view(-1, ntokens),dim=-1)
    result = tokenizer.decode(out)
    print("Your input:\n\t",tokenizer.decode(text_input.view(-1)))
    print("Model's Output:\n\t",result)
    return mem


_ = inference("Hello World!!! This is inference function on the currently trained model")


def train(resume_batch=0,step_scheduler=1,save_intermediate_intervel=8192,mini_batch_size=20,step=step):
    model.train() 
    total_loss = 0.
    start_time = time.time()
    single_pass_mem = None
    #model.alt_mem = None
    #model.alt_mem_tokens(None,False)
    acc = 0
    total_acc = 0
    for batch, i in enumerate(range(0, processed_train_data.size(2), bptt-1-1)):
        if resume_batch != None:
            if batch < resume_batch:
                continue
        if epoch%2==0:
            single_pass_mem = None
        data, targets = get_batch(processed_train_data, i)
        if use_deepspeed:
            with autocast():
                output,single_pass_mem = model(data,mem=single_pass_mem)
                loss = criterion(rearrange(output,'b n c -> n c b'), rearrange(targets,'b n -> n b'))
        else:
            output,single_pass_mem = model(data,mem=single_pass_mem)
            loss = criterion(rearrange(output,'b n c -> n c b'), rearrange(targets,'b n -> n b'))

        if use_deepspeed:
            #scaler.scale(loss).backward()
            #loss.backward(reatain_graph=True)
            model.backward(loss)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        acc = ((torch.argmax(output,dim=-1)) == targets).sum().item()/output.size(1)
        try:
            ppl = math.exp(loss.item())
        except:
            ppl = -1.0
        wandb.log(
            {
                "loss":loss.item(),
                "step":step,
                "accuracy(%)":acc*100,
                "epoch":epoch,
                "batch":batch,
                "perplexity":ppl,
                'Learning_Rate':scheduler.get_lr()[0]
                #"input":wandb.Html(tokenizer.decode(data)),
                #"target":wandb.Html(tokenizer.decode(targets)),
            }
        )
        if batch % mini_batch_size == 0 or i == processed_train_data.size(2)-1:
            if use_deepspeed:
                model.step()
                model.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        del(data,targets)
        torch.cuda.empty_cache()

        total_acc += acc
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0 and batch != resume_batch:
            cur_loss = total_loss / log_interval
            try:
                ppl = math.exp(cur_loss)
            except:
                ppl = -1.0
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:04.5f} | ms/batch {:08.3f} | acc {:3.2f}% | '
                  'loss {:5.3f} | ppl {:10.3f}'.format(
                    epoch, batch, processed_train_data.size(2) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,total_acc*100/log_interval,
                    cur_loss,ppl ))
            total_loss = 0
            total_acc = 0
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
                'train_eval_events': train_eval_event,
                'step_number': step
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
                'train_eval_events': train_eval_event,
                'step_number': step
            })"""
        if step_scheduler != None:
            if (batch % step_scheduler == 0 and batch > 0) or (epoch >1 and batch == 0 and processed_train_data.size(2)//bptt < step_scheduler):
                scheduler.step()
                step += 1

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    total_acc = 0.
    single_pass_mem = None
    with torch.no_grad():
        for i in range(0, data_source.size(2), bptt-1-1):
            data, targets = get_batch(data_source, i)
            if use_deepspeed:
                with autocast():
                    output,single_pass_mem = eval_model(data,mem = single_pass_mem)
                    output_flat = output.view(-1, ntokens)
                    total_loss += data.size(1) * criterion(output_flat, targets).item()
                    total_acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()
            else:
                output,single_pass_mem = eval_model(data,mem = single_pass_mem)
                output_flat = output.view(-1, ntokens)
                total_loss += data.size(1) * criterion(output_flat, targets).item()
                total_acc += ((torch.argmax(output.view(-1,ntokens),dim=-1)) == targets).sum().item()
    return total_loss / (data_source.size(2)), total_acc/data_source.size(2)

while True:
    if epoch >= epochs:
        break
    if resume_batch==0:
        epoch +=1
    step=step
    epoch_start_time = time.time()
    train(resume_batch=resume_batch,mini_batch_size= 1)
    resume_batch = 0
    val_loss, val_acc = evaluate(model, processed_val_data)
    print('-' * 110)
    print('| end of epoch {:3d} | time: {:08.3f}s | valid acc {:3.2f}% | valid loss {:5.3f} | '
          'valid ppl {:10.3f}'.format(epoch, (time.time() - epoch_start_time),val_acc*100,
                                     val_loss, math.exp(val_loss)))
    print('-' * 110)

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
                'train_eval_events': train_eval_event,
                'step_number': step
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
                'train_eval_events': train_eval_event,
                'step_number': step
            })"""
model = best_model

test_loss,test_acc = evaluate(best_model, processed_test_data)
print('=' * 110)
print('| End of training | test acc {:3.2f}% | test loss {:5.3f} | test ppl {:10.3f}'.format(test_acc*100,
    test_loss, math.exp(test_loss)))
print('=' * 110)

_ = inference("Hello World!!! This is inference function on the currently trained model")

while True:
    i = int(input("Enter 2 for reccurent inference,enter 1 for static inference, 0 for exiting:"))
    if i == 0:
        break
    inp = input("input text, 1 string at a time, for inference:")
    mem = None if i==1 else mem
    mem = inference(inp,reccurent_mem=mem)