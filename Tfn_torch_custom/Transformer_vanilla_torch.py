import math
import io, os
import random,wandb
from torch import nn
from torch import Tensor
from einops import rearrange
import math, time, torch #,copy
from typing import List
#from performer_torch import PerformerLM
#from pytorch_model_summary import summary
from inputimeout import inputimeout as inpt
from torchnlp.encoders.text import SubwordEncoder
from torchtext.utils import download_from_url, extract_archive
#from typing import Tuple, Optional, Any, NoReturn, Union, Literal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)

#scaler = torch.cuda.amp.GradScaler(init_scale=2**3)
autocast = torch.cuda.amp.autocast


file = "wikitextv103"
if file == "wikitextv2":
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
elif file == "wikitextv103":
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))

files = []
string_of_files = ""

try:
    retrieve_tokenizer = eval(inpt(prompt="retrieve tokenizer?(default: True):",timeout=15))
    if type(retrieve_tokenizer) != int or type(retrieve_tokenizer) != bool:
        retrieve_tokenizer = True
except:
    retrieve_tokenizer = True

def list_of_all_files(path:str="./") -> str:
    try:
        super_dirs = os.listdir(path)
    except:
        return [path[:-1]]
    dirs = [path+i+"/" for i in super_dirs]
    files = []
    if len(dirs) > 0:
        for i in dirs:
            files += list_of_all_files(i)
    return files

def file_to_str(file_name_with_path:str,files_not_to_be_included: List[str] = [".tar",".zip",".pt",".pth",".onnx"]) -> str:
    for i in files_not_to_be_included:
        if i in file_name_with_path:
            return ""
    f = open(file_name_with_path)
    return str(f)

def initialize_tokenizer(target_vacab = 2**15):
    global files, string_of_files
    sample = "the quick brown fox jumps over the lazy dog.THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG?!@#$%^&*()`~-_+=[{]}\\|\"':;/.>,<1234567890\t\n\f\r\v "
    sample += " ".join([i for i in sample])
    tmp = "".join([i for i in io.open(train_filepath, encoding="utf8")]) + "".join([i for i in io.open(test_filepath, encoding="utf8")]) + "".join([i for i in io.open(valid_filepath, encoding="utf8")])
    sample += " ".join([i for i in tmp]) + tmp
    path = "../"
    files += list_of_all_files(path)
    for i in files:
        string_of_files += file_to_str(i)
    sample += " ".join([i for i in string_of_files])
    sample = "".join(list(set([i for i in sample])))
    print("parsed all files")
    tokenizer = SubwordEncoder(sample,target_vocab_size=target_vacab,reserved_tokens=[
    '[pad]','[unk]','[sos]','[eos]','[copy]','[mask]','[segment_seperator]','[non_text_content]','[/non_text_content]'
    ],
    eos_index=3,unknown_index=1,padding_index=0)
    vocab_size = tokenizer.vocab_size
    torch.save(tokenizer,"models/tokenizer_"+str(vocab_size)+".tar")
    return tokenizer,vocab_size

if retrieve_tokenizer:
    files = os.listdir("models/")
    tokenizer_files = []
    for i in files:
        if "tokenizer" in i:
            tokenizer_files += [i]
    tokenizer_name = tokenizer_files[0]
    for i in tokenizer_files:
        if int(i[10:-4]) < int(tokenizer_name[10:-4]):
            tokenizer_name = i
    print([[i,j] for i,j in enumerate(tokenizer_files)])
    try:
        inp = int(inpt(prompt="index of file to be used(starting from 0):",timeout=15))   
    except:
        inp = None
    if inp != None: 
        if inp < len(tokenizer_files):
            tokenizer_name = tokenizer_files[inp]
    tokenizer = torch.load("models/"+str(tokenizer_name))
    vocab_size = tokenizer.vocab_size
else:
    try:
        inp = int(inpt(prompt="target vocabulary size (default=2**15):",timeout=15))   
        if type(inp) != int:
            inp = 2**15
            print("invalid input")
    except:
        inp = 2**15
    tokenizer, vocab_size = initialize_tokenizer(inp)
vocab = tokenizer.vocab

def batchify(data, bsz,dim=0):
    if data.size(0) == 2 and len(data.size())==3:
        data = data[0]
    nbatch = data.size(dim) // bsz
    data = data.narrow(dim, 0, nbatch * bsz)
    data = data.reshape(bsz, -1).contiguous()
    return data

batch_size = 1
eval_batch_size = batch_size
mini_batch_size = 1

ntokens = tokenizer.vocab_size # None
emsize = 512
nhid = emsize * 4
nlayers = 1
deberta_layers = 3
repeated_deberta_layers = 0
full_block_repeat = False
nhead = 8
dropout = (math.pi/10)
mem_tokens = emsize*4
bptt = (1024*8) #- mem_tokens
seq_scale_down = max(2**(int(math.log(2,math.log(2,emsize)))),8)
max_seq_len = max(2**14,2**17 // seq_scale_down)
mlp_layers = 1
fno_layers = 4
modes = 8
width = 8
causal = False
nystrom = False
attend_to_self = True
attend_to_inp = True
feature_redraw_interval = nhead**2
prev_state_len = emsize*4
prev_state_self_num = 128
local_heads = 2
local_heads = min(local_heads,nhead)

discriminator = False #INTEGRATED DISCRIMINATOR: DEPRECATED
progressive_generation = True
use_deepspeed = False

use_sgd = True

def data_process(raw_text_iter):
  data = tokenizer.encode(raw_text_iter)
  return data.contiguous()

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
    out = inp_2.clone().detach().to(dtype=torch.long).contiguous()
    del(inp_2,inp)
    torch.cuda.empty_cache()
    return out,index_to_be_trained_on

def get_batch(source,j,bptt=bptt,progressive=True,shuffle=True):
    rnd_shuffle = 0 if not shuffle else random.randint(0,100000000)/1000000
    rnd_mask = random.randint(0,7000000000)/100000000
    rnd_mask_together = random.randint(0,seq_scale_down**2)
    rnd_1 = random.randint(0,24)
    rnd_2 = random.randint(0,12)
    rnd_3 = random.randint(0,12)
    if progressive:
        seq_len = min(bptt, source.size(1) - j) -3
        data,index_to_be_trained_on = random_mask_shuffle_encoder(source[:,j:j+seq_len-1],mask_percentage=rnd_mask,mask_together_nos=rnd_mask_together,mask_continuous_pos=170,shuffle_percentage=rnd_shuffle,shuffle_together_nos=seq_scale_down)
        data = torch.cat((torch.full((data.size(0),rnd_1),2,dtype=torch.long,device=device),data.to(device),torch.full((data.size(0),rnd_2+1-min(1,rnd_1)),5,dtype=torch.long,device=device),torch.full((data.size(0),rnd_3),3,dtype=torch.long,device=device)),dim=1).contiguous()
        targets = source[:,j:j+seq_len].to(device)
        targets = torch.cat((torch.full((data.size(0),max(rnd_1-1,0)),2,dtype=torch.long,device=device),targets,torch.full((targets.size(0),rnd_2+rnd_3),3,dtype=torch.long,device=device)),dim=1).contiguous()
    else:
        seq_len = min(bptt, source.size(1) - j) -3
        data,index_to_be_trained_on = random_mask_shuffle_encoder(source[:,j:j+seq_len],mask_percentage=rnd_mask,mask_together_nos=rnd_mask_together,mask_continuous_pos=170,shuffle_percentage=rnd_shuffle,shuffle_together_nos=seq_scale_down)
        data = torch.cat((torch.full((data.size(0),rnd_1),2,dtype=torch.long,device=device),data.to(device),torch.full((data.size(0),rnd_2),5,dtype=torch.long,device=device),torch.full((data.size(0),rnd_3),3,dtype=torch.long,device=device)),dim=1).contiguous()
        targets = source[:,j:j+seq_len].to(device)
        targets = torch.cat((torch.full((targets.size(0),rnd_1),2,dtype=torch.long,device=device),targets,torch.full((targets.size(0),rnd_2+rnd_3),3,dtype=torch.long,device=device)),dim=1).contiguous()
    torch.cuda.empty_cache()
    return data,targets,index_to_be_trained_on

try:
    processed_train_data = torch.load("models/data_"+str(vocab_size)+"/"+file+"_train.tar",map_location=torch.device('cpu'))
    processed_test_data = torch.load("models/data_"+str(vocab_size)+"/"+file+"_test.tar",map_location=torch.device('cpu'))
    processed_val_data = torch.load("models/data_"+str(vocab_size)+"/"+file+"_val.tar",map_location=torch.device('cpu'))

    processed_train_data = batchify(processed_train_data,batch_size,-1)
    processed_test_data = batchify(processed_test_data,eval_batch_size,-1)
    processed_val_data = batchify(processed_val_data,eval_batch_size,-1)
except Exception as e:
    print(e)
    train_portion = int(len(string_of_files) * 0.6)
    test_portion = int(len(string_of_files) * 0.2)

    train_sample = "".join([i for i in io.open(train_filepath, encoding="utf8")]) + string_of_files[:train_portion]
    test_sample = "".join([i for i in io.open(test_filepath, encoding="utf8")]) + string_of_files[train_portion:train_portion+test_portion]
    val_sample = "".join([i for i in io.open(valid_filepath, encoding="utf8")]) + string_of_files[train_portion+test_portion:]
    
    train_data = data_process(train_sample)
    val_data = data_process(val_sample)
    test_data = data_process(test_sample)

    processed_train_data = batchify(train_data, batch_size)
    processed_val_data = batchify(val_data, eval_batch_size)
    processed_test_data = batchify(test_data, eval_batch_size)

    del(train_data,test_data,val_data)

    if not os.path.exists("models/data_"+str(vocab_size)+"/"):
        os.mkdir("models/data_"+str(vocab_size)+"/")

    torch.save(processed_train_data,"models/data_"+str(vocab_size)+"/"+file+"_train.tar")
    torch.save(processed_test_data,"models/data_"+str(vocab_size)+"/"+file+"_test.tar")
    torch.save(processed_val_data,"models/data_"+str(vocab_size)+"/"+file+"_val.tar")

from scripts.model import TransformerModel, Trainer
torch.cuda.empty_cache()

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
        model = TransformerModel( 
                                ninp=emsize, 
                                nhead=nhead, 
                                nhid=nhid, 
                                nlayers=nlayers,
                                ntoken=ntokens,
                                dropout=dropout,
                                deberta_layers=deberta_layers,
                                repeated_deberta_layers=repeated_deberta_layers,
                                mem_token=mem_tokens,
                                #discriminator=discriminator,
                                seq_scale_down=seq_scale_down,
                                max_seq_len=max_seq_len,
                                full_block_repeat=full_block_repeat,
                                causal=causal,
                                nystrom=nystrom,
                                attend_to_self=attend_to_self,
                                fno_layers=fno_layers,
                                modes=modes,
                                width=width,
                                feature_redraw_interval=feature_redraw_interval,
                                prev_state_len=prev_state_len,
                                prev_state_self_num=prev_state_self_num,
                                local_heads=local_heads,
                                attend_to_inp=attend_to_inp,
                                mlp_layers=mlp_layers,
                        ).half()
else:
    model = TransformerModel(
                            ninp=emsize, 
                            nhead=nhead, 
                            nhid=nhid, 
                            nlayers=nlayers, 
                            ntoken=ntokens, 
                            dropout=dropout,
                            mem_token=mem_tokens,
                            deberta_layers=deberta_layers,
                            repeated_deberta_layers=repeated_deberta_layers,
                            max_seq_len=max_seq_len,
                            #discriminator=discriminator,
                            seq_scale_down=seq_scale_down,
                            full_block_repeat=full_block_repeat,
                            causal=causal,
                            device=device,
                            nystrom=nystrom,
                            attend_to_self=attend_to_self,
                            fno_layers=fno_layers,
                            modes=modes,
                            width=width,
                            feature_redraw_interval=feature_redraw_interval,
                            prev_state_len=prev_state_len,
                            prev_state_self_num=prev_state_self_num,
                            local_heads=local_heads,
                            attend_to_inp=attend_to_inp,
                            mlp_layers=mlp_layers,
                    ).to(device)

print("Model Parameters: ",len(model),"\n")
torch.cuda.empty_cache()

model.eval()
inp = torch.randint(0,ntokens-1,(batch_size,bptt),dtype=torch.long,device=device)
#model.toggle_vanilla_attn_mechanism(True,True)
if use_deepspeed:
    with autocast():
        out,mem,mem_ctxt = model(inp)
else:
    out,mem,mem_ctxt = model(inp)
print(torch.argmax((out.reshape(-1,ntokens)),dim=-1))
print(model.get_avg_inference_time()," seconds")
del(out,mem,mem_ctxt,inp)

#print(sum(p.numel() for p in model.parameters()))
date_time = str(time.asctime().replace(" ","_")).replace(":","_")
path = "models"+"/model_"+str(emsize)+"_"+str(nlayers)+"_"+str(deberta_layers)+"_"+str(repeated_deberta_layers)+"_"+str(nhead)+"_"+str(seq_scale_down)+".tar"

criterion = nn.CrossEntropyLoss()
lr = 1

if not use_deepspeed:
    if use_sgd:
        if discriminator:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            optimizer_disc = torch.optim.SGD(discriminator_model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr=lr)
            optimizer_disc = None
    else:
        if discriminator:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
            optimizer_disc = torch.optim.Adadelta(discriminator_model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adadelta(model.parameters(),lr=lr)
            optimizer_disc = None

else:
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.8,0.999),weight_decay=3e-7,eps=1e-8)

step = 1
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
        return sub_func(step_) / (25 * (lr**0.5))
    else:
        return sub_func(step_) / (125 * lr)
#    pseudo_lambda = lambda step: (((a/b * (multiplier*step) + 1) / ((multiplier*step)**2 + a)) + c)/((step*(multiplier/200))**0.1+1)
#    lambda_1 = lambda step: (pseudo_lambda(step) if step<(1024/(multiplier**(math.pi*2/10))) else (pseudo_lambda(step)/25 if step<(2048/(multiplier**(math.pi*2/10))) else pseudo_lambda(step)/625))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda_lr)
scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_disc,lr_lambda=lambda_lr) if discriminator else None

model.tokenzier = tokenizer
model.vocab = vocab
model.optimizer = optimizer
#model.optimizer_disc = optimizer_disc
model.scheduler = scheduler
#model.scheduler_disc = scheduler_disc
model.scheduler_lambda = lambda_lr
#model.scheduler_disc_lambda = lambda_lr

load_optimizer = True
load_scheduler = bool(True and load_optimizer)
load_step_number = True
load_tokenizer = True
epoch = 0
best_val_loss = float("inf")

resume_batch = 0
log_interval = feature_redraw_interval
epochs = 4

import matplotlib.pyplot as plt
plt.plot([lambda_lr(i) for i in range( int((processed_train_data.size(1)*epochs) / (bptt*batch_size)) + 1)])
plt.show()
del(plt)

train_eval_event = [date_time]

if use_deepspeed:
    model,optimizer,_,scheduler = deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler, config_params=deepspeed_args)

best_model = model

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
    "seq_scale_down":seq_scale_down,
    "discriminator":discriminator,
    "Number of Parameters":len(model),
    "Progressive generation training":progressive_generation,
    "use_sgd":use_sgd,
    "full_block_repeat":full_block_repeat,
    "causal":causal,
    "nystromer":nystrom,
    "attend_to_self":attend_to_self,
    "fno_layers":fno_layers,
    "modes":modes,
    "width":width,
    "feature_redraw_intervel":feature_redraw_interval,
    "prev_state_len":prev_state_len,
    "local_heads":local_heads,
    "attend_to_inp":attend_to_inp,
    "prev_state_self_num":prev_state_self_num,
    "mlp_layers":mlp_layers,
}
)

#wandb.watch(model,criterion=criterion,log_freq=20)


try:
    try:
        checkpoint_ = torch.load(path, map_location=device)
    except:
        _,checkpoint_ = model.load_checkpoint(path,)

    epoch = checkpoint_['epoch']
    best_val_loss = checkpoint_['best_val_loss']
    if load_tokenizer:
        vocab = checkpoint_['vocab']
        tokenizer = checkpoint_['tokenizer']
    
    try:
        model.load_state_dict(checkpoint_['model_state_dict'],strict=False)
    except:
        try:
            model = checkpoint_['model']
        except Exception as e:
            print("Exception",e)
            
    if load_optimizer:
        optimizer.load_state_dict(checkpoint_['optimizer_state_dict'])
        if discriminator:
            optimizer_disc.load_state_dict(checkpoint_['optimizer_disc_state_dict'])
    else:
        if discriminator:
            for p in model.discriminator.parameters():
                p.requires_grad_(False)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.discriminator.parameters():
                p.requires_grad_(True)
            optimizer_disc = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            for p in model.parameters():
                p.requires_grad_(True)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.SGD(model.parameters(),lr=lr)

            

    step = checkpoint_['step_number'] if load_step_number else step

    if load_scheduler:
        scheduler.load_state_dict(checkpoint_['scheduler_state_dict'])
        if discriminator:
            scheduler_disc.load_state_dict(checkpoint_['scheduler_disc_state_dict'])

    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda_lr)
        if discriminator:
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_disc,lr_lambda=lambda_lr)

    try:
        resume_batch = checkpoint_['resume_batch']
        train_eval_event = checkpoint_['train_eval_events'] + [date_time]
    except Exception as e:
        print("Exception",e)
        
    try:
        best_model.load_state_dict(checkpoint_['best_model_state_dict'],strict=False)
        best_model = best_model.to(torch.device('cpu'))
    except Exception as e:
        try:
            best_model = checkpoint_['best_model'].to(torch.device('cpu'))
        except Exception as f:
            print("Exception",e,f)
    del(checkpoint_)
    torch.cuda.empty_cache()
except Exception as e:
    print("Exception",e)
    pass

if best_model==None:
    best_model=model

model.to(device)

#inp = torch.zeros([1,bptt],dtype=torch.long).to(device)
#print(summary(model, inp,None,None,None,None,False,False,True,discriminator))

# TODO: Setup 'curses' module to print colored text for inference output
#import curses
def inference(text,size=128,eval_model = best_model,reccurent_mem=None,reccurent_mem_ctxt=None,return_mem=True):
    model.eval()
    torch.cuda.empty_cache()
    text_input = torch.cat((torch.full(tuple([1,1]),2),data_process(text).unsqueeze(0),torch.full(tuple([1,size]),5),torch.full(tuple([1,1]),3)),dim=1).to(device )
    if use_deepspeed:
        with autocast():
            out,mem,mem_ctxt = eval_model(text_input,mem=reccurent_mem,context_mem=reccurent_mem_ctxt)
    else:
        out,mem,mem_ctxt = eval_model(text_input,mem=reccurent_mem,context_mem=reccurent_mem_ctxt)
    out = torch.argmax(out.reshape(-1, ntokens),dim=-1).to(torch.device('cpu'))
    result = tokenizer.decode(out)
    print("Your input:\v",tokenizer.decode(text_input.reshape(-1).to(torch.device('cpu'))))
    print("Model's Output:\v",result)
    print('')
    torch.cuda.empty_cache()
    if return_mem:
        return mem,mem_ctxt


inference("Hello World!!! This is inference function on the currently trained model",return_mem=False)

def evaluate(eval_model, data_source, print_val_loss=False):
    eval_model.eval()
    total_loss = 0.
    total_acc = 0.
    single_pass_mem = None
    single_pass_mem_ctxt = None
    stride_size = bptt-3 if progressive_generation else bptt -3
    with torch.no_grad():
        for i in range(0, data_source.size(1), stride_size):
            torch.cuda.empty_cache()
            data, targets, trainable_index = get_batch(data_source, i)
            if use_deepspeed:
                with autocast():
                    output,single_pass_mem,single_pass_mem_ctxt = eval_model(data,mem = single_pass_mem,context_mem=single_pass_mem_ctxt)
                    total_loss += data.size(1) * criterion(rearrange(output,'b n c -> n c b'), rearrange(targets,'b n -> n b')).item()
                    total_acc += ((torch.argmax(output,dim=-1)) == targets).sum().item()
            else:
                output,single_pass_mem,single_pass_mem_ctxt = eval_model(data,mem = single_pass_mem,context_mem=single_pass_mem_ctxt)
                total_loss += data.size(1) * criterion(rearrange(output,'b n c -> n c b'), rearrange(targets,'b n -> n b')).item()
                total_acc += ((torch.argmax(output,dim=-1)) == targets).sum().item()
    val_loss = total_loss / (data_source.size(1))
    val_acc = total_acc/data_source.size(1)
    if print_val_loss:
        print('-' * 110)
        print('valid acc {:3.2f}% | valid loss {:5.3f} | valid ppl {:10.3f}'.format(val_acc*100,val_loss, math.exp(val_loss)))
        print('-' * 110)
    return val_loss, val_acc

torch.cuda.empty_cache()
def train(resume_batch=0,step_scheduler=1,save_intermediate_intervel=8192,save_intermediate_intervel_time_s=900,optimizer=optimizer,optimizer_disc=optimizer_disc):
    
    global step
    global log_interval
    total_loss = 0.
    total_loss_d = 0.
    total_ppl = 0.
    total_time_per_step = 0.
    start_time = time.time()
    intermediate_save_time = time.time()
    single_pass_mem = None
    single_pass_mem_ctxt = None
    acc = 0
    acc_d = 0
    total_acc = 0
    total_acc_d = 0
    stride_size = bptt-3 if progressive_generation else bptt -3
    for batch, i in enumerate(range(0, processed_train_data.size(1), stride_size)):
        model.train()
        step_time = time.time()
        if resume_batch != None:
            if batch < resume_batch:
                continue
        if ((batch + epoch)%2==1):
            single_pass_mem = None
            single_pass_mem_ctxt = None
        data, targets, trainable_index = get_batch(processed_train_data, i,progressive=progressive_generation) #Indexed Selective training broken
        trainable_index = None
        torch.cuda.empty_cache()

        if not discriminator:
            outputs,losses,loss,acc,time_,single_pass_mem,single_pass_mem_ctxt = model.training_step(data,targets,criterion,single_pass_mem,opt=optimizer,trainable_index=trainable_index,mem_ctxt=single_pass_mem_ctxt,mini_batch_size=mini_batch_size,batch=batch)
        else:
            pass

        total_loss += loss
        total_acc += acc

        tmp_acc = total_acc
        tmp_loss = total_loss

        try:
            ppl = math.exp(losses['loss'])
        except:
            ppl = -1.0

        log_interval = log_interval
        total_ppl += ppl
        inputs = "\n".join([tokenizer.decode(i.to(torch.device('cpu'))) for i in data])
        output = "\n".join([tokenizer.decode(torch.argmax(i,dim=-1).to(torch.device('cpu'))) for i in outputs['output']])
        req_targets = "\n".join([tokenizer.decode(i.to(torch.device('cpu'))) for i in targets])
        del(data,targets,outputs,losses)
        torch.cuda.empty_cache()

        if (batch % save_intermediate_intervel == 0 and batch > 0) or (time.time()-intermediate_save_time) > save_intermediate_intervel_time_s:
            inference("Hello World!!! This is inference function on the currently trained model",return_mem=False)

            model.eval()
            best_model.eval()

            model.tokenzier = tokenizer
            model.vocab = vocab
            model.optimizer = optimizer
            #model.optimizer_disc = optimizer_disc
            model.scheduler = scheduler
            #model.scheduler_disc = scheduler_disc
            model.scheduler_lambda = lambda_lr
            #model.scheduler_disc_lambda = lambda_lr

            if discriminator:
                torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model':model,
                    'best_model':best_model,
                    'best_model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scheduler_disc_state_dict':scheduler_disc.state_dict(),
                    'best_val_loss': best_val_loss,
                    'vocab': vocab,
                    'tokenizer': tokenizer,
                    'resume_batch':batch,
                    'train_eval_events': train_eval_event,
                    'step_number': step
                },
                path
                )
            else:
                torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model':model,
                    'best_model':best_model,
                    'best_model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_disc_state_dict': None,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scheduler_disc_state_dict':None,
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
            })
            """
            intermediate_save_time = time.time()
            model.train()
        if step_scheduler != None and batch%mini_batch_size == 0:
            if (batch % step_scheduler == 0 and batch > 0) or (epoch >1 and batch == 0 and processed_train_data.size(1)//bptt < step_scheduler):
                scheduler.step(step)
                if discriminator:
                    scheduler_disc.step(step)
                step += 1
        if (batch % log_interval == 0 and batch != resume_batch):
            cur_loss = total_loss / log_interval
            cur_loss_d = total_loss_d / log_interval
            total_ppl /= log_interval
            _,__ = evaluate(model,processed_val_data,True)
            elapsed = time.time() - start_time
            if discriminator:
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr_g {:04.5f} | lr_d {:04.5f} | ms/batch {:08.3f} | acc_g {:3.2f}% | '
                    'loss_g {:5.3f} | acc_d {:3.2f}% | loss_d {:5.3f} | ppl {:10.3f}'.format(
                        epoch, batch, processed_train_data.size(1) // bptt, scheduler.get_last_lr()[0],
                        scheduler_disc.get_last_lr()[0],
                        elapsed * 1000 / log_interval,total_acc*100/log_interval,
                        cur_loss,total_acc_d*100/log_interval,cur_loss_d,total_ppl ))
            else:
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:04.5f} | ms/batch {:08.3f} | acc {:3.2f}% | '
                    'loss {:5.3f} | ppl {:10.3f}'.format(
                        epoch, batch, processed_train_data.size(1) // bptt, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,total_acc*100/log_interval,
                        cur_loss,total_ppl ))
            total_loss = 0.
            total_acc = 0.
            total_loss_d = 0.
            total_acc_d = 0.
            total_ppl = 0.
            start_time = time.time()
        total_time_per_step += (time.time() - step_time)
        if batch%mini_batch_size == 0:
            if discriminator:
                wandb.log(
                    {
                        "Loss Generator":loss_g,
                        "Total Loss Generator":tmp_loss,
                        "Loss Discriminator":loss_d,
                        "step":step,
                        "Accuracy Generator(%)":acc*100/2,
                        "Total Accuracy Generator(%)":tmp_acc*100/2,
                        "Accuracy Discriminator(%)":acc_d*100/2,
                        "epoch":epoch,
                        "batch":batch,
                        "Perplexity of Generator":ppl,
                        'Learning_Rate':scheduler.get_last_lr()[0],
                        'Time per Step':total_time_per_step/mini_batch_size,
                        "input":wandb.Html(inputs),
                        "output":wandb.Html(output),
                        "target":wandb.Html(req_targets),
                        "avg_inference_time":model.get_avg_inference_time(),
                    }
                )
            else:
                wandb.log(
                    {
                        "Loss Generator":loss,
                        "step":step,
                        "Accuracy Generator(%)":acc*100/2,
                        "epoch":epoch,
                        "batch":batch,
                        "Perplexity of Generator":ppl,
                        'Learning_Rate':scheduler.get_last_lr()[0],
                        'Time per Step':total_time_per_step/mini_batch_size,
                        "input":wandb.Html(inputs),
                        "output":wandb.Html(output),
                        "target":wandb.Html(req_targets),
                        "avg_inference_time":model.get_avg_inference_time()
                    },
                    
                )
            total_time_per_step = 0

while True:
    if epoch >= epochs:
        break
    if resume_batch==0:
        epoch +=1
    step=step
    epoch_start_time = time.time()
    train(resume_batch=resume_batch)
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

    model.eval()
    best_model.eval()

    model.tokenzier = tokenizer
    model.vocab = vocab
    model.optimizer = optimizer
    #model.optimizer_disc = optimizer_disc
    model.scheduler = scheduler
    #model.scheduler_disc = scheduler_disc
    model.scheduler_lambda = lambda_lr
    #model.scheduler_disc_lambda = lambda_lr

    if discriminator:
        torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model':model,
            'best_model':best_model,
            'best_model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_disc_state_dict': optimizer_disc.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_disc_state_dict':scheduler_disc.state_dict(),
            'best_val_loss': best_val_loss,
            'vocab': vocab,
            'tokenizer': tokenizer,
            'resume_batch':0,
            'train_eval_events': train_eval_event,
            'step_number': step
        },
        path
        )
    else:
        torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_disc_state_dict': None,
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_disc_state_dict':None,
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

inference("Hello World!!! This is inference function on the currently trained model",return_mem=False)
mem = mem_ctxt = None
while True:
    i = int(input("Enter 2 for reccurent inference,enter 1 for static inference, 0 for exiting:"))
    if i == 0:
        break
    inp = input("input text, 1 string at a time, for inference:")
    mem = None if i==1 else mem
    mem_ctxt = None if i==1 else mem_ctxt
    mem, mem_ctxt = inference(inp,reccurent_mem=mem,reccurent_mem_ctxt=mem_ctxt)