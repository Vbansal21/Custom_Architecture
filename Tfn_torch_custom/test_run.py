import deepspeed, torch

batch = 1
deepspeed_args = {
  "train_batch_size": batch,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": True,
    "loss_scale": 1,
    "initial_scale_power": 3,
    "loss_scale_window": 1000,
    "hysteresis": 1,
    "min_loss_scale": 0
    },
  "gradient_clipping":1.0,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer":{
        "device": "nvme",
        "nvme_path":"/mnt/nvme0n1p3/"
        },
    "offload_param": {
        "device": "nvme",
        "nvme_path": "/mnt/nvme0n1p3/"
        },

        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e5,
        #"reduce_bucket_size": "auto",
        #"stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": 1e8,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8
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
    "number_checkpoints":32,
    "synchronize_checkpoint_boundary": True,
    "profile": True
    }
    
}

from performer_torch.performer_pytorch import PerformerLM

with deepspeed.zero.Init(remote_device='nvme',config=deepspeed_args,enabled=True):
  model = PerformerLM(num_tokens=100,max_seq_len=8192,dim=64,depth=32,heads=8,use_rezero=True)
  """
  model = torch.nn.Sequential(
    torch.nn.Linear(2048,16384),
    torch.nn.GELU(),
    torch.nn.Linear(16384,4096),
    torch.nn.GELU(),
    torch.nn.Linear(4096,16384),
    torch.nn.GELU(),
    torch.nn.Linear(16384,1),
    torch.nn.GELU(),
  )
  """

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr= 0.001,betas=[0.8,0.99],eps=1e-8,weight_decay=3e-7)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

model, optimizer,_,scheduler = deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config_params=deepspeed_args)
from einops import rearrange
#rearrange(t, 'b n (h d) -> b h n d', h = h)
for i in range(100):
  out = model(torch.randint(0,99,(batch,8192)).to(dtype=torch.long,device=torch.device('cuda')))
  loss = criterion(rearrange(out,'b n d -> n d b'),torch.randint(0,99,(8192,batch)).to(dtype=torch.long,device=torch.device('cuda')))
  model.backward(loss)
  model.step()
  print(loss)