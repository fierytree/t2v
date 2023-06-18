import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from models.better.layerspp import ResnetBlockBigGANppGN
import yaml
from main import parse_args_and_config


args, config, config_uncond = parse_args_and_config()
act = get_act(config)
net=ResnetBlockBigGANppGN()
bs,ch,h=config.training.batch_size,config.model.ngf,all_resolutions[1]
x = torch.randn(bs,ch,h,h).to(config.device)
temb = torch.randn(1,temb_dim).to(config.device)
timings2=np.zeros((N,1))
for i in range(10):
    _=net(x,temb)
torch.cuda.synchronize()
for i in range(N):
    starter.record()
    _=net(x,temb)
    ender.record()
    torch.cuda.synchronize()
    timings2[i] = starter.elapsed_time(ender)