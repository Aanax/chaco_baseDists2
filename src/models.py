#here ill store models
# import torch

from torch import Tensor, cat
from agents import Agent
# from train import trainSTAR
import torch.nn as nn
from initializations import *
import torch.nn.functional as F
import torch as T
import numpy as np

from torch.distributions.normal import Normal
from ConvLSTM import ConvLSTM,ConvLSTMCell,ConvLSTMwithAbaseCell, ConvLSTMwithAAbaseCell

#kld, x_restored, v, a, hx, cx, s, S

class Decoder(nn.Module):
    def __init__(self, args, device):#1024
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(32, 16, 6, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 6, stride=2, padding=2)

    def forward(self,x):
        x = self.deconv1(x)
        x = self.deconv3(x)
        return x 

class Oracle(nn.Module):
    def __init__(self, args, device):#1024
        super(Oracle, self).__init__()
        self.conv = nn.Conv2d(32+32+6+32, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.layernorm = nn.LayerNorm([32,20,20]) 

    def forward(self, x, diff, previous_action, memory): #previous_g, memory
        
        previous_action = previous_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,6,20,20)).detach()
        #prev_g 1,32,20,20
        #memory 1,32,20,20
        x = torch.cat([x, diff, previous_action, memory], dim=1)  #previous_g, memory
        x = F.relu(self.conv(x))
        x = self.conv2(x)
        x = self.layernorm(x)  
        return x 

class Encoder(nn.Module):
    def __init__(self, args, device):#1024
        super(Encoder, self).__init__()
        
        self.gamma1 = float(args["Training"]["initial_gamma1"])
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.maxp11 = nn.MaxPool2d(2, 2)
        self.layernorm = nn.LayerNorm([32,20,20])   
        self.conv1.apply(init_first)
        self.conv11.apply(init_base)
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None        
    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))        
        mu = self.maxp11(self.conv11(x))
        s = self.layernorm(mu)        
        return s
    
    
class Level1(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level1, self).__init__()
        #         self.oracle = Oracle({},device)
        self.decoder = Decoder({}, device)
        self.oracle = Oracle({}, device)
        self.encoder = Encoder(args, device)
        self.actor_ext = nn.Linear(25606, 6)
        self.actor_int = nn.Linear(12800*2, 6)
#         self.critic = nn.Linear(12800*2, 1)
       
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        self.decoder.apply(init_decoder)
#         self.oracle.apply(init_base)
#         self.ConvLSTM_mu.apply(init_base)
        self.actor_ext.weight.data = norm_col_init(
        self.actor_ext.weight.data, args["Model"]["a_init_std"])
        self.actor_ext.bias.data.fill_(0)
        
        self.actor_int.weight.data = norm_col_init(
        self.actor_int.weight.data, args["Model"]["a_init_std"])
        self.actor_int.bias.data.fill_(0)
#         self.critic.weight.data = norm_col_init(
#         self.critic.weight.data, args["Model"]["v_init_std"])
#         self.critic.bias.data.fill_(0) 
        self.train()
        self.z_EMA_t = 0

    def forward(self, x, previous_action, memory, previous_s):
#          = x
        #previous_g, memory
        s = self.encoder(x)
       
        decoded = self.decoder(s)
        
        g=0
#         g = self.oracle(s.detach(), s.detach()-previous_s.detach(), previous_action.detach(), memory.detach()) #previous_g.detach(), memory.detach()
                        
        z = torch.cat([(s.detach()-previous_s.detach()).view(s.size(0), -1),s.view(s.size(0), -1), previous_action.detach().view(previous_action.size(0), -1)], dim=1)
        z = z.view(z.size(0), -1)
        
        Q11_ext = self.actor_ext(z)
#         Q11_int = Q11_ext
#         Q11_int = self.actor_int(z, s.detach()-previous_s.detach(), previous_action.detach())
        Q11_int =0
        
        ps = torch.nn.functional.softmax(Q11_ext.detach())#+Q11_int.detach())
        v_ext =(ps*Q11_ext).sum()
        v_int =0#(ps*Q11_int).sum()
         
#         print("DECODEd shape ", decoded.shape, flush=True)
        return decoded, v_ext, v_int, Q11_ext,Q11_int, s, g #, hx, cx, s,S  