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
#         print("DECODER 1 x shape3 ", x.shape, flush=True)

        return x 

class Oracle(nn.Module):
    def __init__(self, args, device):#1024
        super(Oracle, self).__init__()
        # 102
        self.conv = nn.Conv2d(32+32+32+6, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 5, stride=1, padding=2)

    def forward(self, x, previous_action, previous_g, memory):
#         print("previous_action2.shape ",previous_action2.shape)
        previous_action = previous_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,6,20,20)).detach()
        #prev_g 1,32,20,20
        #memory 1,32,20,20

        x = torch.cat([x, previous_action, previous_g, memory], dim=1)
        
        x = F.relu(self.conv(x))
        x = self.conv2(x)
        
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
#         self.conv11_logvar.apply(init_base)
        
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        
        
    def forward(self, x):
        
        x = F.relu(self.maxp1(self.conv1(x)))
        
#         x = T.cat((x.view(x.size(0), -1), previous_action),1)
        
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
        self.actor = nn.Linear(12800*2, 6)
#         self.critic = nn.Linear(12800*2, 1)
       
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        self.decoder.apply(init_decoder)
#         self.oracle.apply(init_base)
#         self.ConvLSTM_mu.apply(init_base)
        self.actor.weight.data = norm_col_init(
        self.actor.weight.data, args["Model"]["a_init_std"])
        self.actor.bias.data.fill_(0)
#         self.critic.weight.data = norm_col_init(
#         self.critic.weight.data, args["Model"]["v_init_std"])
#         self.critic.bias.data.fill_(0) 
        self.train()
        self.z_EMA_t = 0

    def forward(self, x, previous_action, previous_g, memory):
#          = x
        
        s = self.encoder(x)
       
        decoded = self.decoder(s)
                
        g = self.oracle(s.detach(), previous_action.detach(), previous_g.detach(), memory.detach())
                        
        z = torch.cat([g.detach(),s], dim=1)
        z = z.view(z.size(0), -1)
        
        Q11 = self.actor(z)
        
        ps = torch.nn.functional.softmax(Q11)
        v =(ps*Q11).sum()
         
#         print("DECODEd shape ", decoded.shape, flush=True)
        return decoded,v,Q11, s, g #, hx, cx, s,S  