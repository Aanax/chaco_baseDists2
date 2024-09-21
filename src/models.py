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
        
#         previous_action = previous_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,6,20,20)).detach()
#         x = torch.cat([x,previous_action], dim=1)

#         print("DECODER 1 x shape in ", x.shape, flush=True)
        
        x = self.deconv1(x)
        
#         print("DECODER 1 x shape2 ", x.shape, flush=True)
        x = self.deconv3(x)
#         print("DECODER 1 x shape3 ", x.shape, flush=True)

        return x 

class Oracle(nn.Module):
    def __init__(self, args, device):#1024
        super(Oracle, self).__init__()
        # 102
        self.conv = nn.Conv2d(32+6+32+32, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 5, stride=1, padding=2)

    def forward(self, x, previous_action, previous_g, memory):
        
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
        
#         self.conv11_logvar = nn.Conv2d(16, 32, 5, stride=1, padding=2)
#         self.maxp11_logvar = nn.MaxPool2d(2, 2)
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
#         logvar = self.maxp11_logvar(self.conv11_logvar(x))

#         z_t = self.N.sample(mu.shape)
#         s = mu + T.exp(logvar / 2) * z_t #self.z_EMA_t
                
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
        
        v = Q11.mean() #self.critic(z)
        
#         print("DECODEd shape ", decoded.shape, flush=True)
        return decoded,v,Q11, s, g #, hx, cx, s,S  

class Decoder2(nn.Module):
    def __init__(self, args, device):#1024
        super(Decoder2, self).__init__()
        
#         self.deconv1 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.deconv1 = nn.ConvTranspose2d(64, 48, 6, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(48, 32, 6, stride=2, padding=2)
    def forward(self,x):
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

class Oracle2(nn.Module):
    def __init__(self, args, device):#1024
        super(Oracle2, self).__init__()
        
        self.conv = nn.Conv2d(64+64+8+64, 128, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(128, 64, 5, stride=1, padding=2)

    def forward(self,x, previous_action, previous_g, memory):
        
        previous_action = previous_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,8,5,5)).detach()
        #prev_g 1,32,5,5
        #memory 1,32,5,5

        x = torch.cat([x, previous_action, previous_g, memory], dim=1)
        
        x = F.relu(self.conv(x))
        x = self.conv2(x)
        
        return x
    
class Encoder2(nn.Module):
    def __init__(self, args, device):#1024
        super(Encoder2, self).__init__()
        
        self.gamma1 = float(args["Training"]["initial_gamma1"])
        self.gamma2 = float(args["Training"]["initial_gamma2"])
        self.layernorm = nn.LayerNorm([64,5,5])        
#         #20 - 10 - 5
        self.conv1 = nn.Conv2d(32, 48, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(48, 64, 5, stride=1, padding=2)
        self.maxp2 = nn.MaxPool2d(2, 2)
        
#         self.conv2_logvar = nn.Conv2d(64, 64, 5, stride=1, padding=2)
#         self.maxp2_logvar = nn.MaxPool2d(2, 2)
        
        self.conv1.apply(init_first)
        self.conv2.apply(init_base)
#         self.conv2_logvar.apply(init_base)
        
        
    def forward(self, x):
        
#         x = self.Layernorm(x)
        x = F.relu(self.maxp1(self.conv1(x)))
        mu = self.maxp2(self.conv2(x))
#         logvar = self.maxp2_logvar(self.conv2_logvar(x))
        
#         z_t = self.N.sample(mu.shape)
#         s = mu + T.exp(logvar / 2) * z_t #self.z_EMA_t
                
        s = self.layernorm(mu)
        
        return s
    
    
class Level2(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level2, self).__init__()
        
        self.encoder2 = Encoder2(args, device)
        self.oracle2 = Oracle2({},device)
        self.decoder2 = Decoder2({}, device)
        self.actor2 = nn.Linear(64*5*5*2, 8) #Actor2(args,device)
        self.actor_base2 = nn.Linear(8, 6)
#         self.critic2 = nn.Linear(64*5*5*2, 1)
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        self.decoder2.apply(init_decoder) #CHECK IF WORKS
        self.actor2.weight.data = norm_col_init(
        self.actor2.weight.data, args["Model"]["a_init_std"])
        self.actor2.bias.data.fill_(0)
        
        print("max(self.actor_base2.weight.data.shape) ", max(self.actor_base2.weight.data.shape))
        expDist = torch.distributions.Exponential(np.sqrt(32))
        self.actor_base2.weight.data = expDist.rsample(self.actor_base2.weight.data.shape)#norm_col_init(
#         self.actor_base2.weight.data, args["Model"]["a_init_std"])
        self.actor_base2.bias.data.fill_(0)
#         self.critic2.weight.data = norm_col_init(
#         self.critic2.weight.data, args["Model"]["v_init_std"])
#         self.critic2.bias.data.fill_(0) 
        self.train()
        self.s_mean=0
        self.smean_not_set = True
        self.z_EMA_t = 0
        
    def forward(self, x, previous_action, previous_g, memory): 
        s = self.encoder2(x)
        decoded = self.decoder2(s)
        g = self.oracle2(s, previous_action, previous_g, memory)
        z = torch.cat([g.detach(),s], dim=1)
        z = z.view(z.size(0), -1)
        Q_22 = self.actor2(z)
        v2 = Q_22.mean() #self.critic2(z)
        V_wave = v2 #(pi*Q_22).sum()
        a_22 = Q_22
#         a_22 =  #((Q_22-V_wave.detach())>=0).float()
#         smax = F.softmax(Q_22, dim=1)
#         a_22 = T.zeros_like(smax)
#         argmax = smax[0].multinomial(1).data #T.argmax(smax[0])
#         a_22[0][argmax] = (Q_22)[0][argmax]
        Q_21 = self.actor_base2(Q_22.detach())
        return decoded, v2, Q_21, a_22, Q_22, s,g, V_wave
