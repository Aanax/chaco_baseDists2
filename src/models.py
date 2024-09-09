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
        
        self.conv = nn.Conv2d(70, 32, 5, stride=1, padding=2)

    def forward(self,x, previous_action, previous_state):
        
        previous_action = previous_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,6,20,20)).detach()
        x = torch.cat([x,previous_action, previous_state], dim=1)        
        x = self.conv(x)
        
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
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        
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
        
        kl = mu.sum()*0 #-0.5*(1 + logvar - mu**2 - T.exp(logvar)).sum() # + mu.detach()**2
        
        s = self.layernorm(mu)

        
        return s, kl
    
    
class Level1(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level1, self).__init__()
        
        
#         self.oracle = Oracle({},device)
        self.decoder = Decoder({}, device)
        self.oracle = Oracle({}, device)
        self.encoder = Encoder(args, device)
        self.actor = nn.Linear(12800, 6)
#         self.critic = nn.Linear(12800, 1)
       
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

    def forward(self,x, previous_action, previous_state):
#          = x
        
        s, kl = self.encoder(x)
       
        decoded = self.decoder(s)
                
        S = self.oracle(s.detach(), previous_action.detach(), previous_state.detach())
        
        z = s.view(s.size(0), -1)
#         v = self.critic(z)
        v=None
        
        z = S.detach()-s
        actor_in = z.view(z.size(0), -1)
        Q11 = self.actor(actor_in)
        
#         hx, cx = self.ConvLSTM_mu(s, (hx1,cx1), prev_action_logits, prev_action1_logits)
#         S = self.oracle(hx)
        
#         print("DECODEd shape ", decoded.shape, flush=True)
        return kl,decoded,v,Q11, s, S #, hx, cx, s,S  

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
        
        self.conv = nn.Conv2d(64, 64, 5, stride=1, padding=2)

    def forward(self,x):
        x = self.conv(x)
        return x
    
class Encoder2(nn.Module):
    def __init__(self, args, device):#1024
        super(Encoder2, self).__init__()
        
        self.gamma1 = float(args["Training"]["initial_gamma1"])
        self.gamma2 = float(args["Training"]["initial_gamma2"])
        self.layernorm = nn.LayerNorm([64,5,5])        
#         #20 - 10 - 5
        self.conv1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.maxp2 = nn.MaxPool2d(2, 2)
        
#         self.conv2_logvar = nn.Conv2d(64, 64, 5, stride=1, padding=2)
#         self.maxp2_logvar = nn.MaxPool2d(2, 2)
        
        self.conv1.apply(init_first)
        self.conv2.apply(init_base)
#         self.conv2_logvar.apply(init_base)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        
        
        
    def forward(self, x):
        
#         x = self.Layernorm(x)
        x = F.relu(self.maxp1(self.conv1(x)))
        mu = self.maxp2(self.conv2(x))
#         logvar = self.maxp2_logvar(self.conv2_logvar(x))
        
#         z_t = self.N.sample(mu.shape)
#         s = mu + T.exp(logvar / 2) * z_t #self.z_EMA_t
        
        kl = mu.sum()*0 #-0.5*(1 + logvar - mu**2 - T.exp(logvar)).mean()
        
        s = self.layernorm(mu)
        
        return s, kl
    
    
class Level2(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level2, self).__init__()
        
        self.encoder2 = Encoder2(args, device)
        self.oracle2 = Oracle2({},device)
        self.decoder2 = Decoder2({}, device)
        self.actor2 = nn.Linear(64*5*5, 8) #Actor2(args,device)
        self.actor_base2 = nn.Linear(8, 6)
        self.critic2 = nn.Linear(64*5*5, 1)
        #32x40x40
        self.ConvLSTM_mu2 = ConvLSTMwithAbaseCell(input_dim=64, #withAbase
                                 hidden_dim=64,
                                 kernel_size=(5, 5),
                                 num_actions=8,
                                 bias=True,
                                       )
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        self.decoder2.apply(init_decoder) #CHECK IF WORKS
        self.ConvLSTM_mu2.apply(init_base)            
        self.actor2.weight.data = norm_col_init(
        self.actor2.weight.data, 0.1)#args["Model"]["a_init_std"])
        self.actor2.bias.data.fill_(0)
        self.actor_base2.weight.data = norm_col_init(
        self.actor_base2.weight.data, 0.2)#args["Model"]["a_init_std"])
        self.actor_base2.bias.data.fill_(0)
        
        self.critic2.weight.data = norm_col_init(
        self.critic2.weight.data, args["Model"]["v_init_std"])
        self.critic2.bias.data.fill_(0) 
        
        self.train()
        self.s_mean=0
        self.smean_not_set = True
        self.z_EMA_t = 0
        
    def forward(self, x, hx2, cx2, prev_action): 
        
#         x = torch.cat(x, dim=1)
        s, kl = self.encoder2(x)
        
        decoded = self.decoder2(s)
        
#         print(prev_action.shape)
#         prev_action = prev_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1,8,5,5)).detach()
        hx, cx = self.ConvLSTM_mu2(s.detach(), (hx2.detach(),cx2.detach()), prev_action.detach()) #(states[0][0][0],states[0][1][0]))
    
        S = self.oracle2(hx)
        
#         print(S.shape)
        
        z = S.detach() - s
        
        z = z.view(z.size(0), -1)
        
        v2 = self.critic2(z)
        
        Q_22 = self.actor2(z)
        
#         pi = F.softmax(Q_22)
        V_wave = v2 #(pi*Q_22).sum()
        
        a_22 = ((Q_22-V_wave.detach())>=0).float()
        
        Q_21 = self.actor_base2(a_22.view(a_22.size(0), -1).detach())
        
        return kl, decoded, v2, Q_21, a_22, Q_22, hx,cx,s,S, V_wave
