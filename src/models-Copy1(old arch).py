#here ill store models
# import torch

from torch import Tensor, cat
from agents import Agent
# from train import trainSTAR
import torch.nn as nn
from initializations import *
import torch.nn.functional as F
import torch as T

from torch.distributions.normal import Normal
from ConvLSTM import ConvLSTM,ConvLSTMCell

#kld, x_restored, v, a, hx, cx, s, S

class Decoder(nn.Module):
    def __init__(self, args, device):#1024
        super(Decoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
    
    def forward(self,x):
        x = self.deconv1(x)
        return x 
    
class Oracle(nn.Module):
    def __init__(self, args, device):#1024
        super(Oracle, self).__init__()
        
        self.conv1 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        
        return x

    
class Level1(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level1, self).__init__()
        
        #S
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        
        self.oracle = Oracle({},device)
        self.decoder = Decoder({}, device)
        self.actor = nn.Linear(51203, 6)
        self.critic = nn.Linear(51200, 1)
        
        #32x40x40
        
        self.ConvLSTM_mu = ConvLSTMCell(input_dim=32,
                                 hidden_dim=32,
                                 kernel_size=(11, 11),
#                                  num_layers=4,
#                                  batch_first=True,
                                 bias=True,
#                                  return_all_layers=False
                                       )
        self.ConvLSTM_logvar = ConvLSTMCell(input_dim=32,
                                 hidden_dim=32,
                                 kernel_size=(11, 11),
#                                  num_layers=4,
#                                  batch_first=True,
                                 bias=True,
#                                  return_all_layers=False
                                           )
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        
        x, hx1, cx1, a2_prev = x
        
        s = F.relu(self.maxp1(self.conv1(x)))
        
#         x = torch.unsqueeze(x,1)
        mu, cx11 = self.ConvLSTM_mu(s, (hx1[0],cx1[0]))#(states[0][0][0],states[0][1][0]))
        logvar, cx12 = self.ConvLSTM_logvar(s, (hx1[1],cx1[1]))#(states[0][0][1],states[0][1][1]))
#         std, cx_std = self.ConvLSTM_std(x)
        
        hx11 = mu
        hx12 = logvar
        
        print(mu.shape)
        print(logvar.shape)
#         dist = Normal(mu[0], std[0])
#         z = dist.rsample()

        
        z = mu + T.exp(logvar / 2) * self.N.sample(mu.shape)
        kl = -0.5*(1 + logvar - mu**2 - T.exp(logvar)).mean()
        
        
        decoded = self.decoder(z)
        S = self.oracle(z)
        
        z = z.view(z.size(0), -1)
        print(z.shape)
        v = self.critic(z)
        
        actor_in = T.cat((z.view(z.size(0), -1), a2_prev.view(a2_prev.size(0), -1)),1)
        a = self.actor(actor_in)
        
        #kld1, x_restored1, v1, a1, hx1, cx1, s1, S1
        return kl,decoded,v,a,torch.stack([hx11,hx12]),torch.stack([cx11,cx12]),s,S 

class Decoder2(nn.Module):
    def __init__(self, args, device):#1024
        super(Decoder2, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2)
    
    def forward(self,x):
        x = self.deconv1(x)
        return x 
    
class Oracle2(nn.Module):
    def __init__(self, args, device):#1024
        super(Oracle2, self).__init__()
        
        self.conv1 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        
        return x
    
class Actor2(nn.Module):
    def __init__(self, args, device = "cpu"):
        super(Actor2, self).__init__()
        self.action_mu = nn.Linear(64*6*6, 3)
        self.action_std = nn.Linear(64*6*6, 3)
    def forward(self, S):
        
        a_mean = self.action_mu(T.clone(S)) #prediction form network
        a_log_std = self.action_std(T.clone(S)) #prediction form network
        
        std = torch.exp(a_log_std)
        dist = Normal(a_mean, std)

        a = dist.rsample()
        
        a = torch.tanh(a) #Надо ли ужать?
        
        return a
    
class Level2(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level2, self).__init__()
        
        self.Layernorm = nn.LayerNorm([32,40,40])
        #S
        self.conv1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(3, 3)
        
        self.oracle = Oracle2({},device)
        self.decoder = Decoder2({}, device)
#         self.actor = nn.Linear(64*6*6, 6)
        self.actor = Actor2({},device)
        self.critic = nn.Linear(64*6*6, 1)
        
        #32x40x40
        
        self.ConvLSTM_mu = ConvLSTMCell(input_dim=64,
                                 hidden_dim=64,
                                 kernel_size=(5, 5),
#                                  num_layers=4,
#                                  batch_first=True,
                                 bias=True,
#                                  return_all_layers=False
                                       )
        self.ConvLSTM_logvar = ConvLSTMCell(input_dim=64,
                                 hidden_dim=64,
                                 kernel_size=(5, 5),
#                                  num_layers=4,
#                                  batch_first=True,
                                 bias=True,
#                                  return_all_layers=False
                                           )
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

        
    def forward(self, x):
        
        x, hx2, cx2 = x
        
        x = self.Layernorm(x)
        
        x = F.relu(self.maxp1(self.conv1(x)))
        s = F.relu(self.maxp2(self.conv2(x)))
        
#         x = torch.unsqueeze(x,1)
        mu, cx21 = self.ConvLSTM_mu(s, (hx2[0],cx2[0]))#(states[0][0][0],states[0][1][0]))
        logvar, cx22 = self.ConvLSTM_logvar(s, (hx2[1],cx2[1]))#(states[0][0][1],states[0][1][1]))
#         std, cx_std = self.ConvLSTM_std(x)
        
        hx21 = mu
        hx22 = logvar
        print(mu.shape)
        print(logvar.shape)

        z = mu + T.exp(logvar / 2) * self.N.sample(mu.shape)
        kl = -0.5*(1 + logvar - mu**2 - T.exp(logvar)).mean()
        
        decoded = self.decoder(z)
        S = self.oracle(z)
        
        z = z.view(z.size(0), -1)
        print(z.shape)
        v = self.critic(z)
        a = self.actor(z)
        
        #kld1, x_restored1, v1, a1, hx1, cx1, s1, S1
        return kl,decoded,v,a,torch.stack([hx21,hx22]),torch.stack([cx21,cx22]),s,S 
