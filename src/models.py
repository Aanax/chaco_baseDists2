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
from ConvLSTM import ConvLSTM,ConvLSTMCell,ConvLSTMwithAbaseCell

#kld, x_restored, v, a, hx, cx, s, S

# class Decoder(nn.Module):
#     def __init__(self, args, device):#1024
#         super(Decoder, self).__init__()
        
#         self.deconv1 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
    
#     def forward(self,x):
#         x = self.deconv1(x)
#         return x 


class Decoder(nn.Module):
    def __init__(self, args, device):#1024
        super(Decoder, self).__init__()
        
#         self.deconv1 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 6, stride=2, padding=2)
#         self.deconv2 = nn.ConvTranspose2d(32, 16, 6, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 6, stride=2, padding=2)
#         self.deconv4 = nn.ConvTranspose2d(16, 1, 6, stride=2, padding=2)
#         self.deconv2 = nn.ConvTranspose2d(16, 1, 5, stride=, padding=0)
    def forward(self,x):
        x = self.deconv1(x)
#         x = self.deconv2(x)
        x = self.deconv3(x)
#         x = self.deconv4(x)
#         x = self.deconv2(x)
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
        self.gamma1 = float(args["Training"]["initial_gamma1"])

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.maxp11 = nn.MaxPool2d(2, 2)
        self.conv11_logvar = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.maxp11_logvar = nn.MaxPool2d(2, 2)      
        self.oracle = Oracle({},device)
        self.decoder = Decoder({}, device)
        self.actor = nn.Linear(12800, 6)
        self.critic = nn.Linear(12800, 1)
        #32x40x40
        self.ConvLSTM_mu = ConvLSTMwithAbaseCell(input_dim=32,
                                 hidden_dim=32,
                                 kernel_size=(5, 5),
                                 bias=True,
                                 num_actions=6,
                                       )
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        if args["Model"]["initialization"]=="xavier":
            self.apply(init_xavier)
        elif args["Model"]["initialization"]=="adaptive":
            self.apply(init_adaptive)
        elif args["Model"]["initialization"]=="base":
            self.apply(init_base)
            
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv11.weight.data.mul_(relu_gain)
        self.conv11_logvar.weight.data.mul_(relu_gain)
        
        self.actor.weight.data = norm_col_init(
        self.actor.weight.data, args["Model"]["a_init_std"])
        self.actor.bias.data.fill_(0)
        
#         self.actor.weight.data[:,-4:] = self.actor.weight.data[:,-4:]*35
#         self.actor.weight.data[:,-4:] = norm_col_init(
#         self.actor.weight.data[:,-4:], args["a_init_std"])
#         self.actor.bias.data.fill_(0)
        
        self.critic.weight.data = norm_col_init(
        self.critic.weight.data, args["Model"]["v_init_std"])
        self.critic.bias.data.fill_(0) 
        
        self.oracle.conv1.weight.data = self.oracle.conv1.weight.data*args["Model"]["S_init_std_multiplier"]
        for name, p in self.named_parameters():
            if "lstm" in name:
                if ("weight_ih" in name) or ("t_layer.weight" in name):
                    nn.init.xavier_uniform_(p.data, gain=args["Model"]["lstm_init_gain"])
                elif ("weight_hh" in name) or ("pre_t_layer.weight" in name):
                    nn.init.orthogonal_(p.data)
                elif ("bias_ih" in name) or ("t_layer.bias" in name):
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif ("bias_hh" in name) or ("pre_t_layer.bias" in name):
                    p.data.fill_(0)
        self.train()
        self.z_EMA_t = 0

    def forward(self, x):
        x, hx1, cx1, prev_action_logits = x
        
        x = F.relu(self.maxp1(self.conv1(x)))
        mu = self.maxp11(self.conv11(x))
        logvar = self.maxp11_logvar(self.conv11_logvar(x))
        
#         s = mu + T.exp(logvar / 2) * self.N.sample(mu.shape)
#         kl = -0.5*(1 + logvar - mu**2 - T.exp(logvar)).mean()

        z_t = self.N.sample(mu.shape)
        self.z_EMA_t= z_t #self.z_EMA_t*self.gamma1 + z_t*(1-self.gamma1)
        s = mu + T.exp(logvar / 2) * self.z_EMA_t
        
        kl = -0.5*(1 + logvar - mu**2 - T.exp(logvar)).mean() # + mu.detach()**2

        decoded = self.decoder(s)
        
#         print(s.shape)
        hx, cx = self.ConvLSTM_mu(s, (hx1,cx1), prev_action_logits)#(states[0][0][0],states[0][1][0]))
        S = self.oracle(hx)
        
#         print(S.shape)
        
        z = hx.view(hx.size(0), -1)
        v = self.critic(z)
        actor_in = z.view(z.size(0), -1) #T.cat((z.view(z.size(0), -1), a2_prev.view(a2_prev.size(0), -1)),1)
        a = self.actor(actor_in)
        return kl,decoded,v,a, hx, cx, s,S  

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
        
        self.conv1 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        
        return x
    
class Actor2(nn.Module):
    def __init__(self, args, device = "cpu"):
        super(Actor2, self).__init__()
        self.action_mu = nn.Linear(64*5*5, 8)
        self.action_std = nn.Linear(64*5*5, 8)
        self.gamma2 = float(args["Training"]["initial_gamma2"])
#         self.action_mu.weight.data = norm_col_init(
#         self.action_mu.weight.data, args["a_init_std"]/4)
#         self.action_mu.bias.data.fill_(0)
#         self.action_std.weight.data = norm_col_init(
#         self.action_std.weight.data, args["a_init_std"]/4)
#         self.action_std.bias.data.fill_(0)
        self.a_mean=0
        self.amean_not_set=True
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl2 = 0
        self.z_EMA_t = 0

    def forward(self, S):
        
        a_mean = self.action_mu(T.clone(S)) #prediction form network
        a_log_std = self.action_std(T.clone(S)) #prediction form network
        
#         a_log_std = T.clamp(a_log_std, min=-20, max=2)
        
#         s = mu + T.exp(logvar / 2) * self.N.sample(mu.shape)
        kl2 = -0.5*(1 + 2*a_log_std - a_mean**2 - T.exp(2*a_log_std)).mean()
    
#         kl2 = kl2/8
    
#         std = torch.exp(a_log_std)
#         dist = Normal(a_mean, std)

#         a = dist.rsample()
        z_t = self.N.sample(a_mean.shape)
#         self.z_EMA_t = self.z_EMA_t*self.gamma1 + np.sqrt(1-self.gamma1**2)*z_t 
#sqrt(1-g2^2)
        self.z_EMA_t=self.z_EMA_t*self.gamma2 + z_t*np.sqrt(1-self.gamma2**2)

        a = a_mean + T.exp(a_log_std) * self.z_EMA_t
        
#         if self.amean_not_set:
#             self.a_mean = T.zeros_like(a)
#             self.amean_not_set=False
#         self.a_mean = self.a_mean.detach()*(self.gamma1)+a*(1-self.gamma1**2) 
#         a = torch.tanh(a) #Надо ли ужать?
        log_probs2 = None #dist.log_prob(a)
        return a, None, log_probs2, kl2

class Level2(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level2, self).__init__()
        self.gamma1 = float(args["Training"]["initial_gamma1"])
        self.gamma2 = float(args["Training"]["initial_gamma2"])
        self.Layernorm = nn.LayerNorm([32,20,20])        
        #20 - 10 - 5
        self.conv1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.maxp2 = nn.MaxPool2d(2, 2)
        
        self.conv2_logvar = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.maxp2_logvar = nn.MaxPool2d(2, 2)
        
        self.oracle = Oracle2({},device)
        self.decoder = Decoder2({}, device)
        self.actor = Actor2(args,device)
        self.actor_base = nn.Linear(8, 6)
        self.critic = nn.Linear(64*5*5, 1)
        #32x40x40
        self.ConvLSTM_mu = ConvLSTMCell(input_dim=64,
                                 hidden_dim=64,
                                 kernel_size=(5, 5),
                                 bias=True,
                                       )
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        
        for m in self.children():
            if not hasattr(m,"name"):
                m.name = None
        if args["Model"]["initialization"]=="xavier":
            self.apply(init_xavier)
        elif args["Model"]["initialization"]=="adaptive":
            self.apply(init_adaptive)
        elif args["Model"]["initialization"]=="base":
            self.apply(init_base)
            
        relu_gain = nn.init.calculate_gain('relu')
        self.actor_base.weight.data = norm_col_init(
        self.actor_base.weight.data, args["Model"]["a_init_std"])
        self.actor_base.bias.data.fill_(0)
        
#         self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2_logvar.weight.data.mul_(relu_gain)    
        self.critic.weight.data = norm_col_init(
        self.critic.weight.data, args["Model"]["v_init_std"])
        self.critic.bias.data.fill_(0) 
        
        self.oracle.conv1.weight.data = self.oracle.conv1.weight.data*args["Model"]["S_init_std_multiplier"]
        
        for name, p in self.named_parameters():
            if "lstm" in name:
                if ("weight_ih" in name) or ("t_layer.weight" in name):
                    nn.init.xavier_uniform_(p.data, gain=args["Model"]["lstm_init_gain"])
                elif ("weight_hh" in name) or ("pre_t_layer.weight" in name):
                    nn.init.orthogonal_(p.data)
                elif ("bias_ih" in name) or ("t_layer.bias" in name):
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif ("bias_hh" in name) or ("pre_t_layer.bias" in name):
                    p.data.fill_(0)
        self.train()
        self.s_mean=0
        self.smean_not_set = True
        self.z_EMA_t = 0
        
    def forward(self, x):
        x, hx2, cx2 = x
        x = self.Layernorm(x)
        x = F.relu(self.maxp1(self.conv1(x)))
        mu = self.maxp2(self.conv2(x))
        logvar = self.maxp2_logvar(self.conv2_logvar(x))
        
        z_t = self.N.sample(mu.shape)
#         self.z_EMA_t = self.z_EMA_t*self.gamma2 + np.sqrt(1-self.gamma2**2) * z_t 
        self.z_EMA_t=self.z_EMA_t*self.gamma1 + z_t*(1-self.gamma1)
        s = mu + T.exp(logvar / 2) * self.z_EMA_t
        
        kl = -0.5*(1 + logvar - mu**2 - T.exp(logvar)).mean() 
        # + mu.detach()**2
        
#         s_separate = mu + T.exp(logvar / 2) * z_t
        decoded = self.decoder(s)
#         if self.smean_not_set:
#             self.s_mean = T.zeros_like(s)
#             self.smean_not_set = False
            
#         self.s_mean = self.s_mean.detach()*(self.gamma1)+s*(1-self.gamma1**2) 

        hx, cx = self.ConvLSTM_mu(s, (hx2,cx2))#(states[0][0][0],states[0][1][0]))

        S = self.oracle(hx)
        z = hx.view(hx.size(0), -1)
        v = self.critic(z)
        a, entropy2, log_prob2, kl_actor2 = self.actor(z)
        
        a_base = self.actor_base(a.view(a.size(0), -1))
        
        return kl,decoded,v,a, a_base, hx,cx,s,S,entropy2,log_prob2,kl_actor2
