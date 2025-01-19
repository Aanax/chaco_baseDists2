from torch import cat, Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
from initializations import init_base, init_decoder, init_first, norm_col_init

class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(32, 16, 6, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 6, stride=2, padding=2)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv3(x)
        return x 

class Oracle(nn.Module):
    def __init__(self, args, device):
        super(Oracle, self).__init__()
        self.conv = nn.Conv2d(32 + 32 + 6 + 32, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.layernorm = nn.LayerNorm([32, 20, 20])

    def forward(self, x, diff, previous_action, memory):
        previous_action = previous_action.squeeze().unsqueeze(1).unsqueeze(2).expand((1, 6, 20, 20)).detach()
        x = T.cat([x, diff, previous_action, memory], dim=1)
        x = F.relu(self.conv(x))
        x = self.conv2(x)
        x = self.layernorm(x)  
        return x 

class Encoder(nn.Module):
    def __init__(self, args, device):
        super(Encoder, self).__init__()
        self.gamma1 = float(args["Training"]["initial_gamma1"])
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.maxp11 = nn.MaxPool2d(2, 2)
        self.layernorm = nn.LayerNorm([32, 20, 20])
        self.conv1.apply(init_first)
        self.conv11.apply(init_base)
        for m in self.children():
            if not hasattr(m, "name"):
                m.name = None        

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))        
        mu = self.maxp11(self.conv11(x))
        s = self.layernorm(mu)        
        return s
    
class Level1(nn.Module):
    def __init__(self, args, shap, n_actions, device):
        super(Level1, self).__init__()
        self.decoder = Decoder({}, device)
        # self.oracle = Oracle({}, device)
        self.encoder = Encoder(args, device)
        self.actor_ext = nn.Linear(25606, 6)
        # self.actor_int = nn.Linear(12800 * 2, 6)
        for m in self.children():
            if not hasattr(m, "name"):
                m.name = None
        self.decoder.apply(init_decoder)
        self.actor_ext.weight.data = norm_col_init(
        self.actor_ext.weight.data, args["Model"]["a_init_std"])
        self.actor_ext.bias.data.fill_(0)
        
        # self.actor_int.weight.data = norm_col_init(
        # self.actor_int.weight.data, args["Model"]["a_init_std"])
        # self.actor_int.bias.data.fill_(0)
        self.train()
        self.z_EMA_t = 0

    def forward(self, x, previous_action, previous_s):
        s = self.encoder(x)
        decoded = self.decoder(s)
        g = 0
        z = T.cat([(s.detach() - previous_s.detach()).view(s.size(0), -1), s.view(s.size(0), -1), previous_action.detach().view(previous_action.size(0), -1)], dim=1)
        z = z.view(z.size(0), -1)
        
        Q11_ext = self.actor_ext(z)
        Q11_int = 0
        ps = T.nn.functional.softmax(Q11_ext.detach())
        v_ext = (ps * Q11_ext).sum()
        v_int = 0
        return decoded, v_ext, v_int, Q11_ext, Q11_int, s, g
