# here will be init funcs to use with self.apply() in models
import torch.nn as nn
import torch
import numpy as np


# def weight_init():
#     #check layer
    
#     if linear:
#         #linear init
#         if name == "":
#             #..
#         else:
#     if conv:
#         #conv init
        

        
        
def init_xavier(m):
    """
    like base but no norm col init
    """
    relu_gain = nn.init.calculate_gain('relu')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.weight.data.mul_(relu_gain)
        m.bias.data.fill_(0)
        
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        
        
def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def init_base(m):
    relu_gain = 1 #nn.init.calculate_gain('relu')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.weight.data.mul_(relu_gain)
            m.bias.data.fill_(0)
        except:
            pass
        
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
#         try:
#             if m.name=="a_out":
#                 m.weight.data = norm_col_init(
#                     m.weight.data, 0.01)
#             if m.name=="t_out":
#                 m.weight.data = norm_col_init(
#                     m.weight.data, 0.1)
#             if m.name=="S_out":
#                 w_bound = np.sqrt(6. / (fan_in + 2*fan_out))
#                 m.weight.data.uniform_(-w_bound, w_bound)
#                 m.bias.data.fill_(0)
                
#         except:
#             pass
#     elif classname.find('LSTM') != -1:
#         m.bias_ih.data.fill_(0)
#         m.bias_hh.data.fill_(0)
     
            
def init_decoder(m):
    relu_gain = 1 #nn.init.calculate_gain('relu')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(3. / (fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.weight.data.mul_(relu_gain)
            m.bias.data.fill_(0)
        except:
            pass
        
def init_first(m):
    relu_gain = 1 #nn.init.calculate_gain('relu')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(3. / (fan_in))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.weight.data.mul_(relu_gain)
            m.bias.data.fill_(0)
        except:
            pass
        
def init_adaptive(m):
    relu_gain = nn.init.calculate_gain('relu')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.weight.data.mul_(relu_gain)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        if m.name=="a_out": # this needed to select A block linear and init it other way 
            #TODO - rewrite initialization to avoid this
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
        else:
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound- (1/(0.1*(fan_in))), w_bound- (1/(0.1*(fan_in))))
        m.bias.data.fill_(0)