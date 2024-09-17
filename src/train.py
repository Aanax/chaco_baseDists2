# file for train function
import os
os.environ["OMP_NUM_THREADS"] = "1"

# file for train function
from functools import wraps
import torch.nn.functional as F


import time

import cv2

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print ('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

from setproctitle import setproctitle as ptitle
import torch
import csv
import numpy as np
from environment import atari_env
from torch.autograd import Variable
from utils import ensure_shared_grads

from models import Level1
from models import Level2

from agents import Agent
import torch.nn as nn


import copy
    
def train(rank, args, shared_model, optimizer, env_conf,lock,counter, num, main_start_time="unknown_start_time"):
    
    #set process title (visible in nvidia-smi!)
    ptitle('Training Agent: {}'.format(rank))
    
    
    #preparing for logging ---------------------------------------------------------------
    try:
        os.mkdir("./"+args["Training"]["log_dir"]+"/stats"+str(main_start_time)+"_"+str(num)+"/")
    except:
        pass
    STATg_CSV_PATH = "./"+args["Training"]["log_dir"]+"/stats"+str(main_start_time)+"_"+str(num)+"/"+"_STATS"+str(rank)+".csv"
    f = open(STATg_CSV_PATH, 'w+')
    f.close()
    
    # getting gpu (rank is jusk number of worker)
    gpu_id = args["Training"]["gpu_ids"][rank % len(args["Training"]["gpu_ids"])]
    torch.manual_seed(int(args["Training"]["seed"]) + rank)    
    if gpu_id >= 0:
        torch.cuda.manual_seed(int(args["Training"]["seed"]) + rank)
        
        
        
    #creating env -----------------------------------------------------  
    env = atari_env(args["Training"]["env"], env_conf, args)
            
    env.seed(int(args["Training"]["seed"]) + rank)
    
    #creating Agent (wrapper around model capable of train step and test step making)------------
    model_params_dict = args["Model"]
    _model1 = Level1(args, env.observation_space.shape[0],
                           env.action_space, device = "cuda:"+str(gpu_id))
    _model2 = Level2(args, env.observation_space.shape[0],
                           env.action_space, device = "cuda:"+str(gpu_id))
    
    player = Agent(_model1, _model2, env, args, None, gpu_id)
    player.rank = rank
#     player.
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    
    #move model on gpu if needed--------------------------------------------------------------
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model1 = player.model1.cuda()
            player.model2 = player.model2.cuda()
    player.model1.train()
    player.model2.train()
    
    
    
    # init constants ---------------------------------------------------------------------------
    train_func = train_A3C_united
    print("-----------USING TRAINFUNC ",train_func)
    
    player.eps_len += 2
    local_counter = 0
    tau = args["Training"]["tau"]
    gamma1 = args["Training"]["initial_gamma1"]
    gamma2 = args["Training"]["initial_gamma2"]
    w_curiosity = float(args["Training"]["w_curiosity"])
    game_num=0
    frame_num = 0
    future_last = 0
    g_last = torch.Tensor([0])
    

    kld_loss_calc = _kld_loss_calc
    
    
    ### LSTM STATES???
        
    while True:
        # on each run? we
        
        #ISTOPPED HERE!!!!!!!!!!!!!!!!!!!
        
        #load shared weights
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model1.load_state_dict(shared_model[0].state_dict())
                player.model2.load_state_dict(shared_model[1].state_dict())
        
        # if was done than initializing lstm hiddens with zeros
        #elese initializeing with data
        if player.done:
            player.reset_lstm_states()
        else:
#             if player.train_episodes_run>=4:
            player.detach_lstm_states(levels=[1,2]) #,2
#             if player.train_episodes_run_2>=16:
#                 player.detach_lstm_states(levels=[2])
            
                

        # running simulation for num_steps
        for step in range(args["Training"]["num_steps"]):
            player.action_train()
            with lock:
                counter.value += 1
            local_counter+=1
            if player.done:
                break
        
        # if simulation ended because game ended
        # -resetting simulation
        #
        if player.done:
            player.eps_len = 0
            game_num+=1
            is_new_episode = True
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        
        
        with torch.cuda.device(gpu_id):
            V_last1 = torch.zeros(1, 1).cuda()
            gae1 = torch.zeros(1, 1).cuda()
            g_last1 = torch.zeros(1, 1).cuda()
            V_last1 = torch.zeros(1, 1).cuda()
            gae2 = torch.zeros(1, 1).cuda()
            g_last2 = torch.zeros(1, 1).cuda()
        
        if not player.done:
            state = player.state
            x_restored1, v1, Q_11, s1, g1 = player.model1(Variable(
                state.unsqueeze(0)), player.prev_action_1, player.prev_g1, player.memory_1)
            
            #kl, v, a_21, a_22, Q_22, hx,cx,s,S
            x_restored2, v2, Q_21, a_22, Q_22, s2,g2, V_wave = player.model2(player.prev_g1.detach(), player.prev_action_2, player.prev_g2, player.memory_2)
            player.train_episodes_run+=1
            V_last1 = v1.detach()
            V_last2 = v2.detach()
            s_last1 = s1#g1.detach()
            g_last2 = g2.detach()
            g_last1 = g1.detach()
            
            
        with torch.autograd.set_detect_anomaly(True):
            adaptive = False

            losses = train_func(player, V_last1, V_last2, s_last1, g_last1, g_last2, tau, gamma1, gamma2, w_curiosity, kld_loss_calc)

#             kld_loss1, policy_loss1, value_loss1, MPDI_loss1, kld_loss2, policy_loss2, value_loss2, MPDI_loss2, policy_loss_base, kld_loss_actor2, loss_restoration1,loss_restoration2, ce_loss1, ce_loss_base = losses
            
            #value_loss1,
            restoration_loss1, loss_Q_11, loss_Q_21, value_loss1, value_loss2, g_loss2, loss_Q_22, g_loss1, restoration_loss2 = losses
            
            losses = list(losses)
            loss_Q11_non_summed_components = loss_Q_11
            loss_Q21_non_summed_components = loss_Q_21
            loss_Q22_non_summed_components = loss_Q_22
            
            loss_Q_11 = loss_Q_11.sum()
            loss_Q_21 = loss_Q_21.sum()
            loss_Q_22 = loss_Q_22.sum()
            
            losses[1] = loss_Q_11
            losses[2] = loss_Q_21
            losses[6] = loss_Q_22
#             loss1 = (args["Training"]["w_policy"]*loss_Q_11)
            (args["Training"]["w_policy"]*loss_Q_11).backward(retain_graph=True)
            (args["Training"]["w_policy"]*loss_Q_21).backward(retain_graph=True)
            (args["Training"]["w_policy"]*loss_Q_22).backward(retain_graph=True)
            (args["Training"]["w_value"]*value_loss2).backward(retain_graph=True)
            (args["Training"]["w_value"]*value_loss1).backward(retain_graph=True)
#             loss2 = (args["Training"]["w_policy"]*loss_Q_21+args["Training"]["w_value"]*value_loss2+args["Training"]["w_policy"]*loss_Q_22)
#             (args["Training"]["w_kld"]*kld_loss1).backward(retain_graph=True)
#             (args["Training"]["w_kld"]*kld_loss2).backward(retain_graph=True)
#             loss1 += args["Training"]["w_kld"]*kld_loss1
#             loss2 += args["Training"]["w_kld"]*kld_loss2
            (args["Training"]["w_MPDI"]*g_loss1).backward(retain_graph=True)
            
            (args["Training"]["w_MPDI"]*g_loss2).backward(retain_graph=True)
            

#             loss2 += args["Training"]["w_MPDI"]*g_loss2
#             loss1 += args["Training"]["w_MPDI"]*g_loss1
            

            def get_max_with_abs(tensor1d):
                arg = torch.argmax(torch.abs(tensor1d))
                return tensor1d[arg].item()
#             mean_V1 = torch.mean(torch.Tensor(player.values1)).cpu().numpy()
#             mean_V2 = torch.mean(torch.Tensor(player.values2)).cpu().numpy()
            mean_Vs_wave = torch.mean(torch.Tensor(player.Vs_wave)).cpu().numpy()
            mean_re1 = float(np.mean(player.rewards1))
            max_Q11_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_11s]))
            max_Q11_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_11s]))
            max_Q11_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_11s]))
            max_Q21_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_21s]))
            max_Q21_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_21s]))
            max_Q21_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_21s]))
            max_Q22_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_22s]))
            max_Q22_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_22s]))
            max_Q22_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_22s]))

            additional_logs = []

            for loss_i in losses:
                if not (loss_i == 0):
                    additional_logs.append(loss_i.item())
                else:
                    additional_logs.append(loss_i)
                    
            print("loss_Q11_non_summed_components ", loss_Q11_non_summed_components.shape, flush=True)
            for qloss in loss_Q11_non_summed_components.squeeze():
                if not (qloss == 0):
                    additional_logs.append(qloss.item())
                else:
                    additional_logs.append(qloss)
            
            for qloss in loss_Q21_non_summed_components.squeeze():
                if not (qloss == 0):
                    additional_logs.append(qloss.item())
                else:
                    additional_logs.append(qloss)
            
            f = open(STATg_CSV_PATH, 'a')
            writer = csv.writer(f)
            writer.writerow([mean_Vs_wave, mean_re1, counter.value, local_counter, max_Q11_1, max_Q11_2, max_Q11_3, max_Q21_1, max_Q21_2, max_Q21_3, max_Q22_1, max_Q22_2, max_Q22_3]+additional_logs)
            f.close()
            
            
            loss_restoration1 = args["Training"]["w_restoration"] * restoration_loss1
            loss_restoration2 = args["Training"]["w_restoration"] * restoration_loss2
            loss_restoration1.backward(retain_graph=True)
            loss_restoration2.backward(retain_graph=True)
#             loss1+=loss_restoration1
#             loss2+=loss_restoration2
            
#             g_loss1 = args["Training"]["w_MPDI"]*g_loss1
#             g_loss2 = args["Training"]["w_MPDI"]*g_loss2
        
#             g_loss2.backward(retain_graph=True)
#             g_loss1.backward(retain_graph=True)

#             loss1.backward()#retain_graph=False)
#             loss2.backward()#retain_graph=False)

            ensure_shared_grads(player.model1, shared_model[0], gpu=gpu_id >= 0)
            ensure_shared_grads(player.model2, shared_model[1], gpu=gpu_id >= 0)
            optimizer.step()
            player.clear_actions()
            player.model1.zero_grad()
            player.model2.zero_grad()
#             del loss1
#             del loss2

    
def MPDI_loss_calc1(player, V_last1, g_last1, tau, gamma1, adaptive, i):
    #Discounted Features rewards (s - after encoding S-after lstm)
    try:
        g_last1 = g_last1*gamma1 + (1-gamma1)*player.ss1[i+1].detach()
        g_advantage1 = g_last1-player.gs1[i]
        return g_last1, 0.5 * g_advantage1.pow(2).sum()
    except Exception as e:
#         print(e, flush=True)
        return g_last1, (g_last1-player.gs1[0]).sum()*0

def MPDI_loss_calc2(player, V_last2, g_last2, tau, gamma2, adaptive, i):
    #Discounted Features rewards (s - after encoding S-after lstm)
    try:
        g_last2 = g_last2*gamma2 + (1-gamma2)*player.ss2[i+1].detach()
        g_advantage2 = g_last2-player.gs2[i]
        return g_last2, 0.5 * g_advantage2.pow(2).sum()
    except Exception as e:
#         print(e, flush=True)
        return g_last2, (g_last2-player.gs2[0]).sum()*0
    

def _kld_loss_calc(player, i):
    return player.klds1[i], player.klds2[i]

def _kld_loss_calc_filler(player, i):
    return 0

def get_pixel_change(pic1, pic2, STEP = 20):
    """
    pic1 pytorch tensor
    pic2
    returnss 4x4 averages tensor (20 20 pixels maps using)
    """
    max_side = pic1.shape[-1]
    res = torch.zeros((max_side//STEP, max_side//STEP))
    for n1,i in enumerate(range(0,max_side,STEP)):
        for n2,j in enumerate(range(0,max_side,STEP)):
            res[n1,n2]=(pic1[:,i:i+STEP, j:j+STEP]-pic2[:,i:i+STEP, j:j+STEP]).mean().abs()
    
    return res

    
def train_A3C_united(player, V_last1, V_last2, s_last1, g_last1, g_last2, tau, gamma1, gamma2, w_curiosity, kld_loss_calc):
    player.values1.append(V_last1) #Variable
#     player.gs1.append(g_last1)
    player.values2.append(V_last2) #Variable
    player.gs2.append(g_last2)
    policy_loss1 = 0
    value_loss1 = 0
    gae1 = torch.zeros(1, 1)
    if player.gpu_id >= 0:
        with torch.cuda.device(player.gpu_id):
            gae1 = gae1.cuda()
    kld_loss1 = 0
    g_loss1 = 0
    policy_loss2 = 0
    policy_loss_base = 0
    value_loss2 = 0
    v2_reward = 0
#     gae2 = torch.zeros(1, 1)
#     if player.gpu_id >= 0:
#         with torch.cuda.device(player.gpu_id):
#             gae2 = gae2.cuda()
    kld_loss2 = 0
    g_loss2 = 0
    loss_Q_21 = 0
    loss_Q_22 = 0
    loss_Q_11 = 0
    target_Q_21 = 0
    target_Q_22 = 0
    kld_loss_actor2 = 0
    V1_runningmean=0
    restoration_loss1=0
    restoration_loss2=0
    CE_loss = nn.CrossEntropyLoss()

    D1=0
    D2=0
    ce_loss_base = 0
    ce_loss1 = 0    
    VD_runningmean = player.VD_runningmean
    VD_runningmeans = []
    
    T = len(player.rewards1)
    player.ss1.append(s_last1)
    
    for i in reversed(range(len(player.rewards1))):
        #restoration_loss
        g_last2, part_g_loss2 = MPDI_loss_calc2(player, V_last2, g_last2, tau, gamma2, None, i)
        g_loss2 += part_g_loss2
        
#         g_loss1 += 0.5*((player.gs1[i]-player.ss1[i+1].detach())**2).sum()

        g_last1, part_g_loss2 = MPDI_loss_calc1(player, V_last1, g_last1, tau, gamma1, None, i)
        g_loss1 += part_g_loss2
        
        delta_t1 = player.rewards1[i] + gamma1 * \
            player.values1[i + 1].data - player.values1[i].data
        
        D1 = D1*gamma1 + delta_t1
        
        # Generalized Advantage Estimataion
#         delta_t2 = (1-gamma1)*(player.values1[i].detach()+D1) + gamma2 * \
#             player.values2[i + 1].data - player.values2[i].data

        v1_corr = player.values1[i].detach()+D1

        delta_t2 = (1-gamma1)*v1_corr + gamma2 * \
            player.values2[i + 1].data - player.values2[i].data
        
        D2 = D2*gamma2 + delta_t2
        
        
        v2_corr = player.values2[i].detach()+D2
                
#         #KL loss
#         kld_delta1, kld_delta2 = kld_loss_calc(player, i)
#         kld_loss1+=kld_delta1 #*(abs(D1) + abs(D2))
#         kld_loss2+=kld_delta2 #*abs(D2)
        
        restoration_loss1_part = (player.restoreds1[i] - player.restore_labels1[i]).pow(2).sum()
        restoration_loss1 += restoration_loss1_part #*(abs(D1) + abs(D2))
        
        restoration_loss2_part = (player.restoreds2[i] - player.restore_labels2[i]).pow(2).sum()
        restoration_loss2 += restoration_loss2_part #*(abs(D1) + abs(D2))
        
        #loss a11
        v2_reward = v2_reward*gamma1 + (1-gamma1)*v2_corr 
        target_Q_11 = v1_corr + v2_reward
        loss_mask = torch.zeros((1,6))
        loss_mask[0][player.actions[i].item()] = 1
        loss_mask = loss_mask.to(player.Q_11s[i].device)
        loss_Q_11 = loss_Q_11 + 0.5 * (((player.Q_11s[i]-target_Q_11)**2)*loss_mask)
        
        #loss a21
#         target_Q_21 = 
#         #gamma2 * target_Q_21 + (1-gamma1)*player.Q_11s[i].detach()
#         advantage_Q_21 = target_Q_21 - player.Q_21s[i]

        loss_Q_21 +=  0.5 * ((player.Q_21s[i] - v2_corr).pow(2)).sum()*loss_mask
        
        loss_mask_Q22 = torch.zeros((1,8))
        loss_mask_Q22[player.a_22s[i]!=0] = 1
        if loss_mask_Q22.sum()==0:
            loss_mask_Q22[torch.randint(8,(1,))] = 1
        loss_mask_Q22 = loss_mask.to(player.Q_22s[i].device)
        
#         target_Q_22 = gamma2 * target_Q_22 + (1-gamma1)*player.values1[i].detach()
#         advantage_Q_22 = target_Q_22 - player.Q_22s[i]
#         loss_Q_22 = loss_Q_22 + 0.5 * (advantage_Q_22.pow(2)*loss_mask)
        loss_Q_22 +=  0.5 * ((player.Q_22s[i] - v2_corr).pow(2)).sum()*loss_mask_Q22
        
        #value loss
        V_last2 = gamma2 * V_last2 + (1-gamma1)*v1_corr
        #gamma2 * V_last2 + (1-gamma1)*player.values1[i].detach()
        advantage2 = V_last2 - player.values2[i]
        value_loss2 = value_loss2 + 0.5 * advantage2.pow(2)
        
        V_last1 = gamma1 * V_last1 + player.rewards1[i]
        advantage1 = V_last1 - player.values1[i]
        value_loss1 = value_loss1 + 0.5 * advantage1.pow(2)


    #value_loss1*0,
    return restoration_loss1, loss_Q_11, loss_Q_21, value_loss1, value_loss2, g_loss2, loss_Q_22, g_loss1, restoration_loss2

