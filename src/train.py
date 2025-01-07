# file for train function
import os
os.environ["OMP_NUM_THREADS"] = "1"

# file for train function
from functools import wraps
import torch.nn.functional as F
from torch.optim import AdamW

import gc
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
# from models import Level2

from agents import Agent,unload_batch_to_cpu
import torch.nn as nn


import copy
    
def train(rank, args, shared_model, optimizer, env_conf,lock,counter, num, main_start_time="unknown_start_time", RUN_KEY="only"):
    
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
#     _model2 = Level2(args, env.observation_space.shape[0],
#                            env.action_space, device = "cuda:"+str(gpu_id))

    shared_model[0]=shared_model[0].to('cuda')
    #redefine optimizer
    optimizer = AdamW(
        [
        {'params': _model1.parameters(), 'lr': args["Training"]["lr"]},], #0.001},
#         {'params': shared_model[1].parameters(), 'lr': args["Training"]["lr"]},], #*0.05
        lr=args["Training"]["lr"], amsgrad=args["Training"]["amsgrad"]   
        )
    
    player = Agent(_model1, None, env, args, None, gpu_id)
    player.rank = rank
#     player.
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    
    #move model on gpu if needed--------------------------------------------------------------
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model1 = player.model1.cuda()
#             player.model2 = player.model2.cuda()
    player.model1.train()
#     player.model2.train()
    
    
    
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
    n_batch = 0
    
    ### LSTM STATES???
        
    while True:
        # on each run? we
        
        #ISTOPPED HERE!!!!!!!!!!!!!!!!!!!
        
        #copy weights
#         with open("./batch_debug5.txt", "a") as ff:
#                 ff.write('n_batch%20 '+str(n_batch%20)+'\n')
        if n_batch%80==0:
#             print("shared_model WAS", shared_model[0].state_dict())
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
#                     player.model1.load_state_dict(shared_model[0].state_dict())
                    shared_model[0].load_state_dict(player.model1.state_dict())
                    with lock:
                        torch.save(player.model1.state_dict(), "./current_state_dict"+str(RUN_KEY)+".torch")
            w = shared_model[0].state_dict()
#             print("shared_model BECAME ", w[list(w.keys())[0]][0][0])
        # if was done than initializing lstm hiddens with zeros
        #elese initializeing with data
        if player.done:
            player.reset_lstm_states()
        else:
#             if player.train_episodes_run>=4:
            player.detach_lstm_states(levels=[1,2]) #,2
#             if player.train_episodes_run_2>=16:
#                 player.detach_lstm_states(levels=[2])
            
#         with open("./q11s_trainWeights_debug6.txt", "a") as ff:
#             w=player.model1.state_dict()
#             ff.write(str(w[list(w.keys())[0]][0][0])+'\n\n')
#             ff.write(str(w[list(w.keys())[4]][0][0])+'\n\n')
#             ff.write(str(w[list(w.keys())[10]][0][0])+'\n\n')
#             ff.write('\n\n\n-------------------------------\n\n\n')        

        # running simulation for num_steps
        n_batch += 1
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
#             print("shared_model[0].device ",shared_model[0].device, flush=True)
            state = player.state
            x_restored1, v1_ext,v1_int, Q11_ext, Q11_int, s1, g1 = shared_model[0](Variable(
                state.unsqueeze(0)), player.prev_action_1, player.memory_1, player.prev_s1) #, player.prev_g1, player.memory_1
            
            #kl, v, a_21, a_22, Q_22, hx,cx,s,S
#             x_restored2, v2, a_22, Q_22, s2,g2, V_wave = player.model2(player.prev_g1.detach(), player.prev_action_2, player.prev_g2, player.memory_2)
            player.train_episodes_run+=1
            V_last1 = v1_ext.detach()
            Target_Qext = Q11_ext.detach()
            Target_Qint = Q11_int#.detach()
            
            
            A_ext = Q11_ext# - v1_ext
            A_int = Q11_int# - v1_int
            A = A_ext # + A_int
            last_action_probs = F.softmax(A)
            player.action_probss.append(last_action_probs)
            #action1 = action_probs.multinomial(1).data
            #self.actions.append(action1)
            
#             V_last2 = v2.detach()
            s_last1 = s1#g1.detach()
#             g_last2 = g2.detach()
            g_last1 = g1#.detach()
            
            
        with torch.autograd.set_detect_anomaly(True):
            adaptive = False
            
            new_batch_dict = {"states":player.states,
                "rewards":player.rewards1,
             "ss1":player.ss1,
             "actions":player.actions,
             "Q_11s":player.Q_11s,
             "values":player.values1,
             "restoreds":player.restoreds1,
              "restore_labels":player.restore_labels1,
             "gs1":player.gs1,
             "V_exts": player.V_exts,
             "V_ints": player.V_ints,
             "action_probss":player.action_probss,
             "Q_11s_int":player.Q_11s_int,
             "Q_11s_ext":player.Q_11s_ext,
             "memories":player.memory_1s,
                             "prev_g1":torch.zeros((1,32,20,20)).to("cuda:"+str(gpu_id)),
                             "prev_action1":player.first_batch_action}
            
#             player.replay_buffer.append(copy.copy(new_batch_dict))
            
#             print(player.replay_buffer.len)
            
#             N_BATCHES_FOR_TRAIN = 1
#             batch_for_train = player.replay_buffer.sample(N_BATCHES_FOR_TRAIN)
#             batch_for_train=batch_for_train[0]
            
#             prev_g1 = batch_for_train["prev_g1"]#[0]
#             prev_action_1 = batch_for_train["prev_action1"]#[0] #?? #put last action from prev bacth!
#             batch_for_train["Q_11s"]=[]
#             batch_for_train["gs1"]=[]
#             for i in range(len(batch_for_train["states"])):
#                 #predicts
#                 x_restored1, v1, Q_11, s1, g1 = player.model1(batch_for_train["states"][i], prev_action_1) #prev_g1, batch_for_train["memories"][i]
#                 batch_for_train["Q_11s"].append(Q_11)
                
#                 action1 = batch_for_train["actions"][i]
#                 prev_action_1 = torch.zeros((1,6)).to(Q_11.device)
#                 prev_action_1[0][action1.item()] = 1
#                 prev_action_1 = prev_action_1.to(Q_11.device)
                
#                 prev_g1 = batch_for_train["prev_g1"]*0 #batch_for_train["gs1"][i]
                    

#             losses_RB = 0,0,0#train_func(batch_for_train, player.gpu_id, V_last1, s_last1, g_last1, tau, gamma1, w_curiosity, kld_loss_calc, TD_len = '1')
            
#             batch_for_train = unload_batch_to_cpu(batch_for_train, True)
            
            losses = train_func(new_batch_dict, player.gpu_id, Target_Qext,Target_Qint, s_last1, g_last1, tau, gamma1, w_curiosity, kld_loss_calc, TD_len = 'max')

#             kld_loss1, policy_loss1, value_loss1, MPDI_loss1, kld_loss2, policy_loss2, value_loss2, MPDI_loss2, policy_loss_base, kld_loss_actor2, loss_restoration1,loss_restoration2, ce_loss1, ce_loss_base = losses
            
            #value_loss1,
            restoration_loss1_newest, loss_Qext_newest = losses  #g_loss1_newest,  loss_Qint_newest 
#             restoration_loss1_RB, g_loss1_RB, loss_V1_RB = losses_RB #not using rb g and rest
            restoration_loss1 = restoration_loss1_newest
#             g_loss1 = g_loss1_newest
#             loss_V1 = loss_V1_newest# + loss_V1_RB
            loss_Qext = loss_Qext_newest #, loss_Qint_newest , loss_Qint
            
#             g_loss1+=g_loss1_RB
#             loss_V1+=loss_V1_RB

            # loss_Q_21,
            #value_loss1, value_loss2,
            
            losses = list(losses)
#             loss_Q11_non_summed_components = loss_Q_11
#             loss_Q21_non_summed_components = loss_Q_21
#             loss_Q22_non_summed_components = loss_Q_22
            
#             loss_Q_11 = loss_Q_11.sum()
#             loss_Q_21 = loss_Q_21.sum()
#             loss_Q_22 = loss_Q_22.sum()
            
#             losses[1] = loss_Q_11
#             losses[2] = loss_Q_21
#             losses[6] = loss_Q_22
#             loss1 = (args["Training"]["w_policy"]*loss_Q_11)
            

            def get_max_with_abs(tensor1d):
                arg = torch.argmax(torch.abs(tensor1d))
                return tensor1d[arg].item()
#             mean_V1 = torch.mean(torch.Tensor(player.values1)).cpu().numpy()
#             mean_V2 = torch.mean(torch.Tensor(player.values2)).cpu().numpy()
#             mean_Vs2 = torch.mean(torch.Tensor(player.values2)).cpu().numpy()
            mean_Vs1 = torch.mean(torch.Tensor(player.values1)).cpu().numpy()
        
            mean_re1 = float(np.mean(player.rewards1))
            # ii.shape=(1,6) 
            max_Q11_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_11s_ext])) #(20)
            max_Q11_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_11s_ext]))
            max_Q11_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_11s_ext]))
#             max_Q21_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_21s]))
#             max_Q21_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_21s]))
#             max_Q21_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_21s]))
#             max_Q22_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_22s]))
#             max_Q22_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_22s]))
#             max_Q22_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_22s]))

            additional_logs = []
#             print("losses ",losses, flush=True)
            for loss_i in losses:
                if not (loss_i == 0):
                    additional_logs.append(loss_i.item())
                else:
                    additional_logs.append(loss_i)
                    
#             print("loss_Q11_non_summed_components ", loss_Q11_non_summed_components.shape, flush=True)
#             for qloss in loss_Q11_non_summed_components.squeeze():
#                 if not (qloss == 0):
#                     additional_logs.append(qloss.item())
#                 else:
#                     additional_logs.append(qloss)
            
#             for qloss in loss_Q21_non_summed_components.squeeze():
#                 if not (qloss == 0):
#                     additional_logs.append(qloss.item())
#                 else:
#                     additional_logs.append(qloss)
            
            f = open(STATg_CSV_PATH, 'a')
            writer = csv.writer(f)
            writer.writerow([mean_Vs1, mean_re1, counter.value, local_counter, max_Q11_1, max_Q11_2, max_Q11_3,]+additional_logs) #max_Q21_1, max_Q21_2, max_Q21_3,
            f.close()
            
            
            loss_restoration1 = args["Training"]["w_restoration"] * restoration_loss1
#             loss_restoration2 = args["Training"]["w_restoration"] * restoration_loss2
            loss_restoration1.backward(retain_graph=True)
#             (args["Training"]["w_MPDI"]*g_loss1).backward(retain_graph=True)
#             (args["Training"]["w_policy"]*loss_V1).backward(retain_graph=False)
            (args["Training"]["w_policy"]*loss_Qext).backward(retain_graph=False)
#             (args["Training"]["w_policy"]*loss_Qint).backward(retain_graph=False)
        
            
            
#             loss_restoration2.backward(retain_graph=True)
#             loss1+=loss_restoration1
#             loss2+=loss_restoration2
            
#             g_loss1 = args["Training"]["w_MPDI"]*g_loss1
#             g_loss2 = args["Training"]["w_MPDI"]*g_loss2
        
#             g_loss2.backward(retain_graph=True)
#             g_loss1.backward(retain_graph=True)

#             loss1.backward()#retain_graph=False)
#             loss2.backward()#retain_graph=False)
            
            if len(player.rewards1)>2:
#                 ensure_shared_grads(player.model1, shared_model[0], gpu=gpu_id >= 0)
    #             ensure_shared_grads(player.model2, shared_model[1], gpu=gpu_id >= 0)
                optimizer.step()
            player.clear_actions()
            player.model1.zero_grad()
            
            torch.cuda.empty_cache()
            gc.collect()
#             player.model2.zero_grad()
            
#             for p in shared_model[1].actor_base2.parameters():
#                 p.data.clamp_(0)
            
#             for p in player.model2.actor_base2.parameters():
#                 p.data.clamp_(0)
                
            
#             del loss1
#             del loss2

    
def MPDI_loss_calc1(batch_dict, g_last1, tau, gamma1, adaptive, i, advantage_ext):
    #Discounted Features rewards (s - after encoding S-after lstm)
#     print('len(batch_dict["gs1"]) ',len(batch_dict["gs1"]))
#     print('len(batch_dict["rews"]) ',len(batch_dict["rewards"]))
    try:
        g_last1 = g_last1*gamma1 + (batch_dict["ss1"][i+1].detach() - batch_dict["ss1"][i].detach()) #(1-gamma1)*batch_dict["ss1"][i+1].detach()
        
#         print("g_last1.shape ",g_last1.shape, flush=True)
        
        g_advantage1 = F.cosine_similarity(g_last1.ravel(), batch_dict["gs1"][i].ravel(),dim=0) #g_last1-batch_dict["gs1"][i]
        
        return g_last1, -g_advantage1*advantage_ext.detach(), g_advantage1.detach() #g_advantage1.pow(2).sum() #*F.sigmoid(A_ext.detach()) #.pow(2).sum()
    
    except Exception as e:
#         print(e, flush=True)
        return g_last1, (g_last1-batch_dict["gs1"][0]).sum()*0

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



#Below is a refactored version of the `train_A3C_united` function. The refactoring focuses on improving readability, organizing the code for better maintainability, and ensuring efficient use of variables while keeping the core logic intact.

#```python 
def train_A3C_united(batch_dict, gpu_id, Target_Qext, Target_Qint, s_last1, g_last1, tau, gamma1, w_curiosity, kld_loss_calc, TD_len="max"):
    # Initialize losses
    kld_loss1 = 0
    g_loss1 = 0
    loss_V1 = 0
    restoration_loss1 = 0
    loss_Qint = 0
    loss_Qext = 0
    
    # Get episode length and append last state
    T = len(batch_dict["rewards"])
    batch_dict["ss1"].append(s_last1)
    
    # Initialize Q-target and running sum
    QTarget = Target_Qext.max().detach()
    RSum = 0
    k = 1
    
    # Iterate over timesteps in reverse order
    for i in reversed(range(T)):
        # Handle different TD-learning lengths
        if T-i <= k:
            # Short sequence: direct computation
            QTarget = QTarget*gamma1 + batch_dict["rewards"][i]
            RSum = RSum*gamma1 + batch_dict["rewards"][i]
        else:
            # Longer sequence: use k-step returns
            RSum = RSum*gamma1 + batch_dict["rewards"][i] - (batch_dict["rewards"][i+k]*(gamma1**k))
            QTarget = torch.max(batch_dict["Q_11s_ext_T"][i+k])*(gamma1**k) + RSum
        
        # Calculate advantage and clip it
        advantage_ext = QTarget.detach() - batch_dict["Q_11s_ext"][i][0][batch_dict["actions"][i].item()]
        advantage_ext = torch.clip(advantage_ext, -1, 1)
        
        # Accumulate Q-learning loss
        loss_Qext = loss_Qext + (0.5*advantage_ext.pow(2))
        
        # Calculate restoration loss for current timestep
        restoration_loss1_part = (batch_dict["restoreds"][i] - batch_dict["restore_labels"][i]).pow(2).sum()
        restoration_loss1 += restoration_loss1_part  # Note: The screenshot shows additional terms (*abs(D1) + abs(D2)) which are commented out
    
    return restoration_loss1, loss_Qext  # Note: The screenshot shows additional return values that are commented out
