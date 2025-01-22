import os
os.environ["OMP_NUM_THREADS"] = "1"
from functools import wraps
import torch.nn.functional as F
from torch.optim import AdamW
import gc
import time
import cv2
from setproctitle import setproctitle as ptitle
import torch
import csv
import numpy as np
from environment import atari_env
from torch.autograd import Variable
from utils import ensure_shared_grads
from models import Level1
from agents import Agent,unload_batch_to_cpu
import torch.nn as nn
import copy

#i need
#STATg_CSV_PATH (or writer), env, gpu_id, player, optimizer, 

#asserts. assert weights changed for shared and for model
# assert batchsize >=1?
# assert ...
    
def inits_for_train(rank, args, target_model, shared_model, optimizer, env_conf,lock,counter, num, main_start_time="unknown_start_time", RUN_KEY="only", logging=True):
    
    #preparing for logging ---------------------------------------------------------------
    STATg_CSV_PATH = "./"+args["Training"]["log_dir"]+"/stats"+str(main_start_time)+"_"+str(num)+"/"+"_STATS"+str(rank)+".csv"
    if logging:
        try:
            os.mkdir("./"+args["Training"]["log_dir"]+"/stats"+str(main_start_time)+"_"+str(num)+"/")
        except:
            pass
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
    _model1 = Level1(args, env.observation_space.shape[0],
                           env.action_space, device = "cuda:"+str(gpu_id))
    target_model[0]=target_model[0].to('cuda')

#     #redefine optimizer
#     optimizer = AdamW(
#         [
#         {'params': _model1.parameters(), 'lr': args["Training"]["lr"]},], #0.001},
# #         {'params': target_model[1].parameters(), 'lr': args["Training"]["lr"]},], #*0.05
#         lr=args["Training"]["lr"], amsgrad=args["Training"]["amsgrad"]   
#         )
#     optimizer.zero_grad()
    
    player = Agent(_model1, target_model[0], env, args, None, gpu_id)
    player.rank = rank
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    
    #move model on gpu if needed--------------------------------------------------------------
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model1 = player.model1.cuda()
    player.model1.train()
    
    return STATg_CSV_PATH, gpu_id, player, optimizer
    
def synchronize_target_model(gpu_id, target_model, shared_model):
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            target_model[0].load_state_dict(shared_model[0].state_dict())

            # with lock:
                # torch.save(player.model1.state_dict(), "./current_state_dict"+str(RUN_KEY)+".torch")
def synchronize_player_model(gpu_id, player, shared_model):
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model1.load_state_dict(shared_model[0].state_dict())

def train(rank, args, target_model, shared_model, optimizer, env_conf,lock,counter, num, main_start_time="unknown_start_time", RUN_KEY="only", logging=True):
    
    #set process title (visible in nvidia-smi!)
    ptitle('Training Agent: {}'.format(rank))
    
    # STATg_CSV_PATH, gpu_id, player, optimizer = inits_for_train(rank, args, target_model, shared_model, optimizer, env_conf,lock,counter, num, main_start_time, RUN_KEY)
     
    #preparing for logging ---------------------------------------------------------------
    if logging:
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
    model1 = Level1(args, env.observation_space.shape[0],
                           env.action_space, device = "cuda:"+str(gpu_id))
    target_model[0]=target_model[0].to('cuda')

#     #redefine optimizer
#     optimizer = AdamW(
#         [
#         {'params': _model1.parameters(), 'lr': args["Training"]["lr"]},], #0.001},
# #         {'params': target_model[1].parameters(), 'lr': args["Training"]["lr"]},], #*0.05
#         lr=args["Training"]["lr"], amsgrad=args["Training"]["amsgrad"]   
#         )
#     optimizer.zero_grad()
    
    player = Agent(model1, target_model[0], env, args, None, gpu_id)
    player.rank = rank
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    
    #move model on gpu if needed--------------------------------------------------------------
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model1 = player.model1.cuda()
    player.model1.train()


    # init constants ---------------------------------------------------------------------------
    train_func = calculate_loss
    print("-----------USING TRAINFUNC ",train_func)
    
    local_counter = 0
    tau = args["Training"]["tau"]
    gamma1 = args["Training"]["initial_gamma1"]
    w_curiosity = float(args["Training"]["w_curiosity"])
    game_num=0
    kld_loss_calc = _kld_loss_calc
    n_batch = 0
    
    ### LSTM STATES???
        
    while True:                
        
        if player.done:
            player.reset_lstm_states()
        else:
            #no BPTT
            player.detach_lstm_states(levels=[1,2]) #,2

        # running simulation for num_steps
        n_batch += 1
        for step in range(args["Training"]["num_steps"]):
            player.action_train()
            if lock:
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
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            player.train_episodes_run+=1
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
            
        
        #setting proper V values and etc (act as defaults for when game ended)
        with torch.cuda.device(gpu_id):
            g_last1 = torch.zeros(1, 1).cuda()
            Q11_ext = torch.zeros(1, 6).cuda()
            Q11_int = torch.zeros(1, 6).cuda()
            s_last1 = torch.zeros(1, 32, 20, 20).cuda()
        
        if not player.done:
            if False:
                state = player.state
                x_restored1, v1_ext, v1_int, Q11_ext, Q11_int, s1, g1 = target_model[0](Variable(
                    state.unsqueeze(0)), player.prev_action_1, player.prev_s1) #, player.prev_g1, player.memory_1
                Target_Qext = torch.max(Q11_ext.detach()).item()
                Target_Qint = Q11_int #.detach()
            else:
                Q11_exts = []
                Q11_ints = []
                for state, prev_action, memory, prev_s1 in zip(
                    player.states, player.prev_action_1s, player.memory_1s, player.prev_s1s
                ):
                    x_restored1, v1_ext, v1_int, Q11_ext, Q11_int, s1, g1 = shared_model[0](Variable(
                        state), prev_action, memory, prev_s1)
                    Q11_exts.append(Q11_ext)
                    Q11_ints.append(Q11_int)

                state = player.state
                x_restored1, v1_ext, v1_int, Q11_ext, Q11_int, s1, g1 = shared_model[0](Variable(
                    state.unsqueeze(0)), player.prev_action_1, player.memory_1, player.prev_s1)

                Q11_exts.append(Q11_ext)
                Q11_ints.append(Q11_int)

                player.train_episodes_run += 1
                Target_Qext = Q11_exts
                Target_Qint = Q11_ints
            last_action_probs = F.softmax(Q11_ext)
            player.action_probss.append(last_action_probs)
            s_last1 = s1 #g1.detach()
            g_last1 = g1 #.detach()
            
            
        with torch.autograd.set_detect_anomaly(True):            
            new_batch_dict = {"states":player.states,
                "rewards":player.rewards1,
             "ss1":player.ss1,
             "actions":player.actions,
             "restoreds":player.restoreds1,
              "restore_labels":player.restore_labels1,
             "action_probss":player.action_probss,
             "Q_11s_ext":player.Q_11s_ext,
             "Q_11s_ext_T":player.Q_11s_ext_T,
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
            restoration_loss1, loss_Qext = losses  #g_loss1_newest,  loss_Qint_newest 
            
            losses = list(losses)

            def get_max_with_abs(tensor1d):
                arg = torch.argmax(torch.abs(tensor1d))
                return tensor1d[arg].item()
            mean_Vs1 = torch.mean(torch.Tensor(player.values1)).cpu().numpy()
        
            mean_re1 = float(np.mean(player.rewards1))
            # ii.shape=(1,6) 
            max_Q11_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_11s_ext])) #(20)
            max_Q11_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_11s_ext]))
            max_Q11_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_11s_ext]))

            additional_logs = []
#             print("losses ",losses, flush=True)
            for loss_i in losses:
                if not (loss_i == 0):
                    additional_logs.append(loss_i.item())
                else:
                    additional_logs.append(loss_i)
            
            if logging:
                f = open(STATg_CSV_PATH, 'a')
                writer = csv.writer(f)
                writer.writerow([mean_Vs1, mean_re1, counter.value, local_counter, max_Q11_1, max_Q11_2, max_Q11_3,]+additional_logs) #max_Q21_1, max_Q21_2, max_Q21_3,
                f.close()


            loss_restoration1 = args["Training"]["w_restoration"] * restoration_loss1
#             loss_restoration2 = args["Training"]["w_restoration"] * restoration_loss2
            loss_restoration1.backward(retain_graph=True)
#             (args["Training"]["w_MPDI"]*g_loss1).backward(retain_graph=True)
#             (args["Training"]["w_policy"]*loss_V1).backward(retain_graph=False)
            (args["Training"]["w_policy"]*loss_Qext).backward(retain_graph=True)
#             (args["Training"]["w_policy"]*loss_Qint).backward(retain_graph=False)


        
            
            if len(player.rewards1)>2:
                # try:
                ensure_shared_grads(player.model1, shared_model[0], gpu=gpu_id >= 0)
                # except Exception as e:
                #     print("LOSS lossQext ", args["Training"]["w_policy"]*loss_Qext, flush=True)
                #     print("LOSS loss_restoration1 ", loss_restoration1, flush=True)
                #     print(e)
    #             ensure_shared_grads(player.model2, target_model[1], gpu=gpu_id >= 0)
                optimizer.step()

            
            #copy weights
            if n_batch%80==0:
                synchronize_target_model(gpu_id, target_model, shared_model)
            
            synchronize_player_model(gpu_id, player, shared_model)

            player.clear_state()
            player.model1.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

#             player.model2.zero_grad()
            
#             for p in target_model[1].actor_base2.parameters():
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
def train_A3C_united(batch_dict, gpu_id, QTarget, Target_Qint, s_last1, g_last1, tau, gamma1, w_curiosity, kld_loss_calc, TD_len="max"):
    # Initialize losses
    restoration_loss1 = 0
    loss_Qext = 0
    
    # Process current batch
    T = len(batch_dict["rewards"])
    batch_dict["ss1"].append(s_last1)
    
    RSum = 0
    k = 1
    
    for i in reversed(range(T)):
        if T-i <= k:
            QTarget = QTarget*gamma1 + batch_dict["rewards"][i]
            RSum = RSum*gamma1 + batch_dict["rewards"][i]
        else:
            RSum = RSum*gamma1 + batch_dict["rewards"][i] - (batch_dict["rewards"][i+k]*(gamma1**k))
            QTarget = torch.max(batch_dict["Q_11s_ext_T"][i+k])*(gamma1**k) + RSum
        
        # print('len(batch_dict["Q_11s_ext"])', len(batch_dict["Q_11s_ext"]), flush=True)
        # print('len actions', len(batch_dict["actions"]), flush=True)
        # print("i",i, flush=True)
        # print('len(batch_dict["Q_11s_ext"].shape)', batch_dict["Q_11s_ext"][0].shape, flush=True)
        advantage_ext = QTarget - batch_dict["Q_11s_ext"][i][0][batch_dict["actions"][i].item()]
        advantage_ext = torch.clamp(advantage_ext, -1, 1)
        
        loss_Qext = loss_Qext + (0.5*advantage_ext.pow(2))
        
        restoration_loss1_part = (batch_dict["restoreds"][i] - batch_dict["restore_labels"][i]).pow(2).sum()
        restoration_loss1 += restoration_loss1_part
    
    return restoration_loss1, loss_Qext


# передать Target_Qext за весь батч
def calculate_loss(batch_dict, gpu_id, Target_Qext, Target_Qint, s_last1, g_last1, tau, gamma1, w_curiosity, kld_loss_calc, TD_len="max"):
    """
        QTarget := TarQNet[H] 
        RSum := 0
        For i in in reversed batch range 
            If H-i<=k then
                QTarget := QTarget*gamma + r[i] 
                RSum := RSum*gamma + r[i] 
            else 
                RSum := RSum*gamma + r[i] - r[i+k] * gamma^k
                QTarget := TarQNet[i+k] * gamma^k + RSum
            Advantage := QTarget.detach - Q[i]
    """    

    H = len(batch_dict["rewards"])
    QTarget = Target_Qext[H]
    r = batch_dict["rewards"]
    RSum = 0
    k = 1
    gamma = gamma1
    
    loss_Qext = 0
    restoration_loss1 = 0
    for i in reversed(range(H)):
        if H - i <= k:
            RSum = RSum * gamma + r[i]
            QTarget = QTarget * gamma + r[i]
        else:
            RSum = RSum * gamma + r[i] - r[i + k] * pow(gamma, k)
            QTarget = Target_Qext[i + k] * pow(gamma, k) + RSum

        advantage_ext = QTarget.max().detach() - batch_dict["Q_11s_ext"][i][0][batch_dict["actions"][i].item()]
        
        advantage_ext = torch.clamp(advantage_ext, -1, 1) # ppo clip

        # изменение градиента
        loss_Qext = loss_Qext + (0.5 * advantage_ext.pow(2))
        
        restoration_loss1_part = (batch_dict["restoreds"][i] - batch_dict["restore_labels"][i]).pow(2).sum()
        restoration_loss1 = restoration_loss1 + restoration_loss1_part 
    return restoration_loss1, loss_Qext