import os
os.environ["OMP_NUM_THREADS"] = "1"

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
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
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
from agents import Agent, unload_batch_to_cpu
import torch.nn as nn
import copy
    
def train(rank, args, shared_model, optimizer, env_conf, lock, counter, num, main_start_time="unknown_start_time", RUN_KEY="only"):
    ptitle('Training Agent: {}'.format(rank))
    
    try:
        os.mkdir("./"+args["Training"]["log_dir"]+"/stats"+str(main_start_time)+"_"+str(num)+"/")
    except:
        pass
    STATg_CSV_PATH = "./"+args["Training"]["log_dir"]+"/stats"+str(main_start_time)+"_"+str(num)+"/"+"_STATS"+str(rank)+".csv"
    f = open(STATg_CSV_PATH, 'w+')
    f.close()
    
    gpu_id = args["Training"]["gpu_ids"][rank % len(args["Training"]["gpu_ids"])]
    torch.manual_seed(int(args["Training"]["seed"]) + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(int(args["Training"]["seed"]) + rank)
        
    env = atari_env(args["Training"]["env"], env_conf, args)
    env.seed(int(args["Training"]["seed"]) + rank)
    
    model_params_dict = args["Model"]
    _model1 = Level1(args, env.observation_space.shape[0],
                     env.action_space, device="cuda:"+str(gpu_id))
    optimizer = AdamW(
        _model1.parameters(),
        lr=args["Training"]["lr"],
        amsgrad=args["Training"]["amsgrad"]   
    )
    optimizer.zero_grad()
    
    shared_model[0] = shared_model[0].to('cuda')
    player = Agent(_model1, shared_model, env, args, None, gpu_id)
    player.rank = rank
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
        
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model1 = player.model1.cuda()
    player.model1.train()

    train_func = train_A3C_united
    print("-----------USING TRAINFUNC ", train_func)
    
    player.eps_len += 2
    local_counter = 0
    tau = args["Training"]["tau"]
    gamma1 = args["Training"]["initial_gamma1"]
    gamma2 = args["Training"]["initial_gamma2"]
    w_curiosity = float(args["Training"]["w_curiosity"])
    game_num = 0
    frame_num = 0
    future_last = 0
    g_last = torch.Tensor([0])
    
    kld_loss_calc = _kld_loss_calc
    n_batch = 0
    
    while True:
        if n_batch % 80 == 0:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    shared_model[0].load_state_dict(player.model1.state_dict())
                    with lock:
                        torch.save(player.model1.state_dict(), "./current_state_dict" + str(RUN_KEY) + ".torch")
            w = shared_model[0].state_dict()
        if player.done:
            player.reset_lstm_states()
        else:
            player.detach_lstm_states(levels=[1, 2])
        n_batch += 1
        for step in range(args["Training"]["num_steps"]):
            player.action_train()
            with lock:
                counter.value += 1
            local_counter += 1
            if player.done:
                break
        
        if player.done:
            player.eps_len = 0
            game_num += 1
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
            x_restored1, v1_ext, v1_int, Q11_ext, Q11_int, s1, g1 = shared_model[0](Variable(
                state.unsqueeze(0)), player.prev_action_1, player.memory_1, player.prev_s1)
            player.train_episodes_run += 1
            V_last1 = v1_ext.detach()
            Target_Qext = Q11_ext.detach()
            Target_Qint = Q11_int
            
            A_ext = Q11_ext
            A_int = Q11_int
            A = A_ext
            new_batch_dict = {"states": player.states,
                              "rewards": player.rewards1,
                              "ss1": player.ss1,
                              "actions": player.actions,
                              "Q_11s": player.Q_11s,
                              "values": player.values1,
                              "restoreds": player.restoreds1,
                              "restore_labels": player.restore_labels1,
                              "gs1": player.gs1,
             "V_exts": player.V_exts,
             "V_ints": player.V_ints,
                              "action_probss": player.action_probss,
                              "Q_11s_int": player.Q_11s_int,
                              "Q_11s_ext": player.Q_11s_ext,
                              "Q_11s_ext_T": player.Q_11s_ext_T,
                              "memories": player.memory_1s,
                              "prev_g1": torch.zeros((1, 32, 20, 20)).to("cuda:"+str(gpu_id)),
                              "prev_action1": player.first_batch_action}
            
            losses = train_func(new_batch_dict, player.gpu_id, Target_Qext, Target_Qint, s_last1, g_last1, tau, gamma1, w_curiosity, kld_loss_calc, TD_len='max')

            restoration_loss1_newest, loss_Qext_newest = losses
            restoration_loss1 = restoration_loss1_newest
            loss_Qext = loss_Qext_newest

            def get_max_with_abs(tensor1d):
                arg = torch.argmax(torch.abs(tensor1d))
                return tensor1d[arg].item()

            mean_Vs1 = torch.mean(torch.Tensor(player.values1)).cpu().numpy()
            mean_re1 = float(np.mean(player.rewards1))
            max_Q11_1 = get_max_with_abs(torch.Tensor([ii[0][0] for ii in player.Q_11s_ext]))
            max_Q11_2 = get_max_with_abs(torch.Tensor([ii[0][1] for ii in player.Q_11s_ext]))
            max_Q11_3 = get_max_with_abs(torch.Tensor([ii[0][2] for ii in player.Q_11s_ext]))

            additional_logs = []
            for loss_i in losses:
                if not (loss_i == 0):
                    additional_logs.append(loss_i.item())
                else:
                    additional_logs.append(loss_i)
                    
            f = open(STATg_CSV_PATH, 'a')
            writer = csv.writer(f)
            writer.writerow([mean_Vs1, mean_re1, counter.value, local_counter, max_Q11_1, max_Q11_2, max_Q11_3,] + additional_logs)
            f.close()
            
            loss_restoration1 = args["Training"]["w_restoration"] * restoration_loss1
            loss_restoration1.backward(retain_graph=True)
            (args["Training"]["w_policy"] * loss_Qext).backward(retain_graph=False)
            if len(player.rewards1) > 2:
                optimizer.step()
            player.clear_actions()
            player.model1.zero_grad()
            
            torch.cuda.empty_cache()
            gc.collect()

def MPDI_loss_calc1(batch_dict, g_last1, tau, gamma1, adaptive, i, advantage_ext):
    try:
        g_last1 = g_last1 * gamma1 + (batch_dict["ss1"][i+1].detach() - batch_dict["ss1"][i].detach())
        g_advantage1 = F.cosine_similarity(g_last1.ravel(), batch_dict["gs1"][i].ravel(), dim=0)
        return g_last1, -g_advantage1 * advantage_ext.detach(), g_advantage1.detach()
    except Exception as e:
        return g_last1, (g_last1 - batch_dict["gs1"][0]).sum() * 0

def MPDI_loss_calc2(player, V_last2, g_last2, tau, gamma2, adaptive, i):
    try:
        g_last2 = g_last2 * gamma2 + (1 - gamma2) * player.ss2[i+1].detach()
        g_advantage2 = g_last2 - player.gs2[i]
        return g_last2, 0.5 * g_advantage2.pow(2).sum()
    except Exception as e:
        return g_last2, (g_last2 - player.gs2[0]).sum() * 0
    
def _kld_loss_calc(player, i):
    return player.klds1[i], player.klds2[i]

def _kld_loss_calc_filler(player, i):
    return 0

def get_pixel_change(pic1, pic2, STEP=20):
    max_side = pic1.shape[-1]
    res = torch.zeros((max_side // STEP, max_side // STEP))
    for n1, i in enumerate(range(0, max_side, STEP)):
        for n2, j in enumerate(range(0, max_side, STEP)):
            res[n1, n2] = (pic1[:, i:i+STEP, j:j+STEP] - pic2[:, i:i+STEP, j:j+STEP]).mean().abs()
    return res

    batch_dict["Q_11s_int"].append(Target_Qint)
    
batch_dict["V_exts"].append((batch_dict["action_probss"][-1] * batch_dict["Q_11s_ext"][-1]).sum())
batch_dict["V_exts"].append((batch_dict["action_probss"][-1] * batch_dict["Q_11s_ext"][-1]).sum())
            
    Target_Qext = Target_Qext.max().detach()
batch_dict["V_exts"].append((batch_dict["action_probss"][-1] * batch_dict["Q_11s_ext"][-1]).sum())
batch_dict["V_ints"].append((batch_dict["action_probss"][-1] * batch_dict["Q_11s_int"][-1]).sum())
    Target_Qint = 0
batch_dict["V_ints"].append((batch_dict["action_probss"][-1] * batch_dict["Q_11s_int"][-1]).sum())
kld_loss1 = 0
Target_Qext = Target_Qext.max().detach()
Target_Qint = 0

Target_Qext = Target_Qext.max().detach()
Target_Qint = 0
loss_Qint = 0
loss_Qext = 0
    
restoration_loss1 = 0
    restoration_loss1 = 0
    batch_dict["ss1"].append(s_last1)
loss_Qext = 0
    loss_Qext = 0
    QTarget = Target_Qext.max().detach()
    RSum = 0
        k = 1

for i in reversed(range(T)):
    k = 1
        if T-i <= k:
        QTarget = torch.max(batch_dict["Q_11s_ext_T"][i+k]) * (gamma1**k) + RSum
            
        advantage_ext = QTarget.detach() - batch_dict["Q_11s_ext"][i][0][batch_dict["actions"][i].item()]
        advantage_ext = torch.clip(advantage_ext, -1, 1)
    loss_Qext = loss_Qext + (0.5 * advantage_ext.pow(2))
        
    return restoration_loss1, loss_Qext

restoration_loss1 += restoration_loss1_part

return restoration_loss1, loss_Qext

return restoration_loss1, loss_Qext