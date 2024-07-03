from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import atari_env_eval
from utils import read_config, setup_logger
from models import *
from agents import * #Agent
import gym
import logging
import time
import sys
import imageio
#from gym.configuration import undo_logger_setup
import configparser
import datetime
from typing import Dict
import cv2

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_dict(config: configparser.ConfigParser) -> Dict[str, Dict[str, str]]:
    """
    function converts a ConfigParser structure into a nested dict
    Each section name is a first level key in the the dict, and the key values of the section
    becomes the dict in the second level
    {
        'section_name': {
            'key': 'value'
        }
    }
    :param config:  the ConfigParser with the file already loaded
    :return: a nested dict
    """
    return {section_name: dict(config[section_name]) for section_name in config.sections()}

        
args = configparser.ConfigParser()
args.optionxform = lambda option: option
args.read("/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/configs/a3c_with_vae.ini")#sys.argv[1])#"./configs/star_config.ini"

print(args.sections())
print([i for i in args.keys()])
## TODO make this into argparser
print(args["Training"]["load"])
print(type(args["Training"]["load"]))
args = to_dict(args)
print(args)

args["Training"]["load"] = str2bool(args["Training"]["load"])
#print("inside argsload ",args.load)
args["Training"]["shared_optimizer"] = str2bool(args["Training"]["shared_optimizer"])
args["Training"]["amsgrad"] = str2bool(args["Training"]["amsgrad"])
args["Training"]["save_max"] = str2bool(args["Training"]["save_max"])
args["Training"]["adaptive_gamma_and_eps"] = str2bool(args["Training"]["adaptive_gamma_and_eps"])
#     args["Model"]["merged_T_and_A"] = str2bool(args["Model"]["merged_T_and_A"])
args["Training"]["gpu_ids"] = list(map(int,args["Training"]["gpu_ids"].split(",")))

args["Model"]["hidden_dim_lstm1"] = int(args["Model"]["hidden_dim_lstm1"])
args["Model"]["hidden_dim_lstm2"] = int(args["Model"]["hidden_dim_lstm2"])
args["Model"]["s1_dim"] = int(args["Model"]["s1_dim"])
args["Model"]["s2_dim"] = int(args["Model"]["s2_dim"])
args["Model"]["S1_dim"] = int(args["Model"]["S1_dim"])
args["Model"]["S2_dim"] = int(args["Model"]["S2_dim"])
#     args["Model"]["hidden_dim_A"] = int(args["Model"]["hidden_dim_A"])
#     args["Model"]["hidden_dim_R"] = int(args["Model"]["hidden_dim_R"])
args["Training"]["max_episode_length"] = int(args["Training"]["max_episode_length"])
args["Training"]["lr"] = float(args["Training"]["lr"])
args["Training"]["tau"] = float(args["Training"]["tau"])
args["Training"]["w_kld"] = float(args["Training"]["w_kld"])
args["Training"]["w_MPDI"] = float(args["Training"]["w_MPDI"])
args["Training"]["w_value"] = float(args["Training"]["w_value"])
args["Training"]["w_policy"] = float(args["Training"]["w_policy"])
args["Training"]["w_restoration"] = float(args["Training"]["w_restoration"])
args["Training"]["w_restoration_future"] = float(args["Training"]["w_restoration_future"])
args["Training"]["num_steps"] = int(args["Training"]["num_steps"])
args["Training"]["skip_rate"] = int(args["Training"]["skip_rate"])
#     args["Training"]["initial_eps"] = float(args["Training"]["initial_eps"])
args["Training"]["initial_gamma1"] = float(args["Training"]["initial_gamma1"])
args["Training"]["initial_gamma2"] = float(args["Training"]["initial_gamma2"])

args["Model"]["v_init_std"] = float(args["Model"]["v_init_std"])
args["Model"]["a_init_std"] = float(args["Model"]["a_init_std"])
args["Model"]["S_init_std_multiplier"] = float(args["Model"]["S_init_std_multiplier"])
args["Model"]["lstm_init_gain"] = float(args["Model"]["lstm_init_gain"])

# python eval.py --env PongDeterministic-v4 --gpu-id 1 --agent A3Clstm --model_path "./trained_models/PongDeterministic-v4_base__21.0.dat"


def saveanimation(frames,address="./movie_base.gif"):
    """ 
    This method ,given the frames of images make the gif and save it in the folder
    
    params:
        frames:method takes in the array or np.array of images
        address:(optional)given the address/location saves the gif on that location
                otherwise save it to default address './movie.gif'
    
    return :
        none
    """
    imageio.mimsave(address, frames, fps=5)





# args = parser.parse_args()
setup_json = read_config(args["Training"]["env_config"])
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args["Training"]["env"]:
        env_conf = setup_json[i]

gpu_id = 0

torch.manual_seed(args["Training"]["seed"])

if gpu_id >= 0:
    torch.cuda.manual_seed(args["Training"]["seed"])

# model_path = "./trained_models/PongDeterministic-v4logs_a3c_united_FIX5_vaeMPDI_non_restricted_wmpdi05_eps_0.0_10__21.0.dat"
model_path = "/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/trained_models/Pong-v0logs_CHACO_f_v31_NDOkNoEnt_Pongv0_v1D1_V1adDel2_clas2a2smean_a2gam2_gaeModulActors_004kld_1lvlAbaseInLstm_nod2_noZEMA_0a1_envrV2_fix_fixPow_01kl2_noSamplA2_detS2_eps_0.0_3__-19.0.dat"

# Pong-v0logs_CHACO_f_v31_NDOkNoEnt_Pongv0_b_dist_actor2kld_runmeanLvl1_fix_disbAC_00Mot_fix_gaeModul_32actor2_v1D1_fix3_demin2_V1adDel2_kldGaeModul_sa2runmen_fix_nossep_eps_0.0_5__2.0.dat"


#Pong-v0logs_CHACO_f_v31_NDOkNoEnt_Pongv0_b_dist_actor2kld_runmeanLvl1_fix_disbAC_00Mot_fix_gaeModul_32actor2_v1D1_fixVrunmen_fix3_demin2_V1adDel2_kldGaeModul_eps_0.0_7__9.0.dat"

# Pong-v0logs_CHACO_fixed_v31_fixedA2inits_NormDistOkNoEntsSumProbs_Pongv0_base_dist_from_zeroed_a2_fix_logprobs4hand_long_actor2kld_abaseLossFix_runnningmeanLvl1_fix_disbalancedAC_00Motiv_fix_gaeModulated_32actor2_eps_0_vaemodul.0_3__9.0.dat"

saved_state = torch.load(
    model_path,
    map_location=lambda storage, loc: storage)

# log = {}
# setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
#     args.log_dir, args.env))
# log['{}_mon_log'.format(args.env)] = logging.getLogger('{}_mon_log'.format(
#     args.env))

# d_args = vars(args)
# for k in d_args.keys():
#     log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = atari_env_eval("{}".format(args["Training"]["env"]), env_conf, args)
num_tests = 0
start_time = time.time()
reward_total_sum = 0


# model_params_dict = args["Model"]
# shared_model = A3C_united(model_params_dict, env.observation_space.shape[0], env.action_space, device="cpu")



#creating Agent (wrapper around model capable of train step and test step making)------------
model_params_dict = args["Model"]
_model1 = Level1(args, env.observation_space.shape[0],
                       env.action_space, device = "cuda:"+str(gpu_id))
_model2 = Level2(args, env.observation_space.shape[0],
                       env.action_space, device = "cuda:"+str(gpu_id))

player = Agent(_model1, _model2, env, args, None, gpu_id)
# player.rank = rank

player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model1 = player.model1.cuda()
        player.model2 = player.model2.cuda()
# if args.new_gym_eval:
#     player.env = gym.wrappers.Monitor(
#         player.env, "{}_monitor".format(args.env), force=True)

d1 = saved_state.__class__({key[2:]: value for (key, value) in saved_state.items() if key[0]=='0'})
d2 = saved_state.__class__({key[2:]: value for (key, value) in saved_state.items() if key[0]=='1'})

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model1.load_state_dict(d1)
        player.model2.load_state_dict(d2)
else:
    player.model1.load_state_dict(d1)
    player.model2.load_state_dict(d2)

player.model1.eval()
player.model2.eval()


import os
LOGSFOLDER = './Eval_'+str(datetime.datetime.now()).replace(' ','_')+'/'
os.makedirs(LOGSFOLDER, exist_ok=True)

render=True
render_freq = 1
watch_z = True
zs = []
print("Starting loop ...", flush=True)
for i_episode in range(1):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    player.eps_len += 2
    reward_sum = 0
    print("Episode "+str(i_episode))
    frames=[]
    frames_normalized = []
    frames_normalized_orig = []
    frames_restored = []
    frames_restored_thr = []
    frames_from_lstm = []
    frames_from_lstm_thr = []
    frames_from_lstm_ranged = []
    Ss = []
    Vs = []
    ss = []
    aas = []
    Ss2 = []
    Vs2 = []
    ss2 = []
    aas2 = []
    aas_base = []
    aas1=[]
    restoreds2 = []
    hxs1=[]
    rewards = []
    deltas1=[]
    deltas2=[]
    
    gamma1 = args["Training"]["initial_gamma1"]
    gamma2 = args["Training"]["initial_gamma2"]
    while True:
        print("Working. frames collected = ",len(frames),flush=True)
        

        player.action_test(ZERO_ABASE=False)
        reward_sum += player.reward

        if render:
            if i_episode % render_freq == 0:
                frames.append(player.env.render(mode = 'rgb_array'))
                
                # logging preprocessed image that was on input-----
#                 print(player.original_state, flush=True)
#                 print(player.original_state.shape, flush=True)
                x_orig = player.original_state
                frames_normalized_orig.append(x_orig)
                img_orig = (np.rollaxis(x_orig,0,3)*player.env.unbiased_std + player.env.unbiased_mean).astype("uint8")
    #             print("img_rest shape ", img_rest.shape, flush=True)
                frames_normalized.append(img_orig)
        
        
#                 x_orig_lstm = player.restored_after_lstm.detach().cpu().numpy()[0]
#                 x_orig = x_orig_lstm*player.env.unbiased_std + player.env.unbiased_mean
                
#                 OldMin = np.min(x_orig)
#                 OldMax = np.max(x_orig)
#                 NewMin = 0
#                 NewMax = 255
#                 OldRange = (OldMax - OldMin)  
#                 NewRange = (NewMax - NewMin)  
#                 decode_last = ((((x_orig - OldMin) * NewRange) / OldRange) + NewMin)
#                 decode_last = (np.rollaxis(decode_last,0,3)).astype("uint8")
#                 frames_from_lstm_ranged.append(decode_last)
                
#                 img_orig = (np.rollaxis(x_orig,0,3)).astype("uint8")
#                 _,thresholded = cv2.threshold(img_orig,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#                 frames_from_lstm.append(img_orig)
#                 frames_from_lstm_thr.append(thresholded)
                
                

                #---------------------------------------------------
                # logging restored image
                x_rest = player.restored_state.detach().cpu().numpy()[0]
                x_rest_ = x_rest*player.env.unbiased_std + player.env.unbiased_mean#127.5 
                img_rest = (np.rollaxis(x_rest_,0,3)).astype("uint8")
                _,thresholded_o = cv2.threshold(img_rest,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                frames_restored.append(img_rest)
                frames_restored_thr.append(thresholded_o)
                
                
                
                
                #---------------------------------------------------
        if watch_z:
#             z, z_mean, z_log_var, kl = player.model.Encoder(Variable(
#                 player.state.unsqueeze(0)))
#             zs.append(z.detach().cpu().numpy())
            
            
            rewards.append(player.reward)
            Ss.append(player.last_S.detach().cpu().numpy())
            Vs.append(player.last_v.detach().cpu().numpy())
            ss.append(player.last_s.detach().cpu().numpy())
            aas.append(player.last_a.detach().cpu().numpy())
            
            aas1.append(player.last_a1.detach().cpu().numpy())
            aas_base.append(player.last_abase.detach().cpu().numpy())


            
            Ss2.append(player.last_S2.detach().cpu().numpy())
            Vs2.append(player.last_v2.detach().cpu().numpy())
            ss2.append(player.last_s2.detach().cpu().numpy())
            aas2.append(player.last_a2.detach().cpu().numpy())
            restoreds2.append(player.restored_state2.detach().cpu().numpy())
            hxs1.append(player.hx1.detach().cpu().numpy())
            
            if len(Vs)>=2:
                delta_t2 = (1-gamma1)*Vs[-2] + gamma2 * \
                Vs2[-1] - Vs2[-2]
                deltas2.append(delta_t2)
                ##+ delta_t2 ?
                delta_t1 = rewards[-2] + gamma1 * \
                    Vs[-1] - Vs[-2]
                deltas1.append(delta_t1)
        
        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
#             log['{}_mon_log'.format(args.env)].info(
#                 "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
#                 format(
#                     time.strftime("%Hh %Mm %Ss",
#                                   time.gmtime(time.time() - start_time)),
#                     reward_sum, player.eps_len, reward_mean))
            model_path = model_path.split("/")[-1][40:]
            print("DONE. writing animations", flush=True)
            player.eps_len = 0
            saveanimation(frames,address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"render_vae_dis.mp4")
            saveanimation(frames_normalized, address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"normalized_vae_dis.mp4")
            saveanimation(frames_restored, address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"restored_vae_dis.mp4")
            saveanimation(frames_restored_thr, address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"restored_vae_THRESH_dis.mp4")
            
            saveanimation(frames_from_lstm, address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"from_lstm_vae_dis.mp4")
            saveanimation(frames_from_lstm_thr, address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"from_lstm_THRESH_vae_dis.mp4")
            saveanimation(frames_from_lstm_ranged, address=LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"from_lstm_RANGED.mp4")
            
            
            if watch_z:
#                 with open("./"+model_path.split("/")[-1].split(".")[0]+"zs.npy", 'wb') as f:
#                     np.save(f, np.array(zs))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"Frames_normalized_orig.npy", 'wb') as f:
                    np.save(f, np.array(frames_normalized_orig))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"hx1.npy", 'wb') as f:
                    np.save(f, np.array(player.hx1.detach().cpu().numpy()))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"hxs1.npy", 'wb') as f:
                    np.save(f, np.array(hxs1))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"deltas1.npy", 'wb') as f:
                    np.save(f, np.array(deltas1))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"deltas2.npy", 'wb') as f:
                    np.save(f, np.array(deltas2))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"hx2.npy", 'wb') as f:
                    np.save(f, np.array(player.hx2.detach().cpu().numpy()))
#                 with open("./"+model_path.split("/")[-1].split(".")[0]+"AFTER_LSTM_decode.npy", 'wb') as f:
#                     np.save(f, x_orig_lstm)
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"just_decode.npy", 'wb') as f:
                    np.save(f, x_rest)
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"original_normalized_state.npy", 'wb') as f:
                    np.save(f, player.original_state)
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"original_render_state.npy", 'wb') as f:
                    np.save(f, frames[-1])
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"unbiased_mean.npy", 'wb') as f:
                    np.save(f, env.unbiased_mean)
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"unbiased_std.npy", 'wb') as f:
                    np.save(f, env.unbiased_std)
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"Ss.npy", 'wb') as f:
                    np.save(f, np.array(Ss))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"ss.npy", 'wb') as f:
                    np.save(f, np.array(ss))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"Vs.npy", 'wb') as f:
                    np.save(f, np.array(Vs))
                
                    
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"Ss2.npy", 'wb') as f:
                    np.save(f, np.array(Ss2))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"ss2.npy", 'wb') as f:
                    np.save(f, np.array(ss2))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"Vs2.npy", 'wb') as f:
                    np.save(f, np.array(Vs2))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"aas2.npy", 'wb') as f:
                    np.save(f, np.array(aas2))
                
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"aas1.npy", 'wb') as f:
                    np.save(f, np.array(aas1))
                
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"aas_base.npy", 'wb') as f:
                    np.save(f, np.array(aas_base))
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"aas.npy", 'wb') as f:
                    np.save(f, np.array(aas))
                    
                with open(LOGSFOLDER+model_path.split("/")[-1].split(".")[0]+"rewards.npy", 'wb') as f:
                    np.save(f, np.array(rewards))
                    
                    
            break