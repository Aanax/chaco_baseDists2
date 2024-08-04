# main. will be rewrited to run from config files. not console args

from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env, mujoco_env
from utils import read_config
from models import *#A3Clstm, StarA3C
from test import *#test, testSTAR
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import time
import configparser
from typing import Dict
from train import *        
from time import gmtime, strftime

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

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

        
if __name__ == '__main__':
    args = configparser.ConfigParser()
    args.optionxform = lambda option: option
    args.read(sys.argv[1])#"./configs/star_config.ini"
    
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
    
    
    torch.manual_seed(args["Training"]["seed"])
    if args["Training"]["gpu_ids"] == -1:
        args["Training"]["gpu_ids"] = [-1]
    else:
        torch.cuda.manual_seed(args["Training"]["seed"])
        mp.set_start_method('spawn')
        
    setup_json = read_config(args["Training"]["env_config"])
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args["Training"]["env"]:
            env_conf = setup_json[i]
            
    
    env = atari_env(args["Training"]["env"], env_conf, args)
    
    model_params_dict = args["Model"]
    shared_model = torch.nn.Sequential(Level1(args, env.observation_space.shape[0], env.action_space, device="cpu"), Level2(args, env.observation_space.shape[0], env.action_space, device="cpu"))
        
    shared_model.share_memory()

    optimizer = SharedAdam(
        [
        {'params': shared_model[0].parameters(), 'lr': args["Training"]["lr"]}, #0.001},
        {'params': shared_model[1].parameters(), 'lr': args["Training"]["lr"]},], #*0.05
        lr=args["Training"]["lr"], amsgrad=args["Training"]["amsgrad"]   
        )
#     optimizer_decoders = SharedAdam(
#         shared_model.Decoder.parameters(), lr=args["Training"]["lr"], amsgrad=args["Training"]["amsgrad"])
    

#     optimizer = SharedAdam(
#         [
#         {'params': shared_model.Encoder.parameters(), 'lr': float(args["Training"]["lr"])},
#         {'params': shared_model.A3C.parameters(),'lr': float(args["Training"]["lr"])},
#         {'params': shared_model.Decoder.parameters(), 'lr': float(args["Training"]["lr_decoders"])},
#         {'params': shared_model.Decoder2.parameters(), 'lr': float(args["Training"]["lr_decoders"])},
#         ],
#         lr=args["Training"]["lr"], amsgrad=args["Training"]["amsgrad"])
            
    optimizer.share_memory()
 
        
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    processes = []
 
    
    main_start_time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    
    try:
        parallel_running_num = sys.argv[2]
    except:
        parallel_running_num = 1234
    
    p = mp.Process(target=test, args=(args, shared_model, env_conf, counter, parallel_running_num))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, int(args["Training"]["workers"])):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf, lock, counter, parallel_running_num, main_start_time))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
