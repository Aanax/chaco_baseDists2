import argparse
import subprocess
import time
import os
import signal
import psutil
import sys
import configparser
from pathlib import Path
import random

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
        
        
            #base_log_name = sys.argv[2]
config_path = sys.argv[1]

args = configparser.ConfigParser()
args.optionxform = lambda option: option
args.read(sys.argv[1])#"./configs/star_config.ini"

#float(args["Training"]["initial_eps"])

    
for eps in [0.0]: #0.1, 0.2, 0.3, 0.5]: #0.0, 0.01, 0.03, 0.05, 0.075, 
    args["Training"]["initial_eps"] = str(eps)
    args["Training"]["log_dir"] = args["Training"]["log_dir"].split("eps")[0] + "eps_" + str(eps) + "_" +str(sys.argv[2])+ "/"
    print("DIR I GOT ",args["Training"]["log_dir"])
    Path(args["Training"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(args["Training"]["log_dir"]+"/stats/").mkdir(parents=True, exist_ok=True)
    
    config_path = './temporary_eps_config'+str(os.getpid())+'_'+str(random.randint(0,100000))+'.ini'
    with open(config_path, 'w') as configfile:    # save
                args.write(configfile)
           
    
    
    print("___________________::RUN::________________________")  
        # do stuff
    #     print("command like :",args_list)
    proc = subprocess.Popen(["python", "/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/src/main.py"] + [config_path] + [sys.argv[2]])


current_process = psutil.Process()
children = current_process.children(recursive=True)
time.sleep(9000*10) #<2hour #9000)#2.5hour #11000) #~3hours
for child in children:
    print('Child pid is {}'.format(child.pid))
    os.kill(child.pid, signal.SIGINT)
for i in range(10):
    os.system("nvidia-smi")
    time.sleep(5)  

    
  
