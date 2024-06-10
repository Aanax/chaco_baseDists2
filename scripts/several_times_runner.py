import argparse
import subprocess
import time
import os
import signal
import psutil
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# base_log_name = sys.argv[2]
config_path = sys.argv[1]

n_runs = 3

for i in range(n_runs):

    print("___________________::RUN " + str(i) + "::________________________")
    # do stuff
    #     print("command like :",args_list)
    proc = subprocess.Popen(["python", "main.py"] + [config_path])
    time.sleep(9000)  # <2hour #9000)#2.5hour #11000) #~3hours

    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        print("Child pid is {}".format(child.pid))
        os.kill(child.pid, signal.SIGINT)

    time.sleep(5)

    try:
        proc.terminate()
        time.sleep(5)
    except:
        pass
