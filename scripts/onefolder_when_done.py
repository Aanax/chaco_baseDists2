import os
import sys
import time

#run this in screen? screen for each run? 
#use "nohup python /path/to/test.py &"90

with open("/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/configs/a3c_with_vae.ini", "r") as ff:
    lines = ff.readlines()

for line in lines:
    line=line.strip()
    if "log_dir" in line:
        res_path = line.split('=')[-1].strip()

res_path = "/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/"+res_path
print("res_path ",res_path)


#check if DONE
filename_proc_ids = sys.argv[1]
DONE = False

while DONE==False:
    time.sleep(60*5)
    squeue = os.popen('squeue').read()

    with open(filename_proc_ids,'r') as ff:
        lines = ff.readlines()

    DONE = True

    for proc_id in lines:
        proc_id = proc_id.replace("Submitted batch job ","")
        if proc_id in squeue:
            DONE=False

# when DONE
# process logs and make onefolder

os.popen('python /s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/parallels_to_one_folder.py '+res_path)

# when onefolder made Draw all always_needed
# and new too (put to special folder?)

#all artifacts. code. and etc. in special archive?

