import os
import sys
import time

print("WORKING!!!!")
sys.stdout.flush()

#run this in screen? screen for each run? 
#use "nohup python /path/to/test.py &"90

with open("/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/configs/a3c_with_vae.ini", "r") as ff:
    lines = ff.readlines()

for line in lines:
    line=line.strip()
    if "log_dir" in line:
        res_path = line.split('=')[-1].strip()

res_path = "/home/users/aamore/BaseDists_ver_before_sVAE_hevyside3/"+res_path
print("res_path ",res_path)
sys.stdout.flush()

#check if DONE
filename_proc_ids = sys.argv[1]
DONE = False

while DONE==False:
    squeue = os.popen('squeue').read()

    with open(filename_proc_ids,'r') as ff:
        lines = ff.readlines()

    DONE = True

    for proc_id in lines:
        proc_id = proc_id.replace("Submitted batch job ","").strip()
#         print("proc_id ",proc_id)
#         sys.stdout.flush()
        if proc_id in squeue:
            DONE=False
            print(str(proc_id)+" is still running") #TODO add time
            sys.stdout.flush()
    if DONE:
        break
    time.sleep(60*5)
    print("--------------")
    sys.stdout.flush()

# when DONE
# process logs and make onefolder

os.popen('python /home/users/aamore/BaseDists_ver_before_sVAE_hevyside3/logs/parallels_to_one_folder.py '+res_path)
print("Done success")
sys.stdout.flush()
# when onefolder made Draw all always_needed
# and new too (put to special folder?)

#all artifacts. code. and etc. in special archive?

