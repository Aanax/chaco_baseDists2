import os
import sys

filename_proc_ids = sys.argv[1]

squeue = os.popen('squeue').read()

#print(squeue)

with open(filename_proc_ids,'r') as ff:
    lines = ff.readlines()

RESULT = True

for proc_id in lines:
    proc_id = proc_id.replace("Submitted batch job ","")
    if proc_id in squeue:
        RESULT=False


print(RESULT)
