import os
import subprocess
root ="/home/skoch/SAC_Hockey/"
files = os.listdir(root)
# print(files)

dirs = [x for x in files if ("runs" in x and os.path.isdir(root+x) and ("csv_runs" not in x) )]

command ="tensorboard --logdir_spec="
for dir_ in dirs:
    command += dir_+":"+root+dir_+","
    # os.system(command)
    # p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # out = p.communicate()[0]
os.system(command[:-1])