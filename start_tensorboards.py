import os
import subprocess
root = "/home/sebastiankoch/SoftActorCriticRNN/"
files = os.listdir(root)
# print(files)
dirs = [x for x in files if (("runs" in x) and os.path.isdir(root+x) and ("csv_runs" not in x ))]
# print(dirs)
# command = "tensorboard --logdir_spec="
for dir_ in dirs:
    pass
#     command+=dir_+":"+root+dir_+","
    subprocess.Popen(['interminal', 'echo','hello'])

# print(command)
# os.system(command)