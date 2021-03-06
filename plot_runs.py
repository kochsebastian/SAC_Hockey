import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import seaborn as sns; sns.set()
import csv
import os
import time

f = 0
def get_csv_log(log_dirs):
    steps, values = [], []
    data = {}
    # for idx, path in enumerate(log_dirs):
    reader = csv.reader(open(log_dirs, 'r'))
    
    for row in reader:
        wall_time, step, value = row
        steps.append(step)
        values.append(value)
    steps.pop(0)
    values.pop(0)
    data["steps"] = steps
    data["values"] = values
    return data

def get_tensorflow_log(log_dirs: list, label: str):
    """Returns log files for one label"""


    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    steps, values = [], []
    for idx, path in enumerate(log_dirs):
        event_acc = EventAccumulator(path, tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file
        #print(event_acc.Tags())
        assert label in event_acc.Tags()["scalars"], "Selected label: {} does not exist in the list of selectable labels:\n {}".format(label, event_acc.Tags()["scalars"])

        # get data by label
        d =   event_acc.Scalars(label)
        #data[label+"_"+str(idx)] = d
        
        for i in range(len(d)):
            steps.append(d[i][1])
            values.append(d[i][2])
    data = {}

    data["steps"] = steps
    data["values"] = values
    return data

def create_dataset(data: dict, label:str):
    d = {'Environment Steps': np.hstack(data["steps"]), label :np.hstack(data["values"])}
    data = pd.DataFrame(data=d)
    return data

def smooth(scalars , weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def swap(data_sets, algorithm, i, j):
    data_sets[i], data_sets[j] = data_sets[j], data_sets[i]
    algorithm[0][i], algorithm[0][j] = algorithm[0][j], algorithm[0][i]
    return data_sets,algorithm
    
def plot(data_sets, title, algorithm, label, dir,xlimit=31000):
    # data_sets, algorithm = swap(data_sets,algorithm,1,2)
    # data_sets, algorithm = swap(data_sets,algorithm,1,3)
    # data_sets, algorithm = swap(data_sets,algorithm,1,2)


    plot_i(data_sets, title, algorithm, label, dir,xlimit,8)
    plot_i(data_sets, title, algorithm, label, dir,xlimit,15)
    plt.show()
    # time.sleep(3)
    # plt.close()
    
def plot_i(data_sets, title, algorithm, label, dir,xlimit,figs):
    global f
    f+=1
    fig = plt.figure(f,figsize=(8,8))
    # plt.clf()
    # plt.subplot(111)
    ax1 = plt.gca()
    
    plt.ticklabel_format(style='sci', axis='x',useOffset=False, scilimits=(0,0))
    max_ = -100000
    # colors =['crimson','lime','deepskyblue','magenta','darkviolet','darkorange','yellow','chocolate']
    for idx, data in enumerate(data_sets):
        plt.figure(f)
        data=data.astype(float)
        if xlimit!=None:
            data = data.drop(data[data.values[...,0] > xlimit].index)
        
        smoothed = data.copy()
        smoothed.values[...,1] = smooth(smoothed.values[...,1],0.9)
        max_ = max(np.amax(data.values[...,1]),max_)

        std = np.std(rolling_window(data.values[...,1], 30), axis=-1)


        color = next(ax1._get_lines.prop_cycler)['color']
        # color = colors[idx]
        ax = data.plot(x='Environment Steps', y=label,alpha=0.3,color=color,label='',figsize=(figs, 8),ax = ax1)
        smoothed.plot(x='Environment Steps', y=label,alpha=1.0,color=color,ax = ax1,label=algorithm[0][idx],linewidth=2.0)
        
        # color = next(ax1._get_lines.prop_cycler)['color']
        plt.fill_between(data.values[...,0], smoothed.values[...,1]-std, smoothed.values[...,1]+std,color=color,alpha=0.1)
        plt.fill_between(data.values[...,0], smoothed.values[...,1]-std, smoothed.values[...,1]+std,color=color,alpha=0.1)
    
    extratick = [max_]
    plt.yticks(list(plt.yticks()[0])[1:-1]+extratick)
    ax.set_ylabel("avg reward")
    if figs==8:
        fsize=15
    else:
        fsize=15
    plt.title(title, fontsize=fsize)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.yaxis.label.set_size(fsize)
    ax.xaxis.label.set_size(fsize)

    plt.legend(loc='lower right',fontsize=15)
    plt.ylim(bottom=-200) 
    
    if figs==8:
        plt.savefig(dir+title+'_square.svg', format='svg')
    else:
        plt.savefig(dir+title+'.svg', format='svg')
    # plt.show()

def chunks(l, n):
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i+n])
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a", "--algorithm", type=str, action="append", nargs="+", help="The name algorithm(s) you want to plot")
    parser.add_argument("-dir", "--logdir", type=str, action='append', nargs='+', help="Path to the run(s) you want to plot, for each algorithm same amount of runs!")
    parser.add_argument("-l", "--label", type=str, action='append', nargs='+', help="Label you want to plot, for example Reward")
    parser.add_argument("-t", "--title", type=str, action='append', nargs='+', help="Title of the plot")
    parser.add_argument("-sd", "--savedir", type=str, default="plots/", help="Save directory, default current directory")
    parser.add_argument("--xlim",  type=int, default=1000000000, help="Largest xs")
    
    args = parser.parse_args()

    #print(args)
    num_alg = len(args.algorithm[0])
    subdirs = sorted(os.listdir(args.logdir[0][0]))
    subdirs_filtered = [d for d in subdirs if ".csv" in d]
    dirs = [args.logdir[0][0]+d for d in subdirs_filtered]
    assert len(dirs) % num_alg == 0, "Algorithm need the same amount of training runs!"
    assert len(args.label[0]) ==  len(args.title[0]), "Not enough titles for the plots. If you compare more than one label you need different titles for each plot!"

    # dirs = [dirs]
    # if num_alg > 1 and len(dirs) != 2:
    #     #print(num_alg)
    #     dirs = chunks(dirs, n= int(len(dirs)/num_alg))
    #     #print(dirs)
    # elif num_alg > 1 and len(args.algorithm[0]) == 2 and len(dirs) == 2:
    #     dirs = [[dirs[0]],[dirs[1]]]


    for i in range(len(args.label[0])):
        print("Process Label: ", args.label[0][i])
        data_per_label = []
        for j in range(num_alg):
            # data_log = get_tensorflow_log(log_dirs=dirs[j], label=args.label[0][i])
            data_log2 = get_csv_log(dirs[j])
            # dataset = create_dataset(data_log, args.label[0][i])
            dataset2 = create_dataset(data_log2, args.label[0][i])
            # data_per_label.append(dataset)
            data_per_label.append(dataset2)
        
        plot(data_per_label, args.title[0][i], args.algorithm, args.label[0][i], args.savedir,xlimit=args.xlim)
