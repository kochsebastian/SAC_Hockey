import numpy as np
# maybe use pip install EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import seaborn as sns; sns.set()

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


def plot(data_sets, title, algorithm, label, dir):
    fig = plt.figure(figsize=(15,8))
    plt.ticklabel_format(style='sci', axis='x',useOffset=False, scilimits=(0,0))
    for idx, data in enumerate(data_sets):
        ax = sns.lineplot(x="Environment Steps", y=label, data=data, label=algorithm[0][idx], ci= 90, n_boot=1000, err_style = 'band')

    plt.title(title, fontsize=15)
    ax.yaxis.label.set_size(15)
    ax.xaxis.label.set_size(15)

    plt.legend(loc='lower right',fontsize=15)
    plt.savefig(dir+title+".png", dpi=300)
    plt.show()

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
    parser.add_argument("-sd", "--savedir", type=str, default="", help="Save directory, default current directory")
    args = parser.parse_args()

    #print(args)
    num_alg = len(args.algorithm[0])
    assert len(args.logdir[0]) % num_alg == 0, "Algorithm need the same amount of training runs!"
    assert len(args.label[0]) ==  len(args.title[0]), "Not enough titles for the plots. If you compare more than one label you need different titles for each plot!"

    dirs = args.logdir
    if num_alg > 1 and len(args.logdir[0]) != 2:
        #print(num_alg)
        dirs = chunks(args.logdir[0], n= int(len(args.logdir[0])/num_alg))
        #print(dirs)
    elif num_alg > 1 and len(args.algorithm[0]) == 2 and len(args.logdir[0]) == 2:
        dirs = [[args.logdir[0][0]],[args.logdir[0][1]]]


    for i in range(len(args.label[0])):
        print("Process Label: ", args.label[0][i])
        data_per_label = []
        for j in range(num_alg):
            data_log = get_tensorflow_log(log_dirs=dirs[j], label=args.label[0][i])
            dataset = create_dataset(data_log, args.label[0][i])
            data_per_label.append(dataset)
        
        plot(data_per_label, args.title[0][i], args.algorithm, args.label[0][i], args.savedir)
