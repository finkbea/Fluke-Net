import os
import csv
from statistics import mean, stdev
#from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calc_AP_5(class_predictions, true_class):
    """
    For calculating the AP@5 score for the predictions on a single 
      datapoint. Averaging these scores over our dev set will 
      give us the full MAP@5 score

    class_predictions: 5 most confident predictions, ordered
      most confident -> least confident

    true_class: True class of the predicted image
    """

    assert(len(class_predictions)==5)
    
    # Search for true_class in class_predictions
    true_class_location = None
    for i in range(5):
        if class_predictions[i] == true_class:
            true_class_location = i+1
            break
    
    # Average Precision if true_class was not found
    AP_5 = 0
    
    # If true_class was found, AP@5 is the precision at that point
    if not true_class_location is None:
        AP_5 = float(1)/true_class_location
    return AP_5

def parse_filter_specs(filter_specs):
    """
    For parsing a filter into its specifications
    Included here because of the way our pipeline works
    """
    specs = []
    for layer in filter_specs.split(','):
        C,K,M=layer.split('x')
        specs.append([int(C),int(K),int(M)])
    return specs

def visualize_embeddings(embeddings,labels,s):
    """Plots TSNE embeddings of fluke image embeddings.

    Args:
        embeddings (torch.Tensor): the embeddings
        labels (list): the list of length nsamples*N classes
    """

    unique_labels = np.unique(labels)
    colors = {label:np.random.rand(3) for label in unique_labels}

    if (embeddings.is_cuda):
        embeddings = embeddings.cpu()

    # dim reduce
    low_d = TSNE(n_components=2).fit_transform(embeddings.data.numpy())

    # plot
    plt.clf()
    for i,label in enumerate(labels):
        plt.text(low_d[i,0],low_d[i,1],str(label),color=colors[label],fontsize=12)
        plt.scatter(low_d[i,0],low_d[i,1],color=colors[label])

    plt.margins(0.1)
    plt.savefig("%s.pdf" % s)

# Recorder that just stores the episode and accuracy
class PerformanceRecord():
    def __init__(self, name, out_path, dbg=False):
        self.dbg = dbg
        self.name = name

        # Create file, making sure we don't override the old stuff
        files = [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))]
        suffix=".csv"
        i=-1
        while name+suffix in files:
            i+=1
            suffix="_"+str(i)+".csv"

        path = os.path.join(out_path,name+suffix)

        # Create csv writter
        self.fieldnames = ["episode", "acc"]
        self.eval_file = open(path, "w", newline='')
        self.eval_file = csv.DictWriter(self.eval_file, fieldnames=self.fieldnames)
        self.eval_file.writeheader()
    
    def write_record(self, episode, acc):
        row = {}
        row["episode"] = episode
        row["acc"] = acc
        if self.dbg:
            print("{}: episode {}, acc {:.4f}".format(self.name, row["episode"], row["acc"]))
        self.eval_file.writerow(row)

# Variation of PerformanceRecord that is more useful for multi-epoch runs
class AggregatePerformanceRecord():
    def __init__(self, name, out_path, dbg=False):
        self.dbg = dbg
        self.name = name

        # Create file, making sure we don't override the old stuff
        files = [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))]
        suffix=".csv"
        i=-1
        while name+suffix in files:
            i+=1
            suffix="_"+str(i)+".csv"

        path = os.path.join(out_path,name+suffix)

        # Create csv writter
        self.fieldnames = ["curr_epoch", "episodes", "avg_acc", "std_dev"]
        self.eval_file = open(path, "w", newline='')
        self.eval_file = csv.DictWriter(self.eval_file, fieldnames=self.fieldnames)
        self.eval_file.writeheader()

        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add_record(self, acc):
        self.buffer.append(acc)

    def write_record_buffer(self, curr_epoch):
        row = {}
        row["curr_epoch"] = curr_epoch
        row["episodes"] = len(self.buffer)
        row["avg_acc"] = mean(self.buffer)
        row["std_dev"] = stdev(self.buffer)
        self.eval_file.writerow(row)
        if self.dbg:
            print("{}: curr_epoch {:.2f}, episodes {}, avg_acc {:.4f}, std_dev {:.4f}".format(self.name, row["curr_epoch"], row["episodes"], row["avg_acc"], row["std_dev"]))
        self.buffer = []

