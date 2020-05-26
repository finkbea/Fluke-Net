import csv
from sklearn.manifold import TSNE
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