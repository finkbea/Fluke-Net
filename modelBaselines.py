"""
@authors: Adicus Finkbeiner and Connor Barlow (Using Dylan's model as a starting point)
A simple example of the kmeans baseline
#! /usr/bin/env python3
"""

import torch
import argparse
import sys
import os
import numpy as np
from dataset import PrototypicalDataset, protoCollate, ImageLoader
from strconv2d import StrengthConv2d
from utils import parse_filter_specs, visualize_embeddings, PerformanceRecord, AggregatePerformanceRecord
from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
from sklearn.metrics import silhouette_score
    
    #runs kmeans on our dataset (second parameter is number of clusters
def kMeans(x, k):
    #to be used once the optimal k (n_clusters) is found
    kmeans = KMeans(n_clusters=k).fit(x)
    print('labels ',kmeans.labels_)
    kmeans.predict(x)
    print('cluster centers ',kmeans.cluster_centers_)
    
    plt.scatter(x[:, 0], x[:, -1])
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='x')
    plt.title('Kmeans cluster')
    plt.show()
    
    #finds the optimal number of clusters for the best prediction for our dataset
def findK(x):
    #silhouette_score is between -1 and 1, higher value means better matched to its cluster
    clusterRange = list (range(2,10))

    for n in clusterRange:
        kmeans = KMeans(n_clusters=n)
        preds = kmeans.fit_predict(x)
        centers = kmeans.cluster_centers_

        score = silhouette_score(x, preds)
        print("For n_clusters = {}, silhouette score is {})".format(n, score))

def parse_all_args():
    'Parses commandline arguments'
    parser = argparse.ArgumentParser()
    # Default arguments. This is only ok because we're running everything with scripts anyways
    parser.add_argument("input_path",help="The path to the data (directory)")
    parser.add_argument("train_path",help="The training set input/target data (csv)")
    parser.add_argument("dev_path",help="The development set input/target data (csv)")
    parser.add_argument("test_path",help="The test set input/target data (csv)")
    parser.add_argument("out_path",help="The path for all of our model evaluations (directory)")
    parser.add_argument("filter_specs",type=str,
            help="Comma-deliminated list of filter specifications. For each layer, the format is "\
            + "CxKxM, where C is the number of filters, K is the size of each KxK filter, and M is either "\
            + "the size of the max-pooling layer, or 0 if not being used")
    return parser.parse_args()

def main(argv):
    # parse arguments
    args = parse_all_args()

    train_set = PrototypicalDataset(args.input_path, args.train_path, n_support=args.support, 
            n_query=args.query)
    dev_set = PrototypicalDataset(args.input_path, args.dev_path, apply_enhancements=False, 
            n_support=args.support, n_query=args.query)

    # Use the same minibatch size to make each dataset use the same episode size
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=0, pin_memory=True,
            collate_fn=protoCollate)
    dev_loader = torch.utils.data.DataLoader(dev_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=0, pin_memory=True,
            collate_fn=protoCollate)

    Filter_specs = parse_filter_specs(args.filter_specs)
    Pre_trained_filters = None
    if not args.pre_trained is None:
        Pre_trained_filters = torch.load(args.pre_trained)

    train_out = AggregatePerformanceRecord("train",args.out_path,dbg=args.print_reports)
    dev_out = AggregatePerformanceRecord("dev",args.out_path,dbg=args.print_reports)
    test_out = PerformanceRecord("test",args.out_path,dbg=args.print_reports)

    N = len(train_set)
    
    
    temp_train_set = torch.cat(train_loader)
    findK(temp_train_set)
    
    #pred = kMeans(temp_train_set, x)
    
    
    
if __name__ == "__main__":
    main(sys.argv)
