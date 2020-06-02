#! /usr/bin/env python3

"""
@authors: Dylan Thompson

A simple example of building a model in PyTorch 
using nn.Module.

Based on the PyTorch neural net skeleton written by
Brian Hutchinson (Brian.Hutchinson@wwu.edu)
for WWU CSCI 481 labs.

For usage, run with the -h flag.

"""

import torch
import argparse
import sys
import os
import numpy as np
from protoDataset import PrototypicalDataset, protoCollate, ImageLoader
from strconv2d import StrengthConv2d
from utils import parse_filter_specs, visualize_embeddings, PerformanceRecord, AggregatePerformanceRecord
import chocolate as choco
import matplotlib.pyplot as plt
import csv
import pandas as pd
from dict_by_a_different_name import make_dicts
from statistics import stdev,mean

class ConvNeuralNet(torch.nn.Module):
    def __init__(self, embed_dim, f1, image_shape, use_strength, Filter_specs=None, Pre_trained_filters=None):
        super(ConvNeuralNet, self).__init__()

        self.F = getattr(torch, f1)

        self.conv_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()
        prev_channels = 3
        out_w = image_shape[0]
        out_h = image_shape[1]
        # C: output channels, K: kernel size, M: max pooling size
        for C,K,M,T in Filter_specs:
            out_w -= (K-1)
            out_h -= (K-1)
            self.conv_list.append(StrengthConv2d(prev_channels, C, K, strength_flag=use_strength))
            prev_channels = C
            if not M == 0:
                out_w //= M
                out_h //= M
                if (T == 'avg'):
                    self.pool_list.append(torch.nn.AvgPool2d(M))
                elif (T == 'max'):
                    self.pool_list.append(torch.nn.MaxPool2d(M))
            # Maintain equal list length with identity function
            else:
                self.pool_list.append(torch.nn.Identity())

        self.batch2d = torch.nn.BatchNorm2d(Filter_specs[0][0])

        self.dense_hidden = torch.nn.Linear(out_w*out_h*C, 512)
        self.drop = torch.nn.Dropout(p=0.5)

        # TODO: Another dense layer? (especially if/when conv filters have fixed params)

        self.embed = torch.nn.Linear(512, embed_dim)
        
        if not Pre_trained_filters is None :
            # If we have pre-trained filters, set our weights to them.
            for conv,filter in zip(self.conv_list,Pre_trained_filters):
                conv.weight.data = filter
                # In case strength_flag is set to true and requires_grad is also
                # still true, set it to false
                if conv.strength_flag and conv.weight.requires_grad :
                    conv.weight.requires_grad = False
            
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x):
        'Takes a Tensor of input data and return a Tensor of output data.'

        for i,(conv,pool) in enumerate(zip(self.conv_list, self.pool_list)):
            x = conv(x)
            if (i == 0):
                x = self.batch2d(x)
            x = self.F(x)
            x = pool(x)

        x = x.reshape(x.size(0), -1) # flatten into vector

        x = self.dense_hidden(x)
        x = self.F(x)

        x = self.drop(x)

        x = self.embed(x)

        return x

def parse_all_args():
    'Parses commandline arguments'

    parser = argparse.ArgumentParser()

    # Default arguments. This is only ok because we're running everything with scripts anyways
    parser.add_argument("input_path",help="The path to the data (directory)")
    parser.add_argument("train_path",help="The training set input/target data (csv)")
    parser.add_argument("dev_path",help="The development set input/target data (csv)")
    parser.add_argument("test_path",help="The test set input/target data (csv)")
    parser.add_argument("dict_sample_space_csv")
    parser.add_argument("out_path",help="The path for all of our model evaluations (directory)")
    
    # General architecture parameters
    parser.add_argument("-f1",choices=["relu", "tanh", "sigmoid"],\
            help='The hidden activation function: "relu" or "tanh" or "sigmoid" (string) [default: "relu"]',default="relu")
    parser.add_argument("-opt",choices=["adadelta", "adagrad", "adam", "rmsprop", "sgd"],\
            help='The optimizer: "adadelta", "adagrad", "adam", "rmsprop", "sgd" (string) [default: "adadelta"]',default="adadelta")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    
    # Prototypical network parameters (mb included since it is actually episode size)
    parser.add_argument("-mb",type=int,\
            help="The size of the train/dev/test episodes (int) [default: 5]",default=5)
    parser.add_argument("-embed_dim",type=int,\
            help="The number of dimensions in the embedding space (int) [default: 100]",default=100)
    parser.add_argument("-pre_trained",type=str,
            help="The directory of the .pt file that contains a list of convolutions. Must either match or prepend filter"\
            +" specs (optional)",default=None)
    parser.add_argument("-support",type=int,\
            help="Size of support set for all datasets (int) [default: 5]",default=5)
    parser.add_argument("-query",type=int,\
            help="Size of query set for all datasets (int) [default: 5]",default=5)
    
    # Reporting args
    parser.add_argument("-print_reports",type=bool,\
            help="Prints report information as well as storing it (bool) [default: True]",default=True)
    parser.add_argument("-train_report_freq",type=int,\
            help="Aggregate train set performance recorded every (train_report_freq*train_report_interval) updates (int) [default: 8]",default=8)
    parser.add_argument("-train_report_interval",type=int,\
            help="The number of updates between each recorded episode performance (int) [default: 4]",default=4)
    parser.add_argument("-dev_report_freq",type=int,\
            help="Aggregate performance dev set recorded every dev_report_freq updates (int) [default: 32]",default=32)
    parser.add_argument("-save_embed_graph",type=bool,
            help="Whether a graph of the dev set embeddings should be saved during each stat log step (optional)",default=False)

    # Saving/loading args
    parser.add_argument("-checkpoint_path",type=str,
            help="Path to a file containing a model parameter checkpoint to be loaded (int)",default=None)
    parser.add_argument("-checkpoint_freq",type=int,
            help="The number of epochs between each model parameter checkpoint (int)",default=-1)

    parser.add_argument("-dict_sample_size",default=50)

    return parser.parse_args()

def distanceFromPrototypes(model, query_set, support_set, support_count, query_count):
    class_count = query_set.shape[0] // query_count

    if (torch.cuda.is_available()):
        support_set = support_set.cuda()
        query_set = query_set.cuda()

    # support_set is (class_count*support_count) x (image dims)
    # support_embeddings is (class_count*support_count) x embed_dim
    support_embeddings = model(support_set)

    # put embeddings for each class's support set into a new dim
    # then average along the support_set dim to get the prototype embedding for each class
    prototypes = support_embeddings.reshape(class_count,support_count,-1).mean(dim=1)

    # repeat prototypes along dim 0; [p1,p2,p3] -> [p1,p2,p3,p1,p2,p3,p1,p2,p3]
    proto_pairs = prototypes.repeat(class_count * query_count, 1)

    # tile query set embeddings along dim 0; [q1,q2,q3] -> [q1,q1,q1,q2,q2,q2,q3,q3,q3]
    query_embeddings = model(query_set)
    query_pairs = query_embeddings.repeat_interleave(class_count,dim=0)

    # get distance between each embedding and all prototypes
    dist = -torch.nn.functional.pairwise_distance(query_pairs, proto_pairs)

    # put each query set element's distance to each prototype set into a new dim
    return query_embeddings,dist.reshape(query_count * class_count,-1)

def get_episode_accuracy(distance, target_ids):
    _,predicted_ids = torch.max(distance,1)
    num_correct = (predicted_ids == target_ids).sum()
    if (num_correct.is_cuda):
        num_correct = num_correct.cpu()
    acc = num_correct.item() / target_ids.shape[0] # jank but the math works out
    return acc

def train(model,train_loader,dev_loader,train_out,dev_out,N,args,params):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adadelta(model.parameters(), params['lr'])

    avg_loss_epoch = []
    stdev_loss_epoch = []
    min_loss_i=None

    for epoch in range(args.epochs):
        loss_epoch = []
        for update,(query_set, support_set, target_ids, _) in enumerate(train_loader):
            # Training
            _,distance = distanceFromPrototypes(model, query_set, support_set, args.support, args.query)

            if (torch.cuda.is_available()):
                target_ids = target_ids.cuda()
            loss = criterion(distance, target_ids)
            loss_epoch.append(loss.item())

            optimizer.zero_grad() # reset the gradient valuedict_out.csvs
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            reported_epoch = epoch + (update/N)

            print(reported_epoch)

        avg=mean(loss_epoch)
        dev=stdev(loss_epoch)
        if min_loss_i is None:
            min_loss_i = 0
        elif avg < min(avg_loss_epoch):
            min_loss_i = epoch
        avg_loss_epoch.append(avg)
        stdev_loss_epoch.append(dev)

        if (epoch == 0):
            # cache during first epoch, then load from every epoch thereafter
            # https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/
            train_loader.dataset.img_loader.setUseCache(True)
            train_loader.dataset.num_workers=8
    return (avg_loss_epoch[min_loss_i],stdev_loss_epoch[min_loss_i],min_loss_i)

def blackBoxfcn(args,params):
    # Get the filter specs from the individually defined jank
    Filter_specs = get_jank_filter_specs(params)

    # Thank goodness they're all square
    image_shape = (params['image_shape'],params['image_shape'])

    # Reverse the parsing direction since that's how the file names are defined
    spec_str = unparse_filter_specs(Filter_specs)

    Pre_trained_filters = None
    if params["pretrained"]:
        # Uniqueness of a filter is dependent on what image shape was used
        pretrained_fname = spec_str + "," + str(image_shape) + ".pt"
        pretrained_fpath = os.path.join(args.out_path, pretrained_fname)

        # Load filters if present, otherwise make and save them
        files = [f for f in os.listdir(args.out_path) if os.path.isfile(os.path.join(args.out_path, f))]
        if not pretrained_fname in files:
            # I hate this
            Pre_trained_filters = make_dicts(Filter_specs, image_shape, args.dict_sample_size, 
                args.input_path, args.dict_sample_space_csv)
            torch.save(Pre_trained_filters, pretrained_fpath)
        else:
            # Load pretrained filters
            Pre_trained_filters = torch.load(pretrained_fpath)

    # Create dataset/dataloader with the correct specifications
    train_set = PrototypicalDataset(args.input_path, args.train_path, n_support=args.support, 
            n_query=args.query,image_shape=image_shape,apply_enhancements=params['use_enhancements'])
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
            drop_last=False, batch_size=params['mb'], num_workers=0, pin_memory=True,
            collate_fn=protoCollate)

    model = ConvNeuralNet(args.embed_dim, args.f1, train_set.image_shape, params["use_strength"], Filter_specs=Filter_specs, Pre_trained_filters=Pre_trained_filters)
    if (torch.cuda.is_available()):
        model = model.cuda()

    N = len(train_loader)

    # I'm glad we wrote reusable code so that it's easier to absolutely destroy it in the 0th hour
    dev_loader,train_out,dev_out = None,None,None
    loss = train(model,train_loader,dev_loader,train_out,dev_out,N,args,params)
    return (spec_str,*loss)

def get_jank_filter_specs(params):
    Filter_specs=[]
    for i in range(params["L"]):
        L=str(i+1)
        c="c"+L
        k="k"+L
        m="m"+L
        t="t"+L
        Filter_specs.append((params[c],params[k],params[m],params[t]))
    return Filter_specs

def unparse_filter_specs(filter_specs):
    specs_str = ""
    for i,(C,K,M,T) in enumerate(filter_specs):
        specs_str = specs_str + str(C) + "x" + str(K) + "x" + str(M) + T
        if i < len(filter_specs)-1:
            specs_str = specs_str + ","
    return specs_str

def main(argv):
    # parse arguments
    args = parse_all_args()

    #Chocolate Code
    # boy this is super cool wowee
    space = {
                "lr": choco.log(low=-3, high=-1, base=10),
                "mb": choco.quantized_uniform(low=5, high=35,step=5),
                "image_shape": choco.quantized_uniform(low=100,high=200,step=10),
                "use_enhancements": choco.choice([True,False]),
                "L": {1: {"c1": choco.quantized_log(low=2,high=8,step=1,base=2),
                          "k1": choco.quantized_uniform(low=3,high=11,step=2),
                          "m1": choco.quantized_uniform(low=2,high=8,step=1),
                          "t1": choco.choice(["max","avg"])},
                      2: {"c1": choco.quantized_log(low=2,high=8,step=1,base=2),
                          "k1": choco.quantized_uniform(low=3,high=11,step=2),
                          "m1": choco.quantized_uniform(low=2,high=8,step=1),
                          "t1": choco.choice(["max","avg"]),
                          "c2": choco.quantized_log(low=2,high=8,step=1,base=2),
                          "k2": choco.quantized_uniform(low=3,high=11,step=2),
                          "m2": choco.quantized_uniform(low=2,high=8,step=1),
                          "t2": choco.choice(["max","avg"])},
                      3: {"c1": choco.quantized_log(low=2,high=8,step=1,base=2),
                          "k1": choco.quantized_uniform(low=3,high=11,step=2),
                          "m1": choco.quantized_uniform(low=2,high=8,step=1),
                          "t1": choco.choice(["max","avg"]),
                          "c2": choco.quantized_log(low=2,high=8,step=1,base=2),
                          "k2": choco.quantized_uniform(low=3,high=11,step=2),
                          "m2": choco.quantized_uniform(low=2,high=8,step=1),
                          "t2": choco.choice(["max","avg"]),
                          "c3": choco.quantized_log(low=2,high=8,step=1,base=2),
                          "k3": choco.quantized_uniform(low=3,high=11,step=2),
                          "m3": choco.quantized_uniform(low=2,high=8,step=1),
                          "t3": choco.choice(["max","avg"])}
                    },
                "pretrained": choco.choice([True,False]),
                "use_strength": choco.choice([True,False])
            }

    # Establish a connection to a SQLite local database
    conn = choco.SQLiteConnection("sqlite:///hpTuning.db")

    # Construct the optimizer
    sampler = choco.Bayes(conn, space)

    # Sample the next point
    token, params = sampler.next()

    try:
        # Average loss and standard deviation of the loss of the epoch with the lowest average loss
        filter_specs,epoch_avg_loss,epoch_std_loss,epoch = blackBoxfcn(args,params)
    except Exception as e:
        # Record that these parameters caused an error (this has already happened)
        err_log=open(os.path.join(args.out_path, "failed_params.txt"),"a")
        err_log.write("error type: " + str(type(e))+"\n")
        err_log.write("error msg: " + str(e)+"\n")
        err_log.write("params: " + str(params)+"\n")
        err_log.write("============================\n")
        err_log.flush()
    else:
        # csv spaghetti since I've never touched a real database in my fucking life
        shared=["lr","mb","image_shape","use_enhancements","pretrained","use_strength"]
        non_shared=["filter_specs","best_epoch","epoch_avg_loss","epoch_std_loss"]
        header = shared + non_shared

        out,out_writer = None,None
        try:
            out = open(os.path.join(args.out_path, "results.csv"), "x", newline='')
            out_writer = csv.DictWriter(out, fieldnames=header)
            out_writer.writeheader()
        except FileExistsError:
            out = open(os.path.join(args.out_path, "results.csv"), "a", newline='')
            out_writer = csv.DictWriter(out, fieldnames=header)

        row={}
        for var in list(shared):
            row[var]=params[var]
        row["filter_specs"]   = filter_specs
        row["best_epoch"]     = epoch
        row["epoch_avg_loss"] = epoch_avg_loss      
        row["epoch_std_loss"] = epoch_std_loss
        
        out_writer.writerow(row)
        out.flush()

        # Add the loss to the database
        sampler.update(token, epoch_avg_loss)

if __name__ == "__main__":
    main(sys.argv)
