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
    def __init__(self, embed_dim, f1, image_shape, Filter_specs=None, Pre_trained_filters=None):
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
            self.conv_list.append(StrengthConv2d(prev_channels, C, K, strength_flag=True))
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
                """
                # If this is a strength based filter, we can also choose to load
                # the strength values
                if conv.strength_flag :
                    conv.strength.data = SOME_STRENGTH
                """

            
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

def train(model,train_loader,dev_loader,train_out,dev_out,N,args,**params):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adadelta(model.parameters(), params.get('lr'))

    averageLoss = []
    min_loss=-1
    min_loss_i=-1

    for epoch in range(args.epochs):
        for update,(query_set, support_set, target_ids, _) in enumerate(train_loader):
            # Training
            _,distance = distanceFromPrototypes(model, query_set, support_set, args.support, args.query)

            if (torch.cuda.is_available()):
                target_ids = target_ids.cuda()
            loss = criterion(distance, target_ids)
            averageLoss.append(loss.item())

            optimizer.zero_grad() # reset the gradient valuedict_out.csvs
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            reported_epoch = epoch + (update / N)
            if min_loss == -1 or loss.item() < min_loss:
                min_loss = loss.item()
                min_loss_i = reported_epoch

            # Reporting for train set
            # if (update % args.train_report_interval) == 0:
            #     train_out.add_record(get_episode_accuracy(distance, target_ids), loss.item())
            #     if len(train_out) == args.train_report_freq:
            #         train_out.write_record_buffer(reported_epoch)

            # Reporting for dev set + optional dev embedding graph
            # if (update % args.dev_report_freq) == 0:
            #     dev_embeddings = []

            #     for _,(dev_query_set, dev_support_set, dev_target_ids, dev_target_names) in enumerate(dev_loader):
            #         dev_embed,dev_distance = distanceFromPrototypes(model, dev_query_set, dev_support_set, args.support, args.query)
            #         dev_embeddings.append(dev_embed.detach())

            #         if (torch.cuda.is_available()):
            #             dev_target_ids = dev_target_ids.cuda()
            #         dev_out.add_record(get_episode_accuracy(dev_distance,dev_target_ids))

            #     dev_out.write_record_buffer(reported_epoch)

            #     if (args.save_embed_graph):
            #         visualize_embeddings(torch.cat(dev_embeddings), dev_target_names, "%d.%d" % (epoch, update))

        # if ((args.checkpoint_freq > 0) and (epoch % args.checkpoint_freq == 0)):
        #     out_path = os.path.join(args.out_path, 'checkpoint_%d.pt' % (epoch))
        #     torch.save(model.state_dict(), out_path)
        #     print('Saved checkpoint at epoch %d to %s' % (epoch, out_path))

            print(reported_epoch)
        if (epoch == 0):
            # cache during first epoch, then load from every epoch thereafter
            # https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/
            train_loader.dataset.img_loader.setUseCache(True)
            train_loader.dataset.num_workers=8
            min_loss 
         #  dev_loader.dataset.img_loader.setUseCache(True)
         #  dev_loader.dataset.num_workers=8
    return (mean(averageLoss),stdev(averageLoss),min_loss,min_loss_i)

def evaluate_test(model, test_loader, test_out, args):
    model.eval() # switch to eval mode to disable dropout
    for i,(query_set,support_set,target_ids,_) in enumerate(test_loader):
        _,distance = distanceFromPrototypes(model, query_set, support_set, args.support, args.query)

        if (torch.cuda.is_available()):
            target_ids = target_ids.cuda()
        test_out.write_record(i,get_episode_accuracy(distance,target_ids))

# def readTune():
    # Establish a connection to a SQLite local database
    # conn = choco.SQLiteConnection("sqlite:///hpTuning.db")
    # results = conn.results_as_dataframe()
    # results = pd.melt(results, id_vars=["_loss"], value_name='value', var_name="variable")

    # sns.lmplot(x="value", y="_loss", data=results, col="variable", col_wrap=3, sharex=False)

    # plt.show()

def blackBoxfcn(args,**params):
    Pre_trained_filters = None
    if params["pretrained"]:
        Pre_trained_filters = torch.load(make_dicts(params["filter_specs"], args.dict_sample_size, args.input_path, args.dict_sample_space_csv, args.out_path))
        

    train_set = PrototypicalDataset(args.input_path, args.train_path, n_support=args.support, 
            n_query=args.query,image_shape=params.get('image size'),apply_enhancements=params.get('enhancements?'))
    # dev_set = PrototypicalDataset(args.input_path, args.dev_path, apply_enhancements=params.get('enhancements?'), 
     #       n_support=args.support, n_query=args.query,image_shape=params.get('image size'))

    # Use the same minibatch size to make each dataset use the same episode size
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
            drop_last=False, batch_size=params.get('mb'), num_workers=0, pin_memory=True,
            collate_fn=protoCollate)
    # dev_loader = torch.utils.data.DataLoader(dev_set, shuffle=True,
    #       drop_last=False, batch_size=params.get('mb'), num_workers=0, pin_memory=True,
    #       collate_fn=protoCollate)

    Filter_specs = parse_filter_specs(params["filter_specs"])
    
    model = ConvNeuralNet(args.embed_dim, args.f1, train_set.image_shape, Filter_specs=Filter_specs, Pre_trained_filters=Pre_trained_filters)
    if (args.checkpoint_path):
        state = torch.load(args.checkpoint_path)
        model.load_state_dict(state)
        print("Loaded checkpoint %s" % (args.checkpoint_path))
        # torch saves the device the model was on, so we don't need to re-load to CUDA if it was saved from CUDA
    else:
        if (torch.cuda.is_available()):
            model = model.cuda()

    # train_out = AggregatePerformanceRecord("train",args.out_path,dbg=args.print_reports)
    # dev_out = AggregatePerformanceRecord("dev",args.out_path,dbg=args.print_reports)
    # test_out = PerformanceRecord("test",args.out_path,dbg=args.print_reports)

    N = len(train_loader)
    # Calculate the loss for the sampled point (minimized)
    # This would be your training code
    dev_loader,train_out,dev_out = None,None,None
    loss = train(model,train_loader,dev_loader,train_out,dev_out,N,args,**params)
    return loss

def make_specs():
    return ["32x7x2,64x3x3avg","8x3x2,16x3x2,32x7x4","8x3x2,32x7x2,64x3x3avg",
            "32x7x2,64x3x3"] 

def main(argv):
    # parse arguments
    args = parse_all_args()

    filter_spec_choices=make_specs()

    #Chocolate Code
    # Define the parameters to tune
    space = {
                "lr": choco.uniform(low=.001, high=.1),
                "mb": choco.quantized_uniform(low=5, high=35,step=5),
                "image size": choco.choice([(100,100),(125,125),(150,150),(175,175),(200,200),(224,224)]),
                "enhancements?": choco.choice([True,False]),
                "filter_specs": choco.choice(filter_spec_choices),
                "pretrained": choco.choice([True,False])
            }

    # Establish a connection to a SQLite local database
    conn = choco.SQLiteConnection("sqlite:///hpTuning.db")

    out = open(os.path.join(args.out_path, "results.csv"), "w", newline='')

    header=list(space)
    header.extend(["epochs","avg_loss","std_loss","min_loss","min_loss_epoch"])

    out_writer = csv.DictWriter(out, fieldnames=header)
    out_writer.writeheader()

    for i in range(2):
        # Construct the optimizer
        sampler = choco.Bayes(conn, space)

        # Sample the next point
        token, params = sampler.next()

        avg_loss,std_loss,min_loss,min_loss_epoch = blackBoxfcn(args,**params)
        params["epochs"]   = args.epochs
        params["avg_loss"] = avg_loss      
        params["std_loss"] = std_loss
        params["min_loss"] = min_loss
        params["min_loss_epoch"] = min_loss_epoch
        
        out_writer.writerow(params)
        out.flush()

        # Add the loss to the database
        sampler.update(token, avg_loss)
    
    # # Get test set performance
    # test_set = PrototypicalDataset(args.input_path, args.test_path, apply_enhancements=False, n_support=args.support, n_query=args.query)
    # test_loader = torch.utils.data.DataLoader(test_set, shuffle=True,
    #         drop_last=False, batch_size=args.mb, num_workers=0, pin_memory=True,
    #         collate_fn=protoCollate)
    
    # # we should make sure this fn works, but we should not run this on the actual test set even once before we are completely done training
    # # evaluate_test(model, test_loader, test_out, args)

    #readTune()

if __name__ == "__main__":
    main(sys.argv)
