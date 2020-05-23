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
import numpy as np
from dataset import PrototypicalDataset, protoCollate, ClassDictionary

class ConvNeuralNet(torch.nn.Module):
    def __init__(self, embed_dim, f1, Filter_specs=None, Pre_trained_filters=None):
        super(ConvNeuralNet, self).__init__()

        self.F = getattr(torch, f1)

        # TODO: Make the number of channels and possibly the number of 
        #   layers arguments since these seem essential but for now are
        #   hard to modify

        # conv1 specs
        self.conv1 = torch.nn.Conv2d(3, 8, 3) # 100 -> 98
        self.maxpool1 = torch.nn.MaxPool2d(2, 2) # 98 -> 49
        # conv2 specs
        self.conv2 = torch.nn.Conv2d(8, 16, 3) # 49 -> 47
        self.maxpool2 = torch.nn.MaxPool2d(2, 2) # 47 -> 23

        # in the future, I wouldn't do the math to figure out the size of this layer
        # it's easier to just print the maxpool2 output shape inside the forward fn
        self.dense_hidden = torch.nn.Linear(23*23*16, 512)

        # TODO: Another dense layer? (especially if/when conv filters have fixed params)

        self.embed = torch.nn.Linear(512, embed_dim)
        
        if not Pre_trained_filters is None :
            # If we have pre-trained filters, set our weights to them.
            dummy_op = 0
            conv1.weight = Pre_trained_filters[0]
            conv2.weight = Pre_trained_filters[1]


            
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x):
        'Takes a Tensor of input data and return a Tensor of output data.'

        x = self.conv1(x)
        x = self.F(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.F(x)
        x = self.maxpool2(x)

        # print(x.shape)

        x = x.view(x.size(0), -1) # flatten into vector

        x = self.dense_hidden(x)
        x = self.F(x)

        x = self.embed(x)

        return x

def parse_all_args():
    'Parses commandline arguments'

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path",help="The training set input data (directory)")
    parser.add_argument("train_path",help="The training set input/target data (csv)")
    parser.add_argument("dev_path",help="The development set input/target data (csv)")

    parser.add_argument("-f1",choices=["relu", "tanh", "sigmoid"],\
            help='The hidden activation function: "relu" or "tanh" or "sigmoid" (string) [default: "relu"]',default="relu")
    parser.add_argument("-opt",choices=["adadelta", "adagrad", "adam", "rmsprop", "sgd"],\
            help='The optimizer: "adadelta", "adagrad", "adam", "rmsprop", "sgd" (string) [default: "adadelta"]',default="adadelta")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 128]",default=128)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 4]",default=4)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-embed_dim",type=int,\
            help="The number of dimensions in the embedding space (int) [default: 100]",default=100)

    # these should be read from the input csv, but this is fine for now
    parser.add_argument("-train_support",type=int,\
            help="Size of support set for train dataset (int) [default: 5]",default=5)
    parser.add_argument("-train_query",type=int,\
            help="Size of query set for train dataset (int) [default: 5]",default=5)
    # todo: use prototypes generated by training?
    parser.add_argument("-dev_support",type=int,\
            help="Size of support set for dev dataset (int) [default: 1]",default=1)
    parser.add_argument("-dev_query",type=int,\
            help="Size of query set for dev dataset (int) [default: 1]",default=1)

    return parser.parse_args()

def distanceFromPrototypes(model, query_set, support_ids, support_map):
    class_count = len(support_map.keys())
    mb_size = query_set.shape[0]

    # this could (should) all probably be parallelized much better once we're tuning things

    prototypes = {}
    for id,class_tensors in support_map.items():
        prototypes[id] = model(class_tensors).mean(dim=0)

    mb_query_embeddings = model(query_set)

    euclid_dist = torch.zeros((mb_size, class_count))
    for d in range(mb_size):
        for class_id in range(class_count):
            # negative because we want to punish high distance and reward low distance
            euclid_dist[d][class_id] = -torch.dist(mb_query_embeddings[d], prototypes[class_id])

    return euclid_dist

def train(model,train_loader,dev_loader,N,args):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        for update,(query_set, support_ids, support_map) in enumerate(train_loader):

            distance = distanceFromPrototypes(model, query_set, support_ids, support_map)

            loss          = criterion(distance,support_ids)

            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            # TODO: Use train prototypes to evaluate the dev set
            if (update % args.report_freq) == 0:
                # eval on dev once per epoch
                num_correct = 0
                num_total = 0

                for _,(dev_query_set, dev_support_ids, dev_support_map) in enumerate(dev_loader):
                    dev_distance = distanceFromPrototypes(model, dev_query_set, dev_support_ids, dev_support_map)

                    _,dev_y_pred_i = torch.max(dev_distance,1)
                    num_correct   += (dev_y_pred_i == dev_support_ids).sum().data.numpy()
                    num_total     += dev_query_set.shape[0]
                dev_acc = num_correct / num_total

                # it's not great that we're comparing one training minibatch to the entire dev set,
                # but better than nothing
                _,train_y_pred_i = torch.max(distance,1)
                train_num_correct = (train_y_pred_i == support_ids).sum().data.numpy()
                train_acc = train_num_correct / query_set.shape[0]

                print("%03d.%04d: train %.3f, dev %.3f" % (epoch,update,train_acc,dev_acc))

def main(argv):
    # parse arguments
    args = parse_all_args()

    train_set = PrototypicalDataset(args.input_path, args.train_path, n_support=args.train_support, 
            n_query=args.train_query)
    dev_set = PrototypicalDataset(args.input_path, args.dev_path, apply_enhancements=False, n_support=args.dev_support, n_query=args.dev_query)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=8,
            collate_fn=protoCollate)
    dev_loader = torch.utils.data.DataLoader(dev_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=8,
            collate_fn=protoCollate)
    
    torch.multiprocessing.set_start_method("spawn")

    # Generate pre-trained filters using dictionary learning
    Pre_trained_filters = None
    Filter_specs = None

    if False :
        Filter_specs = [[32,5,5],[64,3,3]]
        Samples = None
        Dict_alpha = 2
        Dict_epochs = 10
        Dict_minibatch_size = 128
        Dict_jobs = 1
        Debug_flag = True
        Pre_trained_filters = prepare_dictionaries(Samples, Filter_specs, Dict_alpha=Dict_alpha, Dict_epochs=Dict_epochs, Dict_minibatch_size=Dict_minibatch_size, Dict_jobs=Dict_jobs, Debug_flag=Debug_flag)
        
    
    model = ConvNeuralNet(args.embed_dim, args.f1, Filter_specs=Filter_specs, Pre_trained_filters=Pre_trained_filters)
    if (torch.cuda.is_available()):
        model = model.cuda()

    N = len(train_set)
    train(model,train_loader,dev_loader,N,args)


if __name__ == "__main__":
    main(sys.argv)
