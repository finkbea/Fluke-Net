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
from dataset import FlukeDataset, ClassDictionary

class ConvNeuralNet(torch.nn.Module):
    def __init__(self, C, f1, Filter_specs=None, Pre_trained_filters=None):
        super(ConvNeuralNet, self).__init__()

        self.F = getattr(torch, f1)

        # conv1 specs
        # in_channels = 3, out_channels = 32, kernel_size = 5
        self.conv1 = torch.nn.Conv2d(3, 32, 5) # 100 -> 96
        self.maxpool1 = torch.nn.MaxPool2d(2, 2) # 96 -> 48
        # conv2 specs
        # in_channels = 32, out_channels = 64, kernel_size = 3
        self.conv2 = torch.nn.Conv2d(32, 64, 3) # 48 -> 45
        self.maxpool2 = torch.nn.MaxPool2d(4, 4) # 45 -> 11

        # in the future, I wouldn't do the math to figure out the size of this layer
        # it's easier to just print the maxpool2 output shape inside the forward fn
        self.dense_hidden = torch.nn.Linear(11*11*64, 512)

        self.output = torch.nn.Linear(512, C)


        
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

        x = self.output(x)

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

    return parser.parse_args()

def train(model,train_loader,dev_loader,N,args):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        for update,(mb_x,mb_y) in enumerate(train_loader):

            mb_y_pred = model(mb_x) # evaluate model forward function
            loss      = criterion(mb_y_pred,mb_y) # compute loss

            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            if (update % args.report_freq) == 0:
                # eval on dev once per epoch
                num_correct = 0
                num_total = 0

                for _,(dev_mb_x,dev_mb_y) in enumerate(dev_loader):
                    dev_y_pred     = model(dev_mb_x)
                    _,dev_y_pred_i = torch.max(dev_y_pred,1)
                    num_correct   += (dev_y_pred_i == dev_mb_y).sum().data.numpy()
                    num_total     += len(dev_mb_y)
                dev_acc = num_correct / num_total

                # it's not great that we're comparing one training minibatch to the entire dev set,
                # but better than nothing
                _,train_y_pred_i = torch.max(mb_y_pred,1)
                train_num_correct = (train_y_pred_i == mb_y).sum().data.numpy()
                train_acc = train_num_correct / len(mb_y)

                print("%03d.%04d: train %.3f, dev %.3f" % (epoch,update,train_acc,dev_acc))

def main(argv):
    # parse arguments
    args = parse_all_args()

    class_dict = ClassDictionary()

    train_set = FlukeDataset(args.input_path, args.train_path, class_dict)
    dev_set = FlukeDataset(args.input_path, args.dev_path, class_dict)

    train_targets = train_set.getUniqueTargets()
    dev_targets = dev_set.getUniqueTargets()
    dev_only_target_count = len(np.setdiff1d(dev_targets, train_targets))

    # print some info to help interpret dev/training set accuracy
    print("%d distinct classes in %s" % (len(train_targets), args.train_path))
    print("%d distinct classes in %s" % (len(dev_targets), args.dev_path))
    if (dev_only_target_count):
        print("%d classes (%.1f%%) in dev set not in training set" % (dev_only_target_count, 100 * dev_only_target_count / len(dev_targets)))
    print('training classes:\n', train_targets)
    print('dev classes:\n', dev_targets)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=8)
    dev_loader = torch.utils.data.DataLoader(dev_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=8)


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
        
    
    # this C may not include all of the classes in the dev set if dev_targets is not inside train_targets,
    # which is okay - that just means that for now we will always fail to classify some of the dev set
    C = len(train_targets)
    model = ConvNeuralNet(C, args.f1, Filter_specs=Filter_specs, Pre_trained_filters=Pre_trained_filters)
    if (torch.cuda.is_available()):
        model = model.cuda()

    N = len(train_set)
    train(model,train_loader,dev_loader,N,args)


if __name__ == "__main__":
    main(sys.argv)
