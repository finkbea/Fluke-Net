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
from dataset import PrototypicalDataset, protoCollate, ImageLoader
from utils import parse_filter_specs, visualize_embeddings

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
        for C,K,M in Filter_specs:
            out_w -= (K-1)
            out_h -= (K-1)
            self.conv_list.append(torch.nn.Conv2d(prev_channels, C, K))
            prev_channels = C
            if not M == 0:
                out_w //= M
                out_h //= M
                self.pool_list.append(torch.nn.MaxPool2d(M))
            # Maintain equal list length with identity function
            else:
                self.pool_list.append(torch.nn.Identity())

        self.dense_hidden = torch.nn.Linear(out_w*out_h*C, 512)

        # TODO: Another dense layer? (especially if/when conv filters have fixed params)

        self.embed = torch.nn.Linear(512, embed_dim)
        
        if not Pre_trained_filters is None :
            # If we have pre-trained filters, set our weights to them.
            for conv,filter in zip(self.conv_list,Pre_trained_filters):
                conv.weight.data = filter
                # If this is a strength based filter, we can also choose to load
                # the strength values
                if conv.strength_flag :
                    #conv.strength.data = STRENGTH
                    dummy_op = 0

            
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x):
        'Takes a Tensor of input data and return a Tensor of output data.'

        for conv,pool in zip(self.conv_list, self.pool_list):
            x = conv(x)
            x = self.F(x)
            x = pool(x)

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
    parser.add_argument("filter_specs",type=str,
            help="Comma-deliminated list of filter specifications. For each layer, the format is "\
            + "CxKxM, where C is the number of filters, K is the size of each KxK filter, and M is either "\
            + "the size of the max-pooling layer, or 0 if not being used")

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

    parser.add_argument("-pre_trained",type=str,
            help="The directory of the .pt file that contains a list of convolutions. Must either match or prepend filter"\
            +" specs (optional)",default=None)
    parser.add_argument("-save_embed_graph",type=bool,
            help="Whether a graph of the dev set embeddings should be saved during each stat log step (optional)",default=False)

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

def distanceFromPrototypes(model, query_set, support_set, support_count, query_count):
    class_count = query_set.shape[0] // query_count

    if (torch.cuda.is_available()):
        support_set = support_set.cuda()

    # support_set is (class_count*support_count) x (image dims)
    # support_embeddings is (class_count*support_count) x embed_dim
    support_embeddings = model(support_set)

    # put embeddings for each class's support set into a new dim
    # then average along the support_set dim to get the prototype embedding for each class
    prototypes = support_embeddings.reshape(class_count,support_count,-1).mean(dim=1)

    # repeat prototypes along dim 0; [p1,p2,p3] -> [p1,p2,p3,p1,p2,p3,p1,p2,p3]
    proto_pairs = prototypes.repeat(class_count * query_count, 1)

    if (torch.cuda.is_available()):
        query_set = query_set.cuda()
    # tile query set embeddings along dim 0; [q1,q2,q3] -> [q1,q1,q1,q2,q2,q2,q3,q3,q3]
    query_embeddings = model(query_set)
    query_pairs = query_embeddings.repeat_interleave(class_count,dim=0)

    # get distance between each embedding and all prototypes
    dist = -torch.nn.functional.pairwise_distance(query_pairs, proto_pairs)

    # put each query set element's distance to each prototype set into a new dim
    return query_embeddings,dist.reshape(query_count * class_count,-1)

def train(model,train_loader,dev_loader,N,args):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        for update,(query_set, support_set, target_ids, _) in enumerate(train_loader):

            _,distance = distanceFromPrototypes(model, query_set, support_set, args.train_support, args.train_query)

            if (torch.cuda.is_available()):
                target_ids = target_ids.cuda()
            loss = criterion(distance, target_ids)

            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            if (update % args.report_freq) == 0:
                # eval on dev once per epoch
                num_correct = 0
                num_total = 0

                dev_embeddings = []

                for _,(dev_query_set, dev_support_set, dev_target_ids, dev_target_names) in enumerate(dev_loader):
                    dev_embed,dev_distance = distanceFromPrototypes(model, dev_query_set, dev_support_set, args.dev_support, args.dev_query)
                    dev_embeddings.append(dev_embed.detach())

                    if (torch.cuda.is_available()):
                        dev_target_ids = dev_target_ids.cuda()
                    _,dev_y_pred_i = torch.max(dev_distance,1)
                    cur_correct    = (dev_y_pred_i == dev_target_ids).sum()
                    if (cur_correct.is_cuda):
                        cur_correct = cur_correct.cpu()
                    num_correct   += cur_correct.item()
                    num_total     += dev_query_set.shape[0]
                dev_acc = num_correct / num_total

                if (args.save_embed_graph):
                    visualize_embeddings(torch.cat(dev_embeddings), dev_target_names, "%d.%d" % (epoch, update))

                # it's not great that we're comparing one training minibatch to the entire dev set,
                # but better than nothing
                _,train_y_pred_i = torch.max(distance,1)
                train_cur_correct = (train_y_pred_i == target_ids).sum()
                if (train_cur_correct.is_cuda):
                    train_cur_correct = train_cur_correct.cpu()
                train_num_correct = train_cur_correct.item()
                train_acc = train_num_correct / query_set.shape[0]

                print("%03d.%04d: train %.3f, dev %.3f" % (epoch,update,train_acc,dev_acc))

        if (epoch == 0):
            # cache during first epoch, then load from every epoch thereafter
            # https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/
            train_loader.dataset.img_loader.setUseCache(True)
            train_loader.dataset.num_workers=8
            dev_loader.dataset.img_loader.setUseCache(True)
            dev_loader.dataset.num_workers=8

def main(argv):
    # parse arguments
    args = parse_all_args()

    train_set = PrototypicalDataset(args.input_path, args.train_path, n_support=args.train_support, 
            n_query=args.train_query)
    dev_set = PrototypicalDataset(args.input_path, args.dev_path, apply_enhancements=False, n_support=args.dev_support, n_query=args.dev_query)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=0, pin_memory=True,
            collate_fn=protoCollate)
    dev_loader = torch.utils.data.DataLoader(dev_set, shuffle=True,
            drop_last=False, batch_size=args.mb, num_workers=0, pin_memory=True,
            collate_fn=protoCollate)
    
    # torch.multiprocessing.set_start_method("spawn")

    Filter_specs = parse_filter_specs(args.filter_specs)
    Pre_trained_filters = None

    if not args.pre_trained is None:
        Pre_trained_filters = torch.load(args.pre_trained)

        # Validate that overlapping portions of Filter_specs and Pre_trained_filters match
        for (spec_channels,spec_kernel,_),filter_weights in zip(Filter_specs, Pre_trained_filters):
            channels_out,channels_in,kernel,_ = filter_weights.shape
            assert(spec_kernel == kernel)
            assert(spec_channels == channels_out)
            prev_channels_out = channels_out
    
    model = ConvNeuralNet(args.embed_dim, args.f1, train_set.image_shape, Filter_specs=Filter_specs, Pre_trained_filters=Pre_trained_filters)
    if (torch.cuda.is_available()):
        model = model.cuda()

    N = len(train_set)
    train(model,train_loader,dev_loader,N,args)

if __name__ == "__main__":
    main(sys.argv)
