# Caelan Booker
# Modified copy of my lab 4

import torch
import argparse
import sys
import numpy as np
from strconv2d import StrengthConv2d

class MINSTDataset():
    def __init__(self, features_path, targets_path):
        self.inputs = np.load(features_path).astype(np.float32)
        self.targets = np.load(targets_path).astype(np.int64)
        
        inshape = self.inputs.shape        
        self.inputs = self.inputs.reshape((inshape[0], 1, 28, 28))        

    def __len__(self):
        return len(self.inputs)
        
    def D(self):
        N, D = self.inputs.shape
        return D

    def __getitem__(self, index):        
        X = self.inputs[index]
        y = self.targets[index]
        return X, y

class SimpleConvNeuralNet(torch.nn.Module):
    def __init__(self, func, C=10):
        super(SimpleConvNeuralNet, self).__init__()

        if (func == 'relu') :
            self.actifunc = torch.nn.ReLU()
        elif (func == 'tanh') :
            self.actifunc = torch.nn.Tanh()
        elif (func == 'sigmoid') :
            self.actifunc = torch.nn.Sigmoid()

        #Conv layer L = 1, w = 28, k = 3, L' = 8, w' = 26, p = 0, s = 1, f1() = parameter
        self.conv1 = StrengthConv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0, stride=1, bias=True, strength_flag=True)

        #Conv layer L = 8, w = 26, k = 3, L' = 16, w' = 24, p = 0, s = 1, f1() = parameter
        self.conv2 = StrengthConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0, stride=1, bias=True)

        #Max pooling layer L = 16, w = 24, k = 2, L' = 16, w' = 12, p = 0, s = 2, no f1()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2)

        #Fully connected (dense) layer 128 units, all connected to 12x12x16 items of max pool, f1() = parameter
        self.fullconnected1 = torch.nn.Linear(in_features=12*12*16, out_features=128, bias=True)

        #Fully connected (dense) layer to map from 128 input to a 10 dimensional output
        self.fullconnected2 = torch.nn.Linear(in_features=128, out_features=C, bias=True)

        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x, debug_flags=(False, False, False, False)):
        #reshape input data points from R^784 to R^28x28
        #x shape is (# of items, 1, 28, 28)
        xsh = x.shape
        x = self.conv1(x, debug_flags)
        x = self.actifunc(x)
        x = self.conv2(x)
        x = self.actifunc(x)
        x = self.maxpool1(x)
        x = x.reshape((xsh[0], 12*12*16))
        x = self.fullconnected1(x)
        x = self.actifunc(x)
        x = self.fullconnected2(x)
        return x
    
def parse_all_args():
    # Parses commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("C",help="The number of classes if classification or output dimension if regression (int)",type=int)
    parser.add_argument("train_x",help="The training set input data (npz)")
    parser.add_argument("train_y",help="The training set target data (npz)")
    parser.add_argument("dev_x",help="The development set input data (npz)")
    parser.add_argument("dev_y",help="The development set target data (npz)")

    parser.add_argument("-f1", help="The type of hidden activation function (str) [default: relu]", choices=['relu', 'tanh', 'sigmoid'], default='relu', type=str)
    parser.add_argument("-opt", help="The type of optimizer to be used (str) [default: adam]", choices=['adadelta', 'adagrad', 'adam', 'rmsprop', 'sgd'], default='adam', type=str)
    parser.add_argument("-lr",type=float, help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-mb",type=int, help="The minibatch size (int) [default: 32]",default=32)
    parser.add_argument("-report_freq",type=int, help="Dev performance is reported every report_freq updates (int) [default: 128]",default=128)
    parser.add_argument("-epochs",type=int, help="The number of training epochs (int) [default: 100]",default=100)
    return parser.parse_args()

def train(model, args, train_loader, dev_loader):
    if (args.opt == 'adam') :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif (args.opt == 'adadelta') :
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif (args.opt == 'adagrad') :
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif (args.opt == 'rmsprop') :
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif (args.opt == 'sgd') :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(args.epochs):

        totalsteps = 0
        for update,(mb_x,mb_y) in enumerate(train_loader):
            totalsteps += 1
            
            mb_y_pred = model(mb_x, debug_flags=(True, True, False, False)) # evaluate model forward function
            loss      = criterion(mb_y_pred,mb_y) # compute loss

            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            if (update % args.report_freq) == 0:
                devsteps = 0
                dev_acc = 0
                # eval on dev once per epoch
                for j,(dev_x,dev_y) in enumerate(dev_loader):                    
                    devsteps += 1
                    devN = dev_x.shape
                    dev_y_pred     = model(dev_x)
                    _,dev_y_pred_i = torch.max(dev_y_pred,1)
                    dev_acc        += (dev_y_pred_i == dev_y).sum().data.numpy()/devN[0]
                dev_acc /= devsteps
                print("%03d.%04d: dev %.3f" % (epoch,update,dev_acc))
                
def main(argv):
    # parse arguments
    args = parse_all_args()

    traindataset = MINSTDataset(args.train_x, args.train_y)
    devdataset = MINSTDataset(args.dev_x, args.dev_y)

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.mb, shuffle=True, drop_last=False)
    dev_loader = torch.utils.data.DataLoader(devdataset, batch_size=args.mb, shuffle=False, drop_last=False)
    
    model = SimpleConvNeuralNet(args.f1, args.C)

    train(model, args, train_loader, dev_loader)

if __name__ == "__main__":
    main(sys.argv)

    
