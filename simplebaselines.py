# Some simple sklearn baselines
# Caelan Booker

import sys
import argparse
from dataset import PrototypicalDataset
from sklearn.dummy import DummyClassifier

from collections import Counter
import random 



def test_baselines(train_x, train_y, dev_x, dev_y, r_st) :
    
    clf = DummyClassifier(strategy='stratified', random_state=r_st)
    clf.fit(train_x, train_y)
    strat_score = clf.score(dev_x, dev_y)

    clf = DummyClassifier(strategy='most_frequent', random_state=r_st)
    clf.fit(train_x, train_y)
    freq_score = clf.score(dev_x, dev_y)

    clf = DummyClassifier(strategy='prior', random_state=r_st)
    clf.fit(train_x, train_y)
    prior_score = clf.score(dev_x, dev_y)

    clf = DummyClassifier(strategy='uniform', random_state=r_st)
    clf.fit(train_x, train_y)
    unif_score = clf.score(dev_x, dev_y)

    print("========================================")
    print("Baseline testing results.\n")
    print("stratified baseline accuracy: " + str(strat_score) + "\n")
    print("frequency baseline accuracy:  " + str(freq_score) + "\n")
    print("prior baseline accuracy:      " + str(prior_score) + "\n")
    print("uniform baseline accuracy:    " + str(unif_score) + "\n")
    print("========================================")



def parse_all_args():
    'Parses commandline arguments'
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="The path to the data (directory)")
    parser.add_argument("train_path", help="The training set input/target data (csv)")
    
    parser.add_argument("-dev_path", help="The development set input/target data (csv)", default=None)
    
    parser.add_argument("-seed_clf_state", help="Seed for the classifier random state. If not provided, no random state is set [int]", default=None)
    parser.add_argument("-seed_random_shuffle", help="Seed for the random list shuffling. If not provided, no random state is set [int]", default=None)

    parser.add_argument("-num_elements",type=int, help="Number of elements to extract from each class. Will throw error if not enough elements (int) [default: 10]",default=10)

    parser.add_argument("-eval_same_set",type=bool, help="Whether or not to pass the same list to the classifiers for both training and eval. Will do nothing if dev_path is given (bool) [default: False]",default=False)

    
    parser.add_argument("-dbg",type=bool, help="Status of the debug flag (bool) [default: False]",default=False)

    return parser.parse_args()



def main(argv):
    # parse arguments
    args = parse_all_args()
    debug = args.dbg

    if not args.seed_random_shuffle is None :
            random.seed(args.seed_random_shuffle)            

    nsup = args.num_elements
    nque = 0
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []

    # Set up the training sets
    if True:
        sample_set = PrototypicalDataset(args.input_path, args.train_path, apply_enhancements=False, n_support=nsup, n_query=nque)
        sample_ids=list(range(len(sample_set)))

        # Compose a list of images and their corresponding classes
        for i in sample_ids:
            qu, su, cl = sample_set[i]
            for j in qu :
                train_x.append(j)
                train_y.append(cl)
            for j in su :
                train_x.append(j)
                train_y.append(cl)
        if debug :
            print(len(sample_set))
            print(len(train_x))

        # Randomly shuffle the pairs in the training set (this does keep each image aligned with its corresponding class)
        temp = list(zip(train_x, train_y)) 
        random.shuffle(temp) 
        train_x, train_y = zip(*temp)

    # Print number of unique classes in training set
    if debug :
        print(Counter(train_y).keys())


    # Set up the dev sets
    if not args.dev_path is None :
        sample_set = PrototypicalDataset(args.input_path, args.dev_path, apply_enhancements=False, n_support=nsup, n_query=nque)
        sample_ids=list(range(len(sample_set)))

        for i in sample_ids:
            qu, su, cl = sample_set[i]
            for j in qu :
                dev_x.append(j)
                dev_y.append(cl)
            for j in su :
                dev_x.append(j)
                dev_y.append(cl)
        if debug :
            print(len(sample_set))
            print(len(dev_x))

        temp = list(zip(dev_x, dev_y)) 
        random.shuffle(temp) 
        dev_x, dev_y = zip(*temp)

    # If no dev set is given, make one with part of or all of the training data
    else :
        if args.eval_same_set :
            dev_x = train_x
            dev_y = train_y
        else :
            split_size = len(train_x) // 5 # make 20% of our training stuff our dev

            dev_x = train_x[:split_size]
            train_x = train_x[split_size:]

            dev_y = train_y[:split_size]
            train_y = train_y[split_size:]
    
    # Run the baselines
    test_baselines(train_x, train_y, dev_x, dev_y, args.seed_clf_state)

    
  
if __name__ == "__main__":
    main(sys.argv)
