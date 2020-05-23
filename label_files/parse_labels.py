import argparse
import csv
import math
from random import shuffle

class Logger():
    """
    An extremely simple logger that just provides a 
    dummy write method when we don't care about logging
    """
    def __init__(self, log_path):
        if (log_path == ""):
            self.log_file = None
        else:
            self.log_file = open(log_path, "w")

    def log(self, msg):
        if (not self.log_file == None):
            self.log_file.write(msg+"\n")

def parse_labels(logger, args):
    with open(args.input_path, "r", newline='') as labels:
        # Read csv and 
        labels = csv.DictReader(labels)
        label_list = filter_images(labels, logger, args)
        (train,dev,dev_unseen,dev_new_whale) = get_output_rows(label_list, logger, args)
        
        logger.log("Outputing labels...")
        # Output the rows, maintaing the csv's format
        with open(args.train_output, "w", newline='') as out:
            out = csv.DictWriter(out, fieldnames=labels.fieldnames)
            out.writeheader()
            for row in train:
                out.writerow(row)
            logger.log("\t"+str(len(train)) + " added to train set")

        with open(args.dev_output, "w", newline='') as out:
            out = csv.DictWriter(out, fieldnames=labels.fieldnames)
            out.writeheader()
            for row in dev:
                out.writerow(row)
            logger.log("\t"+str(len(dev)) + " added to dev set")

        if args.dev_unseen_output == "":
            logger.log("\tdev_unseen_output not specified, skipping "+str(len(dev_unseen)))
        else:
            with open(args.dev_unseen_output, "w", newline='') as out:
                out = csv.DictWriter(out, fieldnames=labels.fieldnames)
                out.writeheader()
                for row in dev_unseen:
                    out.writerow(row)
                logger.log("\t"+str(len(train)) + " added to dev_unseen set")

        if args.dev_new_whale_output == "":
            logger.log("\tdev_new_whale_output not specified, skipping "+str(len(dev_new_whale)))
        else:
            with open(args.dev_new_whale_output, "w", newline='') as out:
                out = csv.DictWriter(out, fieldnames=labels.fieldnames)
                out.writeheader()
                for row in dev_new_whale:
                    out.writerow(row)
                logger.log("\t"+str(len(train)) + " added to dev_new_whale set")

def get_output_rows(labels, logger, args):
    """
    Gets the outputs for train, dev, dev_unseen and dev_new_whale.
    """
    # Get labels per class
    class_ex = {}
    for row in labels:
        if row['Id'] in class_ex:
            class_ex[row['Id']].append(row)
        else:
            class_ex[row['Id']] = [row]

    logger.log("Sorting labels...")
    # Remove all new_whale labels
    dev_new_whale = class_ex.pop("new_whale")

    train, dev, dev_unseen = [],[],[]

    # Warning! These are used for logging AND program logic
    train_classes, dev_classes, dev_unseen_classes = 0, 0, 0

    # Remove unseen classes
    for id in list(class_ex):
        if len(class_ex[id]) < args.train_examples_min:
            dev_unseen.extend(class_ex.pop(id))
            dev_unseen_classes += 1

    # Randomize class order so dev classes are effectively pulled at random
    rem = list(class_ex)
    shuffle(rem)

    targ_dev_classes = int(len(rem)*args.train_dev_split)
    for id in rem:
        # Split labels if dev quota not satisfied and minimum dev examples met
        if dev_classes < targ_dev_classes and len(class_ex[id]) >= args.train_examples_min + args.dev_examples_min:
            if len(class_ex[id]) >= args.train_examples_min + args.dev_examples_max:
                dev.extend(class_ex[id][:args.dev_examples_max])
                train.extend(class_ex[id][args.dev_examples_max:])
            else:
                train.extend(class_ex[id][:args.train_examples_min])
                dev.extend(class_ex[id][args.train_examples_min:])
            train_classes += 1
            dev_classes += 1
        # Otherwise simply add to train labels
        else:
            train.extend(class_ex[id])
            train_classes += 1

    logger.log("\tTrain classes: " + str(train_classes))
    logger.log("\tDev classes: " + str(dev_classes))
    logger.log("\tUnseen classes: " + str(dev_unseen_classes))
    logger.log("")

    return (train, dev, dev_unseen, dev_new_whale)

def filter_images(labels, logger, args):
    filtered = []

    AR_LT_skipped = 0
    AR_UT_skipped = 0
    Min_Dim_skipped = 0

    logger.log("Filtering images...")
    for row in labels:
        skip = False
        if (float(row["AR"]) < args.AR_LT):
            AR_LT_skipped += 1
            skip = True
        elif (float(row["AR"]) > args.AR_UT):
            AR_UT_skipped += 1
            skip = True
        elif (int(row["W"]) < args.Min_Dim or int(row["H"]) < args.Min_Dim):
            Min_Dim_skipped += 1
            skip = True
        
        if not skip:
            filtered.append(row)
    
    logger.log("\tAR_LT: " + str(AR_LT_skipped))
    logger.log("\tAR_UT: " + str(AR_UT_skipped))
    logger.log("\tMin_Dim: " + str(Min_Dim_skipped))
    logger.log("Total skips: " + str(AR_LT_skipped + AR_UT_skipped + Min_Dim_skipped))
    logger.log("Remaining data points: " + str(len(filtered)))
    logger.log("")
    return filtered


def parse_args():
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument("input_path", type=str, help="The path to the input CSV")
    parser.add_argument("train_output", type=str, help="The path to output the train CSV")
    parser.add_argument("dev_output", type=str, help="The path to output the dev CSV for seen classes")

    # Optional datasets
    parser.add_argument("-dev_unseen_output", type=str,
        help="The path to output the dev CSV for unseen classes (optional)", default="")
    parser.add_argument("-dev_new_whale_output", type=str,
        help="The path to output the dev CSV for unseen new_whale classes (optional)", default="")

    # Dataset properties
    parser.add_argument("-train_dev_split", type=float, 
        help="The proportion of shared classes from train to dev [default: 0.5]", default=0.5)
    parser.add_argument("-train_examples_min", type=int,
        help="The minimum examples per class in the train set [default: 10]", default=10)
    parser.add_argument("-dev_examples_max", type=int,
        help="The maximum examples per seen class in the dev set (int) [default: 3]", default=3)
    parser.add_argument("-dev_examples_min", type=int,
        help="The minimum examples per seen class in the dev set (int) [default: 2]", default=2)

    # Image-quality filters
    parser.add_argument("-AR_LT", type=float, 
        help="A lower threshold for image AR (float) [default: 1.0]", default=1.0)
    parser.add_argument("-AR_UT", type=float, 
        help="An upper threshold for image AR (float) [default: 5.0]", default=5.0)
    parser.add_argument("-Min_Dim", type=int,
        help="The minimum size of either width or height that can be included in the dataset (int)"+
        " [default: 224]", default=224) 
    
    # Other
    parser.add_argument("-log_path", help="Path to log file if desired (optional)", default="")

    return parser.parse_args()

def main():
    args = parse_args()
    logger = Logger(args.log_path)
    parse_labels(logger, args)

if __name__ == "__main__":
    main()
