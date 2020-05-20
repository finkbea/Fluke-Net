import argparse
import csv
import math
from random import shuffle

"""An extremely simple logger that just provides a 
     dummy write method when we don't care about logging"""
class Logger():
    def __init__(self, log_path):
        if (log_path == ""):
            self.log_file = None
        else:
            self.log_file = open(log_path, "w")

    def log(self, msg):
        if (not self.log_file == None):
            self.log_file.write(msg+"\n")

"""A function for parsing the master csv into 
     train and dev sets, filtered accordingly."""
def parse_labels(logger, args):
    with open(args.input_path, "r", newline='') as labels:
        # Read csv and get list of filtered rows
        labels = csv.DictReader(labels)
        filtered = filter_labels(labels, logger, args)

        # Split data into two seperate lists of rows
        (dev, train) = (None, None)
        if (args.class_split):
            (dev, train) = class_split(filtered, logger, args)
        else:
            (dev, train) = data_split(filtered, logger, args)
        
        logger.log("Adding data points...")
        # Output the rows, maintaing the csv's format
        with open(args.dev_output, "w", newline='') as dev_out:
            dev_out = csv.DictWriter(dev_out, fieldnames=labels.fieldnames)
            dev_out.writeheader()
            for row in dev:
                dev_out.writerow(row)
            logger.log("\t"+str(len(dev)) + " added to dev set")

        with open(args.train_output, "w", newline='') as train_out:
            train_out = csv.DictWriter(train_out, fieldnames=labels.fieldnames)
            train_out.writeheader()
            for row in train:
                train_out.writerow(row)
            logger.log("\t"+str(len(train)) + " added to train set")
        
def data_split(filtered, logger, args):
    shuffle(filtered)
    split = int(args.dev_split * len(filtered))
    return (filtered[:split], filtered[split:])

def class_split(filtered, logger, args):
    # Count examples per class for all classes
    # It also has to contain the rows and my god this is dumb
    class_num = {}
    for row in filtered:
        if row['Id'] in class_num:
            class_num[row['Id']].append(row)
        else:
            class_num[row['Id']] = [row]

    # Reverse! Reverse!
    num_class = {}
    for k in list(class_num):
        num = len(class_num[k])
        if num in num_class:
            num_class[num].append(class_num[k])
        else:
            num_class[num] = [class_num[k]]
    
    # Cha cha real smooth
    logger.log("Splitting classes...")
    (dev, train) = ([], [])
    ceil = True
    for num in sorted(list(num_class)):
        split = None
        # Switch between floor/ceiling resolutions of class splitting, 
        # More necessary when the number of classes being split is around 1
        if ceil:
            split = math.ceil(args.dev_split * len(num_class[num]))
            ceil = False
        else:
            split = math.floor(args.dev_split * len(num_class[num]))
            ceil = True
        
        for full_class in num_class[num][:split]:
            dev.extend(full_class)
        for full_class in num_class[num][split:]:
            train.extend(full_class)
        
        logger.log("\tNumber of classes with " + str(num) + " examples, " +
            "dev=" + str(split) + ",train=" + str(len(num_class[num][split:])))

    # Lets go to work!
    logger.log("")
    return (dev,train)

"""Filter labels out using the args, and returns a list 
     of "rows" (just dicts) for further processing"""
def filter_labels(labels, logger, args):
    filtered = []

    new_whale_skipped = 0
    AR_LT_skipped = 0
    AR_UT_skipped = 0

    for row in labels:
        skip = False
        if (not args.new_whale and row["Id"]=="new_whale"):
            new_whale_skipped += 1
            skip = True
        elif (float(row["AR"]) < args.AR_LT):
            AR_LT_skipped += 1
            skip = True
        elif (float(row["AR"]) > args.AR_UT):
            AR_UT_skipped += 1
            skip = True
        
        if not skip:
            filtered.append(row)
    
    logger.log("Skips from...")
    logger.log("\tnew_whale: " + str(new_whale_skipped))
    logger.log("\tAR_LT: " + str(AR_LT_skipped))
    logger.log("\tAR_UT: " + str(AR_UT_skipped))
    logger.log("Total skips: " + str(new_whale_skipped + AR_LT_skipped + AR_UT_skipped))
    logger.log("Remaining data points: " + str(len(filtered)))
    logger.log("")
    return filtered


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", help="The path to the input CSV")
    parser.add_argument("train_output", help="The path to the output training CSV")
    parser.add_argument("dev_output", help="The path to the output development CSV")

    parser.add_argument("-class_split", type=bool,
        help="Whether split is done per-data point or per-class. \
            If per class, the splits try and keep an even distrabution of examples-per-class \
            per class per train/dev dataset relative to dev_split (bool) [default: True]", default=True)
    parser.add_argument("-dev_split", type=float, 
        help="The proportion of data for dev set (float) [default: 0.2]", default=0.2)
    parser.add_argument("-new_whale", type=bool,
        help="Whether whales with the new_whale tag are included (bool) [default: False]", default=False)
    parser.add_argument("-AR_LT", type=float, 
        help="A lower threshold for image AR (float) [default: 1.0]", default=1.0)
    parser.add_argument("-AR_UT", type=float, 
        help="An upper threshold for image AR (float) [default: 5.0]", default=5.0)
    parser.add_argument("-log_path", help="Path to log file if desired (optional)", default="")

    return parser.parse_args()

def main():
    args = parse_args()
    logger = Logger(args.log_path)
    parse_labels(logger, args)

if __name__ == "__main__":
    main()
