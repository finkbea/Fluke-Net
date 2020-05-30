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
        (train,dev,test) = get_output_rows(label_list, logger, args)
        
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

        with open(args.test_output, "w", newline='') as out:
            out = csv.DictWriter(out, fieldnames=labels.fieldnames)
            out.writeheader()
            for row in test:
                out.writerow(row)
            logger.log("\t"+str(len(test)) + " added to test set")

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
    class_ex.pop("new_whale")

    train, dev, test = [],[],[]

    # Remove classes below example threshold
    for id in list(class_ex):
        if len(class_ex[id]) < args.examples_min:
            class_ex.pop(id)

    classes = list(class_ex)
    shuffle(classes)

    assert(args.dev_split + args.test_split < 1.0)

    dev_classes = int(len(classes)*args.dev_split)
    test_classes = int(len(classes)*args.test_split)

    logger.log("\tTrain classes: " + str(len(classes) - (dev_classes + test_classes)))
    logger.log("\tDev classes: " + str(dev_classes))
    logger.log("\tTest classes: " + str(test_classes))
    logger.log("")

    for _ in range(dev_classes):
        dev += class_ex[classes.pop()]

    for _ in range(test_classes):
        test += class_ex[classes.pop()]

    for id in classes:
        train += class_ex[id]

    return (train, dev, test)

def filter_images(labels, logger, args):
    filtered = []

    AR_LT_skipped = 0
    AR_UT_skipped = 0
    Min_Dim_skipped = 0

    logger.log("Filtering images...")
    for row in labels:
        skip = False
        if not args.skip_quality_filter:
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
    parser.add_argument("dev_output", type=str, help="The path to output the dev CSV")
    parser.add_argument("test_output", type=str, help="The path to output the test CSV")

    # Dataset properties
    parser.add_argument("-dev_split", type=float, 
        help="The proportion of classes to go to the dev set [default: 0.125]", default=0.125)
    parser.add_argument("-test_split", type=float, 
        help="The proportion of classes to go to the test set [default: 0.125]", default=0.125)
    parser.add_argument("-examples_min", type=int,
        help="The minimum examples per class [default: 10]", default=10)

    # Image-quality filters
    parser.add_argument("-skip_quality_filter", type=bool,
        help="Skip filtering images by quality (bool) [default: False]", default=False)
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
