import argparse
import csv
import math
from random import shuffle
from parse_labels import Logger, filter_images

def parse_dict_labels(logger, args):
    with open(args.input_path, "r", newline='') as labels:
        # Read csv and 
        labels = csv.DictReader(labels)
        label_list = filter_images(labels, logger, args)
        
        logger.log("Outputing labels...")
        # Output the rows, maintaing the csv's format
        with open(args.dict_out, "w", newline='') as out:
            out = csv.DictWriter(out, fieldnames=labels.fieldnames)
            out.writeheader()
            for row in label_list:
                out.writerow(row)
            logger.log("\t"+str(len(label_list)) + " added to dict set")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path of the input .csv")
    parser.add_argument("dict_out", help="Path of the output .csv")

    # Image-quality filters
    parser.add_argument("-AR_LT", type=float, 
        help="A lower threshold for image AR (float) [default: 1.0]", default=1.0)
    parser.add_argument("-AR_UT", type=float, 
        help="An upper threshold for image AR (float) [default: 5.0]", default=3.0)
    parser.add_argument("-Min_Dim", type=int,
        help="The minimum size of either width or height that can be included in the dataset (int)"+
        " [default: 500]", default=500) 
    
    # Other
    parser.add_argument("-log_path", help="Path to log file if desired (optional)", default="")

    return parser.parse_args()

def main():
    args = parse_args()
    logger = Logger(args.log_path)
    parse_dict_labels(logger, args)

if __name__ == "__main__":
    main()
