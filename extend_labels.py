from skimage import io
import os
import csv

TRAIN_DIR = "/cluster/academic/CSCI481/fluke_net/kaggle_dataset/train"
ALL_LABELS_CSV = "all_train_labels"

with open(ALL_LABELS_CSV + ".csv", newline='') as orig_f:
    orig = csv.DictReader(orig_f)
    fieldnames = orig.fieldnames
    fieldnames.append("BW")
    fieldnames.append("W")
    fieldnames.append("H")
    fieldnames.append("AR")
    with open(ALL_LABELS_CSV + "_extended.csv", "w", newline='') as out_f:
        out = csv.DictWriter(out_f, fieldnames=fieldnames)
        out.writeheader()
        for row in orig:
            img_path = row['Image']
            image = io.imread(os.path.join(TRAIN_DIR, img_path))
            row["BW"] = (len(image.shape) == 2)
            row["W"]  = image.shape[1]
            row["H"]  = image.shape[0]
            row["AR"] = float(row["W"])/row["H"]
            out.writerow(row)
