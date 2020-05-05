from skimage import io
import os
import csv

DIR = "~/classes/3_20/deep/whales/train"

IMG_STATS = ["BW", "W", "H", "AR"]

with open("train_labels.csv", newline='') as orig_f:
    orig = csv.DictReader(orig_f)
    fieldnames = orig.fieldnames
    for field in IMG_STATS:
        fieldnames.append(field)
    with open("extended_train_labels.csv", "w", newline='') as out_f:
        out = csv.DictWriter(out_f, fieldnames=fieldnames)
        for row in orig:
            file = row['Image']
            image = io.imread(os.path.join(DIR, file))
            BW = (len(image.shape) == 2)
            W = image.shape[0]
            H = image.shape[1]
            AR = float(W)/H
            out.writerow({fieldnames[0]: row[fieldnames[0]], fieldnames[1]: row[fieldnames[1]],
                "BW": BW, "W": W, "H": H, "AR": AR})
        