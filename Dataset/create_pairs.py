import os
import glob
import random

DATASET_SIZE = 1000
TRAIN_TEST_SPLIT = 0.90

dataset_path = "./Authors"

train = open("train_" + str(DATASET_SIZE)+ ".txt","w")
valid = open("valid_" + str(DATASET_SIZE)+ ".txt","w")

# Author directories
author_paths = glob.glob(os.path.join(dataset_path,'*'))
num_authors = len(author_paths)

print("Detected authors:", num_authors)
if num_authors < 2:
    raise Exception("At least 2 authors required!")

train_size = int((DATASET_SIZE/2) * TRAIN_TEST_SPLIT)

poslabel = 1
neglabel = 0

for i in range(int(DATASET_SIZE/2)):

    # Pick random positive author
    pos_id = random.choice(author_paths)

    # Select two different images from same author
    pos_files = glob.glob(os.path.join(pos_id, "*.png"))
    if len(pos_files) < 2:
        continue

    img1, img2 = random.sample(pos_files, 2)

    # Write positive example
    if i < train_size:
        train.write(img1 + " " + img2 + " " + str(poslabel) + "\n")
    else:
        valid.write(img1 + " " + img2 + " " + str(poslabel) + "\n")

    # Pick two different authors for negative pair
    neg_author1, neg_author2 = random.sample(author_paths, 2)

    neg_files1 = glob.glob(os.path.join(neg_author1, "*.png"))
    neg_files2 = glob.glob(os.path.join(neg_author2, "*.png"))

    if len(neg_files1) == 0 or len(neg_files2) == 0:
        continue
    
    img1_neg = random.choice(neg_files1)
    img2_neg = random.choice(neg_files2)

    # Write negative example
    if i < train_size:
        train.write(img1_neg + " " + img2_neg + " " + str(neglabel) + "\n")
    else:
        valid.write(img1_neg + " " + img2_neg + " " + str(neglabel) + "\n")

train.close()
valid.close()

print("Done.")
