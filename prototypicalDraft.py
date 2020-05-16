import argparse

""" Dependencies:
      -Image: Fixed size, k different sizes, any arbirtrary size
"""

parser = argparse.ArgumentParser()

""" Proto-Net Hyperparameters """
# In any other model this value would be about the easiest one to 
#   set. Unfortunately, as it stands 
parser.add_argument("C",help="Number of classes in training set")

# Mini-batch equivalent for few-shot learning. The number of samples 
#   per MB is K * (S + Q)
parser.add_argument("K",help="Number of classes sampled per episode")

# S and Q are where our first major problems start (as if we didn't
#   already have enough).
parser.add_argument("S",help="Number of support examples per class")
parser.add_argument("Q",help="Number of querey examples per class")

args = parser.parse_args()