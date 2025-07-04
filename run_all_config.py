import argparse

parser = argparse.ArgumentParser()

# Probability parameter for diagonals of block connecitivity matrix
parser.add_argument("-p1", "--probability1", type=float, default=0.5)
parser.add_argument("-p2", "--probability2", type=float, default=0.7)
parser.add_argument("-p3", "--probability3", type=float, default=0.9)

args = parser.parse_args()