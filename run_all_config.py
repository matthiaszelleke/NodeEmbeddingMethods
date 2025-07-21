import argparse

parser = argparse.ArgumentParser()

# Probability parameter for diagonals of block connecitivity matrix
parser.add_argument("-p1", "--probability1", type=float)
parser.add_argument("-p2", "--probability2", type=float)
parser.add_argument("-p3", "--probability3", type=float)

args = parser.parse_args()