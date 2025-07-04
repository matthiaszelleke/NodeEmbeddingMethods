import argparse

parser = argparse.ArgumentParser()

# Probability parameter for diagonals of block connecitivity matrix
parser.add_argument("-p", "--probability", type=float)

args = parser.parse_args()