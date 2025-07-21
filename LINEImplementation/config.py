import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--graph_path", type=str)
parser.add_argument("-save", "--save_path", type=str)
parser.add_argument("-lossdata", "--lossdata_path", type=str)

# Hyperparams.
parser.add_argument("-order", "--order", type=int, default=2)
parser.add_argument("-neg", "--negsamplesize", type=int, default=5)
parser.add_argument("-dim", "--dimension", type=int, default=64)
parser.add_argument("-batchsize", "--batchsize", type=int, default=512)
parser.add_argument("-epochs", "--epochs", type=int, default=10)
parser.add_argument("-lr", "--learning_rate", type=float,
                    default=0.025)  # As starting value in paper
parser.add_argument("-negpow", "--negativepower", type=float, default=0.75)
args = parser.parse_args()