import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model_and_labels_fname", "--model_and_labels_filename", type=str)
parser.add_argument("-plot_fname", "--plot_filename", type=str)

args, _ = parser.parse_known_args()