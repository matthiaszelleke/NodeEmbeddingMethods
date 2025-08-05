import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-num_nodes", "--num_nodes", type=int, default=1000)
parser.add_argument("-p1", "--probability1", type=float, default=0.5)
parser.add_argument("-p2", "--probability2", type=float, default=0.7)
parser.add_argument("-p3", "--probability3", type=float, default=0.9)

args, _ = parser.parse_known_args()