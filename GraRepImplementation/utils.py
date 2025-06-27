import numpy as np
from tqdm import tqdm

def read_graph_edgelist(fname):
    print("Reading edgelist file...")

    n_lines = 0
    max_index = 1

    # Calculating the number of nodes in the graph and
    # the number of lines in the file 
    with open(fname, "r") as f:
        for line in f:
            n_lines += 1
            vertex1, vertex2, _ = line.replace("\n", "").split(sep=" ")
            vertex1, vertex2 = int(vertex1), int(vertex2)

            if vertex1 > max_index:
                max_index = vertex1
            elif vertex2 > max_index:
                max_index = vertex2

    # Constructing the graph's adjacency matrix
    with open(fname, "r") as f:
        adj_matrix = np.zeros(shape=(max_index, max_index), dtype=np.float32)
        for line in tqdm(f, total=n_lines):
            vertex1, vertex2, weight = line.replace("\n", "").split(sep=" ")
            vertex1, vertex2, weight = int(vertex1), int(vertex2), int(weight)
            
            # Subtract 1 since the nodes in the file are 1-indexed
            adj_matrix[vertex1 - 1, vertex2 - 1] = weight
            adj_matrix[vertex2 - 1, vertex1 - 1] = weight # Because graph insundirected

    return adj_matrix