from LINEImplementation.utils.utils import negSampleBatch
import networkx as nx
import numpy as np
import random
from scipy.sparse import coo_matrix
import torch
import torch.nn.functional as F

class UnspecifiedNumberOfFeatures(Exception):
    """Custom exception to handle the case when the user's
       network doesn't have node features, and they don't
       specify the number of learnable features they want
       for each node"""
    pass

class UnequalAttributeCounts(Exception):
    """Custom exception to handle the case when the nodes
       in a user's network don't all have the same number of
       attributes"""
    pass

def sample_neighbourhood(network, node, size):
    neighbours = list(network.neighbors(node))
    if len(neighbours) < size:
        sampled_neighbours = random.choices(population=neighbours, k=size)
    else:
        sampled_neighbours = random.sample(population=neighbours, k=size)

    return sampled_neighbours

def generate_weighted_random_walks(network, num_walks=10, walk_length=80, weight_attr='weight'):
    """
    Generate `num_walks` random walks from each node,
    where next step is chosen by weighted sampling of neighbors,
    including the previous node (backtracking) as a neighbor.

    If no weight attribute found, treats graph as unweighted (equal probs).

    Args:
        network: networkx graph (weighted or unweighted)
        num_walks: number of walks per node
        walk_length: length of each walk
        weight_attr: edge attribute name storing weights (default 'weight')

    Returns:
        List of random walks (each a list of node IDs)
    """
    all_walks = []
    nodes = list(network.nodes())

    for node in nodes:
        for _ in range(num_walks):
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = list(network.neighbors(curr))
                if len(neighbors) == 0:
                    # dead end: stop early
                    break

                weights = []
                for nbr in neighbors:
                    edge_data = network.get_edge_data(curr, nbr, default=None)
                    if edge_data is None or weight_attr not in edge_data:
                        # Unweighted edge or no weight attribute: assign equal weight 1
                        w = 1
                    else:
                        w = edge_data[weight_attr]
                    weights.append(w)

                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]

                next_node = random.choices(neighbors, weights=probs, k=1)[0]
                walk.append(next_node)

            all_walks.append(walk)

    return all_walks

def get_positive_pairs(random_walks, num_nodes, walk_length=80, window_size=5):
    rows = []
    cols = []
    for random_walk in random_walks:
            left = 0
            while left < walk_length:
                right = left+1
                while right<=(left+window_size) and right<walk_length:
                    node_u = random_walk[left]
                    node_v = random_walk[right]

                    rows.append(node_u-1) # since array indices are 0-indexed but node IDs are 1-indexed
                    cols.append(node_v-1)
                    rows.append(node_v-1)
                    cols.append(node_u-1)

                    right += 1
                left += 1

    # equals 1 if nodes u and v co-occur on a random walk of a fixed length
    positive_pairs = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(num_nodes, num_nodes))
    positive_pairs.tocsr()
    positive_pairs = positive_pairs.astype(bool)

    return positive_pairs

def sample_negatives_with_rejection(node_u, node_v, neg_sample_size, nodealiassampler):
    neg_nodes = []
    while len(neg_nodes) < neg_sample_size:
        candidates = list(nodealiassampler.sample_n(neg_sample_size * 2))
        filtered = [c-1 for c in candidates if c-1 != node_u and c-1 != node_v]
        
        neg_nodes.extend(filtered)
    
    return neg_nodes[:neg_sample_size]

def loss_function(embeddings, network, batch_size, nodealiassampler, 
                  num_walks=10, walk_length=80, window_size=5, neg_sample_size=5):

    loss = 0.0
    device = embeddings.device

    # Generating positive pairs
    random_walks = generate_weighted_random_walks(network, num_walks, walk_length)
    positive_pairs = get_positive_pairs(
        random_walks,
        network.number_of_nodes(),
        walk_length,
        window_size
    )
    rows, cols = positive_pairs.nonzero()

    num_pairs = len(rows)
    embeddings = embeddings.to(device)

    # Processing in batches
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        batch_u = rows[start:end]
        batch_v = cols[start:end]

        # Positive embeddings
        z_u_batch = embeddings[batch_u]           # (B, D)
        z_v_batch = embeddings[batch_v]           # (B, D)

        # Positive loss
        pos_score = torch.sum(z_u_batch * z_v_batch, dim=1)  # (B,)
        positive_part = -F.logsigmoid(pos_score)             # (B,)

        # Sampling negatives for each u in the batch
        neg_samples = torch.stack([
            torch.tensor(sample_negatives_with_rejection(u.item(), v.item(), neg_sample_size, nodealiassampler),
                        device=device) for u, v in zip(batch_u, batch_v)
        ])  # Shape: (B, K)

        # Vectorizing negative loss
        z_neg_batch = embeddings[neg_samples]  # (B, K, D)
        # Dot product between u and each of its K negatives
        neg_scores = torch.bmm(z_neg_batch, z_u_batch.unsqueeze(2)).squeeze(2)  # (B, K)
        negative_part = torch.sum(F.logsigmoid(-neg_scores), dim=1)  # (B,)

        # Accumulating loss
        loss += torch.sum(positive_part - negative_part)

    return loss