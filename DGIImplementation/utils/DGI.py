#TODO: Cite ChatGPT, since it helped me fix up my code, and it provided some snippets
#      here and there when I needed them

from scipy.sparse import diags_array, eye_array
import networkx as nx
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(grandparent_dir)
from utils import UnequalAttributeCounts

class Encoder(nn.Module):
    def __init__(self, dim_in=32, dim_out=32):
        super().__init__()

        self.output_layer = nn.Linear(dim_in, dim_out)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(dim_out)

    def forward(self, X):
        X = self.output_layer(X)
        X = self.batch_norm(X)
        embeddings = self.activation(X)

        return embeddings
    
class Discriminator(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.output_layer = nn.Linear(dim_in, dim_out)
        self.activation = nn.Sigmoid()

    def forward(self, H, s):
        left_mult = self.output_layer(H)
        similarity_measure = left_mult @ s

        similarity_measure = self.activation(similarity_measure)

        return similarity_measure
    
class Readout(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.Sigmoid()

    def forward(self, H):
        s = H.mean(dim=0)
        s = self.activation(s)

        return s

class DeepGraphInfomax(nn.Module):
    def __init__(self, network: nx, embed_dim=32, K=2, lr=0.001, weight_decay=0.0001, epochs=200):
        super(DeepGraphInfomax, self).__init__()

        self.network = network
        self.num_nodes = nx.number_of_nodes(self.network)
        self.adj_matrix = nx.adjacency_matrix(self.network)
        self.adj_matrix_self_loops = self.adj_matrix + eye_array(self.num_nodes, self.num_nodes) # A + Identity matrix
        
        degrees = np.array(self.adj_matrix_self_loops.sum(axis=0)).flatten()
        deg_inv_sqrt = np.power(degrees, -0.5, where = degrees!=0)
        deg_inv_sqrt_matrix = diags_array(deg_inv_sqrt)

        norm_adj_matrix_self_loops = deg_inv_sqrt_matrix @ self.adj_matrix_self_loops @ deg_inv_sqrt_matrix

        coo = norm_adj_matrix_self_loops.tocoo()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        values = torch.tensor(coo.data, dtype=torch.float32)
        self.norm_adj_matrix_self_loops = torch.sparse_coo_tensor(indices, values, size=coo.shape, dtype=torch.float)
        
        self.embed_dim = embed_dim
        self.K = K
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.encoders = nn.ModuleList([Encoder(self.embed_dim, self.embed_dim)
                                        for k in range(1, self.K+1)])
        self.readout = Readout()
        self.discriminator = Discriminator(self.embed_dim, self.embed_dim)

        nodes_and_attributes = list(self.network.nodes(data=True))
        
        attribute_counts = [len(attributes) for node, attributes in nodes_and_attributes]
        if min(attribute_counts) != max(attribute_counts):
            raise UnequalAttributeCounts("Please make sure all nodes have the same number of features")
        
        num_features = attribute_counts[0]-1 # -1 because the "label" attribute isn't a node feature

        if num_features == 0:
            degrees = torch.tensor([d for n, d in self.network.degree()], dtype=torch.float)
            node_IDs = torch.tensor([id for id in range(1, self.num_nodes+1)], dtype=torch.float)
            
            self.unmapped_features = torch.stack([degrees, node_IDs], dim=1)
            self.feature_mapper = nn.Linear(2, embed_dim, bias=False)
        
        else:
            features_np = np.zeros((self.num_nodes, num_features))
            # Iterating over each node's features
            for node, features in nodes_and_attributes:

                feature_num = 0
                # Iterating over each feature (and its corresponding value)
                for feature, value in features.items():
                    if feature == "label": # we don't want the node label to be a feature
                        continue
                    # Assinging the corresponding value for this feature
                    features_np[node-1, feature_num] = value
                    feature_num += 1

            self.unmapped_features = torch.tensor(features_np, dtype=torch.float)
            self.feature_mapper = nn.Linear(num_features, embed_dim, bias=False)

    def forward(self, nodes, device):
        node_features = self.feature_mapper(self.unmapped_features)

        # Positive embeddings
        embeddings = node_features.to(device)
        embeddings = embeddings[nodes] # embeddings of only the training (or validation) nodes
        self.norm_adj_matrix_self_loops = self.norm_adj_matrix_self_loops.to(device)
        norm_adj_matrix_self_loops = self.norm_adj_matrix_self_loops.to_dense()[nodes][:, nodes]
        for k in range(1, self.K+1):
            embeddings = self.encoders[k-1](norm_adj_matrix_self_loops @ embeddings)
            embeddings = F.dropout(embeddings, training=self.training)

        summary = self.readout(embeddings)

        # Negative (corrupted) embeddings
        rand_perm = torch.randperm(embeddings.size(dim=0), device=embeddings.device)
        embeddings_corr = embeddings[rand_perm]
        for k in range(1, self.K+1):
            embeddings_corr = self.encoders[k-1](norm_adj_matrix_self_loops @ embeddings_corr)

        # Get discrimnator scores
        disc_score_pos = self.discriminator(embeddings, summary.view(self.embed_dim, 1))
        disc_score_neg = self.discriminator(embeddings_corr, summary.view(self.embed_dim, 1))

        # Loss
        loss = -(torch.log(disc_score_pos) + torch.log(1 - disc_score_neg)).mean()

        return loss