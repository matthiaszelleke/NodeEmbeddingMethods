#TODO: Cite ChatGPT, since it helped me fix up my code, and it provided some snippets
#      here and there when I needed them

from LINEImplementation.utils.utils import makeDist, VoseAlias
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import sample_neighbourhood, UnspecifiedNumberOfFeatures, UnequalAttributeCounts

class MaxPoolingAggregator(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.output_layer = nn.Linear(dim_in, dim_out)
        self.activation = nn.ReLU()

    def forward(self, neighbour_embeddings):
        neighbour_embeddings = self.output_layer(neighbour_embeddings)
        neighbour_embeddings = self.activation(neighbour_embeddings)

        return torch.max(neighbour_embeddings, dim=0).values

class Updater(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.output_layer = nn.Linear(dim_in, dim_out, bias=False)
        self.activation = nn.ReLU()

    def forward(self, concatenated):
        new_embedding = self.output_layer(concatenated)
        new_embedding = self.activation(new_embedding)

        return new_embedding

class GraphSage(nn.Module):
    def __init__(self, network, neighbourhood_sizes, depth=2, embed_dim=32, num_learnable_features=None,
                 num_walks=10, walk_length=80, window_size=5, neg_sample_size=5):
        super(GraphSage, self).__init__()

        self.embed_dim = embed_dim
        self.network = network
        self.num_nodes = self.network.number_of_nodes()

        self.neighbourhood_sizes = neighbourhood_sizes

        self.depth = depth

        self.node_embeddings = nn.Embedding(self.num_nodes, self.embed_dim)

        self.agg = MaxPoolingAggregator(dim_in=self.embed_dim, dim_out=self.embed_dim)
        self.upd = Updater(dim_in = 2*self.embed_dim, dim_out = self.embed_dim)
        
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size

        _, nodedistdict = makeDist(network, power=0.75)
        self.nodealiassampler = VoseAlias(nodedistdict)

        nodes_and_attributes = list(self.network.nodes(data=True))

        attribute_counts = [len(attributes) for node, attributes in nodes_and_attributes]
        if min(attribute_counts) != max(attribute_counts):
            raise UnequalAttributeCounts("Please make sure all nodes have the same number of features")
        
        num_features = attribute_counts[0]-1 # -1 because the "label" attribute isn't a node feature

        if num_features == 0:
            if num_learnable_features is None:
                raise UnspecifiedNumberOfFeatures("Please specify how many learnable features you want each node to have")
            
            self.node_features = nn.Embedding(self.num_nodes, num_learnable_features, dtype=torch.float)
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

            # Register as a buffer so it's moved to the right device with model.to()
            self.register_buffer("node_features", torch.tensor(features_np, dtype=torch.float))

    def sample_neighbourhoods(self, batch_nodes):
        self.neighbourhoods = [[] for i in range(self.depth+1)] # +1 because there are technically K+1 layers: 0,1,...,K
        # 1st dim is the nodes, 2nd dim is what the cur value of k is, 3rd dim is the nodes at that k which were sampled as the neighbour of a given node 
        self.neighbours = np.zeros((self.num_nodes, self.depth+1, self.num_nodes))
        
        self.neighbourhoods[self.depth] = batch_nodes
        for k in reversed(range(1, self.depth+1)):
            self.neighbourhoods[k-1] = list(self.neighbourhoods[k])
            for node in self.neighbourhoods[k]:
                current_neighbours = sample_neighbourhood(self.network, node, self.neighbourhood_sizes[k-1])

                self.neighbours[node-1, k, np.array(current_neighbours) - 1] = 1
                self.neighbourhoods[k-1].extend(current_neighbours)
    
    def forward(self, batch_nodes, device):
        GraphSage.sample_neighbourhoods(self, batch_nodes)

        embeddings = self.node_features.weight.to(device)
        for k in range(1, self.depth+1):
            new_embeddings = embeddings.clone()
            for node in self.neighbourhoods[k]:
                current_neighbours = np.where(self.neighbours[node-1, k] == 1)[0] + 1
                current_neighbours = torch.tensor(current_neighbours, dtype=torch.long, device=device)

                neighbour_embeddings = embeddings[current_neighbours - 1]
                aggregated = self.agg(neighbour_embeddings)

                current_embedding = embeddings[node-1]
                concatenated = torch.cat((current_embedding, aggregated), dim=0)

                updated = self.upd(concatenated)
                updated = F.normalize(updated, dim=0)

                new_embeddings[node-1] = updated

            embeddings = new_embeddings

        return embeddings