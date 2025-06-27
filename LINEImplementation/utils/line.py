import torch
import torch.nn as nn
import torch.nn.functional as F

class LINE(nn.Module):
    def __init__(self, n_nodes, embed_dim=128):
        super(LINE, self).__init__()

        self.embed_dim = embed_dim
        self.node_embeddings = nn.Embedding(n_nodes, embed_dim) # node embeddings
        self.contextnode_embeddings = nn.Embedding(n_nodes, embed_dim) # context node embeddings (i.e. when the node is viewed as the context for another node)

        initrange = 2.0 # used to define the allowed range of initial values for entries in the embeddings
        
        # divide by embed_dim to prevent the size/norm of the embeddings from becoming large
        # when the # of embedding dimesions is large
        self.node_embeddings.weight.data = self.node_embeddings.weight.data.uniform_(-initrange, initrange) / embed_dim
        self.contextnode_embeddings.weight.data = self.contextnode_embeddings.weight.data.uniform_(-initrange, initrange) / embed_dim

    def forward_ord1(self, v_i, v_j, negsamples, device):

        v_i = self.node_embeddings(v_i).to(device)
        v_j = self.node_embeddings(v_j).to(device)

        # the embeddings of the context nodes for the "negative" nodes chosen during negative sampling
        # the embedding is negated because we will later want the negative of the dot product between node i and the negative node
        negative_nodes = -self.node_embeddings(negsamples).to(device)

        mul_positive_batch = torch.mul(v_i, v_j) # element-wise product
        positive_batch = F.logsigmoid(torch.sum(mul_positive_batch, dim=1)) # summing to get dot product

        # reshaping and then element-wise product with negative samples
        mul_negative_batch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negative_nodes)
        
        # applying logsigmoid and then summing over the negative samples
        negative_batch = torch.sum(F.logsigmoid(
            torch.sum(mul_negative_batch, dim=2) # summing to get dot product
        ),
        dim=1)

        loss = positive_batch + negative_batch
        return -torch.mean(loss) # average loss across nodes in the batch
    
    
    def forward_ord2(self, v_i, v_j, negsamples, device):

        v_i = self.node_embeddings(v_i).to(device)
        v_j = self.contextnode_embeddings(v_j).to(device)

        # the embeddings of the context nodes for the "negative" nodes chosen during negative sampling
        # the embedding is negated because we will later want the negative of the dot product between node i and the negative node
        negative_nodes = -self.contextnode_embeddings(negsamples).to(device)

        mul_positive_batch = torch.mul(v_i, v_j) # element-wise product
        positive_batch = F.logsigmoid(torch.sum(mul_positive_batch, dim=1)) # summing to get dot product

        # reshaping and then element-wise product with negative samples
        mul_negative_batch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negative_nodes)

        # applying logsigmoid and then summing over the negative samples
        negative_batch = torch.sum(F.logsigmoid(
            torch.sum(mul_negative_batch, dim=2) # summing to get dot product
        ),
        dim=1)

        loss = positive_batch + negative_batch
        return -torch.mean(loss) # average loss across nodes in the batch
    

    def forward(self, v_i, v_j, negsamples, order, device):
        if order == 1:
            return LINE.forward_ord1(self, v_i, v_j, negsamples, device)
        elif order == 2:
            return LINE.forward_ord2(self, v_i, v_j, negsamples, device)
        else:
            print(f"Error: Order must be either 1 or 2, the value {order} is not allowed")
            return
