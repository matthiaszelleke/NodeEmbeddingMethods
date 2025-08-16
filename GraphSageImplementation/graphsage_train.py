from .config import args
import networkx as nx
import numpy as np
import random
import torch
import torch.optim as optim
from tqdm import trange
from .utils.graphsage import GraphSage
from .utils.utils import get_batch_size, loss_function

def setup(network, K, neigh_sizes, embed_dim=32, lr=0.001, num_walks=10, 
          walk_length=80, window_size=5, neg_sample_size=5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    graph_sage = GraphSage(network, neigh_sizes, K, embed_dim, num_walks, 
                           walk_length, window_size, neg_sample_size)
    
    opt = optim.Adam(graph_sage.parameters(), lr=lr, weight_decay=0.0001)

    return graph_sage, opt, device

def train(graph_sage, opt, num_epochs, device, val_split=0.1, patience=2):
    print(f"\nTraining on {device}...\n")

    all_nodes = list(range(1, graph_sage.num_nodes+1))
    random.shuffle(all_nodes)
    split_idx = int(len(all_nodes) * (1 - val_split))
    train_nodes = all_nodes[:split_idx]
    val_nodes = all_nodes[split_idx:]

    batch_size = get_batch_size(len(train_nodes))
    num_batches = (len(train_nodes) + batch_size - 1) // batch_size

    best_loss = float('inf')
    patience_counter = 0
    best_embeddings = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")

        # Training step
        random.shuffle(train_nodes)
        graph_sage.train()
        for batch_num in trange(num_batches):
            batch_nodes = train_nodes[batch_num*batch_size : (batch_num+1)*batch_size]

            graph_sage.zero_grad()
            train_embeddings = graph_sage(batch_nodes, device)
            loss = loss_function(train_embeddings, graph_sage.network, batch_size, graph_sage.nodealiassampler, 
                                 graph_sage.num_walks, graph_sage.walk_length, graph_sage.window_size, 
                                 graph_sage.neg_sample_size)
            loss.backward()
            opt.step()

        # Validation step
        graph_sage.eval()
        with torch.no_grad():
            val_embeddings = graph_sage(val_nodes, device)
            val_loss = loss_function(
                val_embeddings, 
                graph_sage.network, 
                graph_sage.num_nodes, 
                graph_sage.nodealiassampler, 
                graph_sage.num_walks, 
                graph_sage.walk_length, 
                graph_sage.window_size, 
                graph_sage.neg_sample_size
            ).item()
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            with torch.no_grad():
                all_nodes = list(range(1, graph_sage.num_nodes+1))
                best_embeddings = graph_sage(all_nodes, device)  # single forward pass
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                return graph_sage, best_embeddings
            
    return graph_sage, best_embeddings

def save(model, embeddings, model_save_path, embeddings_filename):
    
    print(f"\nDone training, saving model at GraphSageImplementation/{model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    embeddings = embeddings.detach().cpu().numpy()

    print(f"Saving embeddings at GraphSageImplementation/{embeddings_filename}")
    np.savetxt("GraphSageImplementation/"+embeddings_filename, embeddings, delimiter=",", fmt='%.6f')

    return embeddings

def get_embeddings_graphsage(network, num_epochs, K, neigh_sizes, embed_dim=32, 
                             lr=0.001, num_walks=10, walk_length=80, window_size=5, 
                             neg_sample_size=5, patience=2,
                             embeddings_filename="graphsage_node_embeddings.csv", 
                             model_save_path="GraphSageImplementation/GraphSage_model.pt"):
    
    graph_sage, opt, device = setup(network, K, neigh_sizes, embed_dim, lr, num_walks, 
                                    walk_length, window_size, neg_sample_size)
    
    graph_sage, embeddings = train(graph_sage, opt, num_epochs, device, 0.1, patience)
    
    embeddings = save(graph_sage, embeddings, model_save_path, embeddings_filename)

    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network)
    get_embeddings_graphsage(network, args.num_epochs, args.K, args.neigh_sizes, args.dim, 
                             args.lr, args.num_walks, args.walk_length, args.window_size, 
                             args.Q, args.patience, args.emb_fname, args.model_save_path)