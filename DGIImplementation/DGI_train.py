from .config import args
import copy
import networkx as nx
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from .utils.DGI import DeepGraphInfomax

def setup(network, embed_dim=32, K=2, lr=0.001, weight_decay=0.0001, num_epochs=200):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dgi = DeepGraphInfomax(network, embed_dim, K, lr, weight_decay, num_epochs)
    
    opt = optim.Adam(dgi.parameters(), lr=lr, weight_decay=dgi.weight_decay)

    return dgi, opt, device

def train(dgi, opt, num_epochs, device, val_split=0.1, patience=2):
    print(f"\nTraining on {device}...\n")

    all_nodes = list(range(dgi.num_nodes))
    random.shuffle(all_nodes)
    split_idx = int(len(all_nodes) * (1 - val_split))
    train_nodes = all_nodes[:split_idx]
    val_nodes = all_nodes[split_idx:]

    best_loss = float("inf")
    best_state_dict = None
    best_embeddings = None
    patience_counter = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")

        dgi.train()
        dgi.zero_grad()
        loss = dgi(train_nodes, device)

        loss.backward()
        opt.step()

        dgi.eval()
        embeddings = None
        with torch.no_grad():
            node_features = dgi.feature_mapper(dgi.unmapped_features)
            embeddings = node_features.to(device)
            embeddings = embeddings[train_nodes]
            norm_adj_matrix_self_loops_train = dgi.norm_adj_matrix_self_loops.to_dense()[train_nodes][:, train_nodes]
            for k in range(1, dgi.K+1):
                embeddings = dgi.encoders[k-1](norm_adj_matrix_self_loops_train @ embeddings)

            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Validation step
        dgi.eval()
        with torch.no_grad():
            val_loss = dgi(val_nodes, device).item()
        print(f"Validation Loss: {val_loss:.4f}")

         # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(dgi.state_dict())
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    dgi.load_state_dict(best_state_dict)
    dgi.eval()
    
    with torch.no_grad():
        node_features = dgi.feature_mapper(dgi.unmapped_features).to(device)
        embeddings = node_features
        for k in range(1, dgi.K+1):
            embeddings = dgi.encoders[k-1](dgi.norm_adj_matrix_self_loops @ embeddings)

        best_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return dgi, best_embeddings

def save(model, embeddings, model_save_path, embeddings_filename):
    
    print(f"\nDone training, saving model at DGIImplementation/{model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    embeddings = embeddings.detach().cpu().numpy()

    print(f"Saving embeddings at DGIImplementation/{embeddings_filename}")
    np.savetxt("DGIImplementation/"+embeddings_filename, embeddings, delimiter=",", fmt='%.6f')

    return embeddings

def get_embeddings_dgi(network, num_epochs, K, embed_dim=32, lr=0.001, 
                       weight_decay=0.0001, patience=2,
                       embeddings_filename="dgi_node_embeddings.csv", 
                       model_save_path="DGIImplementation/dgi_model.pt"):
    
    dgi, opt, device = setup(network, embed_dim, K, lr, weight_decay, num_epochs)
    
    dgi, embeddings = train(dgi, opt, num_epochs, device, 0.1, patience)
    
    embeddings = save(dgi, embeddings, model_save_path, embeddings_filename)

    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network)
    get_embeddings_dgi(network, args.num_epochs, args.K, args.dim, args.lr, 
                       args.weight_decay, args.patience, args.emb_fname, 
                       args.model_save_path)