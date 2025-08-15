# Adapted from: https://github.com/dmpierre/LINE
# Copyright (c) 2025 Pierre Daix-Moreux
# Licensed under the MIT License (see LICENSE or LICENSE-pierre in this repository)
# Implements the LINE algorithm from:
# Tang et al. (2015). "LINE: Large-scale Information Network Embedding"

from .config import args
import networkx as nx
import numpy as np
import pickle
import torch
import torch.optim as optim
from tqdm import trange
from .utils.line import LINE
from .utils.utils import VoseAlias
from .utils.utils import makeDist, makeData


def setup(network, batch_size, negative_power):
    edgedistdict, nodedistdict = makeDist(network, negative_power)

    edgealiassampler = VoseAlias(edgedistdict)
    nodealiassampler = VoseAlias(nodedistdict)

    batchrange = int(len(edgedistdict) / batch_size)
    print(f"Number of nodes: {network.number_of_nodes()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return batchrange, edgealiassampler, nodealiassampler, device

def setupMethods(network, dimensions, learning_rate, device):
    line_ord1 = LINE(network.number_of_nodes(), embed_dim=dimensions).to(device)
    line_ord2 = LINE(network.number_of_nodes(), embed_dim=dimensions).to(device)

    opt_ord1 = optim.Adam(line_ord1.parameters(), lr=learning_rate)
    opt_ord2 = optim.Adam(line_ord2.parameters(), lr=learning_rate)

    return line_ord1, line_ord2, opt_ord1, opt_ord2

def train(epochs, batch_size, size_of_negative_sample, edgealiassampler, nodealiassampler, 
          batchrange, line_ord1, line_ord2, opt_ord1, opt_ord2, device):
    lossdata = {"iter": [], "loss_ord1": [], "loss_ord2": []}
    iter = 0

    print(f"\nTraining on {device}...\n")

    for epoch in range(epochs):
        for b in range(batchrange):
            samplededges = edgealiassampler.sample_n(batch_size)
            batch = list(makeData(samplededges, size_of_negative_sample, nodealiassampler))
            batch = torch.tensor(batch, dtype=torch.long).to(device)

            v_i = batch[:, 0] - 1
            v_j = batch[:, 1] - 1
            negsamples = batch[:, 2:] - 1 

            line_ord1.zero_grad()
            loss_ord1 = line_ord1(v_i, v_j, negsamples, 1, device)
            loss_ord1.backward()
            opt_ord1.step()

            line_ord2.zero_grad()
            loss_ord2 = line_ord2(v_i, v_j, negsamples, 2, device)
            loss_ord2.backward()
            opt_ord2.step()

            lossdata["iter"].append(iter)
            lossdata["loss_ord1"].append(loss_ord1.item())  # .item() to convert tensor to Python float
            lossdata["loss_ord2"].append(loss_ord2.item())
            iter += 1

    return line_ord1, line_ord2, opt_ord1, opt_ord2, lossdata

def save(model_save_path, loss_data_path, line_ord1, line_ord2, lossdata, 
         embeddings_filename):
    print(f"\nDone training, saving models")

    torch.save(line_ord1.state_dict(), f"{model_save_path.split('.')[0]}_ord1.pt")
    torch.save(line_ord2.state_dict(), f"{model_save_path.split('.')[0]}_ord2.pt")

    print(f"Saving loss data at {loss_data_path}")
    with open(loss_data_path, "wb") as ldatafile:
        pickle.dump(lossdata, ldatafile)

    embeddings_ord1 = line_ord1.node_embeddings.weight.data
    embeddings_ord2 = line_ord2.contextnode_embeddings.weight.data

    final_emb = torch.cat([embeddings_ord1,
                        embeddings_ord2], dim=1)
    final_emb_np = final_emb.cpu().numpy()

    print(f"Saving embeddings at LINEImplementation/{embeddings_filename}")
    np.savetxt("LINEImplementation/"+embeddings_filename, final_emb_np, delimiter=",", fmt='%.6f')

    return final_emb_np

def get_embeddings_line(network, epochs=10, batch_size=64, learning_rate=0.025,
                        size_of_negative_sample=5, embeddings_filename="line_node_embeddings.csv",
                        dimensions=16, negative_power=0.75, 
                        model_save_path="LINEImplementation/LINE_model.pt",
                        loss_data_path="LINEImplementation/loss_data.pkl"):
    
    batchrange, edgealiassampler, nodealiassampler, device = setup(network, batch_size, negative_power)
    line_ord1, line_ord2, opt_ord1, opt_ord2 = setupMethods(network, dimensions, learning_rate, device)
    line_ord1, line_ord2, opt_ord1, opt_ord2, lossdata = train(epochs, batch_size, size_of_negative_sample, edgealiassampler, nodealiassampler, 
                                                               batchrange, line_ord1, line_ord2, opt_ord1, opt_ord2, device)
    embeddings = save(model_save_path, loss_data_path, line_ord1, line_ord2, lossdata, embeddings_filename)

    return embeddings

if __name__ == "__main__":
    network = nx.read_edgelist(args.network)

    get_embeddings_line(network, epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.lr, size_of_negative_sample=args.neg_sample_size,
                        embeddings_filename=args.emb_fname, dimensions=args.dim, 
                        negative_power=args.neg_pow, model_save_path=args.save_path,
                        loss_data_path=args.loss_data_path)