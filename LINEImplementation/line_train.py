from config import args
import numpy as np
from utils.utils import VoseAlias
from utils.utils import makeDist, makeData
from utils.line import LINE
from tqdm import trange
import torch
import torch.optim as optim
import sys
import pickle

EMBEDDING_FILENAME = "line_node_embeddings.csv"

def setup(args):
    edgedistdict, nodedistdict, maxindex = makeDist(args.graph_path, args.negativepower)

    edgealiassampler = VoseAlias(edgedistdict)
    nodealiassampler = VoseAlias(nodedistdict)

    batchrange = int(len(edgedistdict) / args.batchsize)
    print(f"Number of nodes: {maxindex}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return maxindex, batchrange, edgealiassampler, nodealiassampler, device

def setupMethods(args, n_nodes, device):
    line_ord1 = LINE(n_nodes, embed_dim=args.dimension).to(device)
    line_ord2 = LINE(n_nodes, embed_dim=args.dimension).to(device)
    opt_ord1 = optim.Adam(line_ord1.parameters(), lr=args.learning_rate)
    opt_ord2 = optim.Adam(line_ord2.parameters(), lr=args.learning_rate)

    return line_ord1, line_ord2, opt_ord1, opt_ord2

def train(args, edgealiassampler, nodealiassampler, batchrange, 
          line_ord1, line_ord2, opt_ord1, opt_ord2, device):
    lossdata = {"iter": [], "loss_ord1": [], "loss_ord2": []}
    iter = 0

    print(f"\nTraining on {device}...\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for b in trange(batchrange):
            samplededges = edgealiassampler.sample_n(args.batchsize)
            batch = list(makeData(samplededges, args.negsamplesize, nodealiassampler))
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

def save(args, line_ord1, line_ord2, lossdata):
    print(f"\nDone training, saving models")

    torch.save(line_ord1.state_dict(), f"{args.save_path.split('.')[0]}_ord1.pt")
    torch.save(line_ord2.state_dict(), f"{args.save_path.split('.')[0]}_ord2.pt")

    print(f"Saving loss data at {args.lossdata_path}")
    with open(args.lossdata_path, "wb") as ldatafile:
        pickle.dump(lossdata, ldatafile)

    embeddings_ord1 = line_ord1.node_embeddings.weight.data
    embeddings_ord2 = line_ord2.contextnode_embeddings.weight.data

    final_emb = torch.cat([embeddings_ord1,
                        embeddings_ord2], dim=1)
    final_emb_np = final_emb.cpu().numpy()

    print(f"Saving embeddings at LINEImplementation/{EMBEDDING_FILENAME}")
    np.savetxt("LINEImplementation/" + EMBEDDING_FILENAME, final_emb_np, delimiter=",", fmt='%.6f')

    sys.exit()


if __name__ == "__main__":
    maxindex, batchrange, edgealiassampler, nodealiassampler, device = setup(args)
    line_ord1, line_ord2, opt_ord1, opt_ord2 = setupMethods(args, maxindex, device)
    line_ord1, line_ord2, opt_ord1, opt_ord2, lossdata = train(args, edgealiassampler, nodealiassampler, 
                                                            batchrange, line_ord1, line_ord2, opt_ord1, opt_ord2, device)
    save(args, line_ord1, line_ord2, lossdata)