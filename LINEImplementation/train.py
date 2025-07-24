from config import args
from utils.utils import VoseAlias
from utils.utils import makeDist, makeData
from utils.line import LINE
from tqdm import trange
import torch
import torch.optim as optim
import sys
import pickle

def setup(args):
    # Creating the node and edge probability distribution
    edgedistdict, nodedistdict, maxindex = makeDist(args.graph_path, args.negativepower)

    # Objects to be used for sampling, based on the created distributions
    edgealiassampler = VoseAlias(edgedistdict)
    nodealiassampler = VoseAlias(nodedistdict)

    # Number of training batches for each epoch
    batchrange = int(len(edgedistdict) / args.batchsize)

    print(f"Number of nodes: {maxindex}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    return maxindex, batchrange, edgealiassampler, nodealiassampler, device

def setupMethods(args, n_nodes, device):
    # Instantiating LINE models
    line_ord1 = LINE(n_nodes, embed_dim=args.dimension).to(device)
    line_ord2 = LINE(n_nodes, embed_dim=args.dimension).to(device)

    # Choosing the Adam optimizer, which can tweak the learning rate to speed up convergence
    opt_ord1 = optim.Adam(line_ord1.parameters(), lr=args.learning_rate)
    opt_ord2 = optim.Adam(line_ord2.parameters(), lr=args.learning_rate)

    return line_ord1, line_ord2, opt_ord1, opt_ord2

def train(args, edgealiassampler, nodealiassampler, batchrange, 
          line_ord1, line_ord2, opt_ord1, opt_ord2, device):
    # Will store the losses, for both order 1 and order 2, in
    # each iteration (over all epochs)
    lossdata = {"iter": [], "loss_ord1": [], "loss_ord2": []}
    iter = 0

    print(f"\nTraining on {device}...\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for b in trange(batchrange):
            # Sampling edges
            samplededges = edgealiassampler.sample_n(args.batchsize)

            # Sampling negative nodes
            batch = list(makeData(samplededges, args.negsamplesize, nodealiassampler))
            batch = torch.tensor(batch, dtype=torch.long).to(device)

            # Current node (when computing order 2 proximity)
            v_i = batch[:, 0] - 1  # Changing from 1-indexed to 0-indexed 

            # Context node (when computing order 2 proximity)
            v_j = batch[:, 1] - 1

            # Negative nodes produced by negative sampling
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
            lossdata["loss_ord1"].append(loss_ord1.item())
            lossdata["loss_ord2"].append(loss_ord2.item())
            iter += 1

    return line_ord1, line_ord2, opt_ord1, opt_ord2, lossdata

def save(args, line_ord1, line_ord2, lossdata):
    print(f"\nDone training, saving models")

    torch.save(line_ord1.state_dict(), f"{args.save_path.split('.')[0]}_ord1.pt")
    torch.save(line_ord2.state_dict(), f"{args.save_path.split('.')[0]}_ord2.pt")

    print(f"Saving loss data at {args.lossdata_path}")
    with open(args.lossdata_path, "wb") as ldatafile:
        # Saving the order 1 and order 2 loss data
        pickle.dump(lossdata, ldatafile)

    sys.exit()


if __name__ == "__main__":
    maxindex, batchrange, edgealiassampler, nodealiassampler, device = setup(args)
    line_ord1, line_ord2, opt_ord1, opt_ord2 = setupMethods(args, maxindex, device)
    line_ord1, line_ord2, opt_ord1, opt_ord2, lossdata = train(args, edgealiassampler, nodealiassampler, 
                                                            batchrange, line_ord1, line_ord2, opt_ord1, opt_ord2, device)
    save(args, line_ord1, line_ord2, lossdata)