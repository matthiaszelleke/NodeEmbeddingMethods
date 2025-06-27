import numpy as np
import pandas as pd
from scipy.linalg import svd, sqrtm
from tqdm import tqdm

class GraRep(object):
    def __init__(self, A, args):
        self.A = np.array(A, dtype="float32")
        self.num_nodes = self.A.shape[0]
        self.args = args
        self.normalize()

    def normalize(self):
        # Creating the normalized adjacency matrix (1-step probability
        # matrix) which will be the base case for computing its higher powers
        row_sums = self.A.sum(axis=1)
        # Only performing the division when the sum is non-zero, otherwise
        # output a zero if the sum is zero
        self.A_hat = self.A * (np.divide(1, row_sums[:, np.newaxis], 
                                         out=np.zeros_like(row_sums[:, np.newaxis]), 
                                         where = row_sums[:, np.newaxis]!=0))

    def learn_embeddings(self):
        self.embeddings = np.zeros(shape=(self.num_nodes, self.args.dimension*self.args.order))

        # Learning the embeddings for each value of k (step)
        for step in tqdm(range(1, self.args.order + 1)):

            # Represents how likely a node is to appear in a random walk in general
            surprise_factor = self.A_hat.sum(axis=0)

            # Scaling values in the transition matrix based on how surprising a given transition is
            # Surprising transitions (low raw likelihood) get larger values, and vice versa
            # The constant term added is the same one used in the paper  
            self.log_trans_matrix = np.log(np.divide(self.A_hat, surprise_factor[np.newaxis, :], 
                                                     out=np.zeros_like(self.A_hat), 
                                                     where = surprise_factor[np.newaxis, :]!=0) # Only dividing when denom. != 0 like before
                                     + np.log(self.num_nodes))

            negative_indices = self.log_trans_matrix < 0
            self.log_trans_matrix[negative_indices] = 0 # Setting negative entries in the log transition matrix to 0

            self.U, self.Sigma, _ = svd(self.log_trans_matrix)

            # Using only the top r components, as specified by the dimension param
            self.U = self.U[:, :self.args.dimension]
            self.Sigma = self.Sigma[:self.args.dimension]

            # Obtain the row embeddings using the left singular vectors (U)
            # that's why the V^T matrix isn't needed (it would give column embeddings)
            embeddings = self.U @ sqrtm(np.diag(self.Sigma))

            # Append obtained embeddings for this k (step)
            self.embeddings[:, (step-1)*self.args.dimension:step*self.args.dimension] = embeddings

            # Compute next power of transition matrix
            self.A_hat = self.A @ self.A_hat

    def save_embeddings(self):
        node_labels = np.array([idx for idx in range(1, self.num_nodes + 1)])

        # Add node labels to embeddings
        self.embeddings = pd.DataFrame(self.embeddings, index=node_labels)

        with open(self.args.embeddings_file, "w") as fname:
            self.embeddings.to_csv(fname, index=True) # Include node labels