import numpy as np
import pickle
from run_all_config import args
from sklearn.metrics import classification_report, accuracy_score
import subprocess

NUM_METHODS = 4

# Creating the Stochastic Block Model network with the specified probs on the diagonal
# The network will be stored in an edgelist file
subprocess.run(["python", "sbm_simulator.py", "-p1", str(args.probability1),
                                              "-p2", str(args.probability2),
                                              "-p3", str(args.probability3)], 
                check=True)


# Learning the SBM's node embeddings using spectral clustering and training a
# logistic regression model to predict the node's class/label
print("Running Spectral Clustering...")

subprocess.run(["python", "SpectralClusteringImplementation/train.py"], 
                check=True)
subprocess.run(["python", "SpectralClusteringImplementation/logistic_regression.py"],
                check=True)

# Plotting the actual labels for the test set, along with prediction regions for the 
# multiclass logistic regression model
subprocess.run(["python", "SpectralClusteringImplementation/plot_embeddings.py"], check=True)


# Learning the node embeddings using the node2vec method
print("Running Node2vec...")
subprocess.run(["python", "Node2vecImplementation/train.py"], check=True)

# Training a logistic regression model to predict the node's class/label
subprocess.run(["python", "Node2vecImplementation/logistic_regression.py"], check=True)

# Plotting the actual labels for the test set, along with prediction regions for the 
# multiclass logistic regression model
subprocess.run(["python", "Node2vecImplementation/plot_embeddings.py"], check=True)


# Learning the node embeddings using LINE method
print("Running LINE...") 
line_subprocess = subprocess.Popen(["python", "LINEImplementation/train.py", "-g", 
                    "./sbm_graph.edgelist", "-save", "LINEImplementation/LINE_model.pt", 
                    "-lossdata", "LINEImplementation/loss_data.pkl", 
                    "-epochs", "10", "-batchsize", "512", "-dim", "64"])
line_subprocess.wait()

# Training a logistic regression model to predict the node's class/label
subprocess.run(["python", "LINEImplementation/logistic_regression.py"], check=True)

# Plotting the actual labels for the test set, along with prediction regions for the 
# multiclass logistic regression model
subprocess.run(["python", "LINEImplementation/plot_embeddings.py"], check=True)


# Learning the node embeddings using GraRep method
print("Running GraRep...")
grarep_subprocess = subprocess.Popen(["python", "GraRepImplementation/train.py", "-g", "./sbm_graph.edgelist", 
                    "-order", "6", "-dim", "21", "-iters", "20"])
grarep_subprocess.wait()

# Training a logistic regression model to predict the node's class/label
subprocess.run(["python", "GraRepImplementation/logistic_regression.py"], check=True)

# Plotting the actual labels for the test set, along with prediction regions for the 
# multiclass logistic regression model
subprocess.run(["python", "GraRepImplementation/plot_embeddings.py"], check=True)


# Printing the classification report for each embedding method
with open("SpectralClusteringImplementation/spectral_clustering_labels.pkl", "rb") as labels:
    spec_labels = pickle.load(labels)
spec_y_true, spec_y_pred = spec_labels["Actual labels (test)"], spec_labels["Predicted labels"]

print(f"Classification report for Spectral Clustering:\n\n{classification_report(spec_y_true, spec_y_pred)}\n\n")

with open("Node2vecImplementation/node2vec_labels.pkl", "rb") as labels:
    n2v_labels = pickle.load(labels)
n2v_y_true, n2v_y_pred = n2v_labels["Actual labels (test)"], n2v_labels["Predicted labels"]

print(f"Classification report for Node2vec:\n\n{classification_report(n2v_y_true, n2v_y_pred)}\n\n")

with open("LINEImplementation/LINE_labels.pkl", "rb") as labels:
    line_labels = pickle.load(labels)
line_y_true, line_y_pred = line_labels["Actual labels (test)"], line_labels["Predicted labels"]

print(f"Classification report for LINE:\n\n{classification_report(line_y_true, line_y_pred)}\n\n")

with open("GraRepImplementation/grarep_labels.pkl", "rb") as labels:
    grarep_labels = pickle.load(labels)
grarep_y_true, grarep_y_pred = grarep_labels["Actual labels (test)"], grarep_labels["Predicted labels"]

print(f"Classification report for GraRep:\n\n{classification_report(grarep_y_true, grarep_y_pred)}\n\n")


print("Accuracy scores for Spectral Clustering, Node2vec, LINE, and GraRep:")
print(accuracy_score(spec_y_true, spec_y_pred),
      accuracy_score(n2v_y_true, n2v_y_pred),
      accuracy_score(line_y_true, line_y_pred),
      accuracy_score(grarep_y_true, grarep_y_pred))