import csv
import subprocess
import pickle

FILENAME = "classification_accuracy_all_density.csv"

METHODS = ["Spectral Clustering", "Node2vec", "LINE", "GraRep"]
NUM_METHODS = 4

within_class_probs = [0.5, 0.6, 0.7, 0.8, 0.9] # the within-class connection probability (to be used in SBM)

with open(FILENAME, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Density", "Method", "Accuracy"])

    for within_class_prob in within_class_probs:
        # Making each of the 3 classes have the same within-class connection probability
        result = subprocess.run(["python", "run_all_classification.py", "-p1", str(within_class_prob),
                                                         "-p2", str(within_class_prob),
                                                         "-p3", str(within_class_prob)],
                                          capture_output=True, text=True, check=True)
        
        # Extracting the printed accuracies from run_all.py
        last_line = result.stdout.strip().split("\n")[-1]
        accuracies = [accuracy for accuracy in last_line.strip().split(" ")]

        # Reading in the network
        with open("sbm_actual_labels.pkl", "rb") as f:
            sbm_data = pickle.load(f)
        G = sbm_data["Graph"]

        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()

        # The denominator is the total number of possible edges (n choose 2)
        density = n_edges / (n_nodes*(n_nodes-1)/2)

        # Writing the info to the final csv file
        for method, accuracy in zip(METHODS, accuracies):
            writer.writerow([density, method, accuracy])