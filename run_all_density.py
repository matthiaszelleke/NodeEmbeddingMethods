import csv
import subprocess
import pandas as pd
import pickle

FILENAME = "nmi_scores_all_density.csv"
DENSITY_PLOT_FNAME = "nmi_scores_all_density.png"
within_cluster_probs = [0.5, 0.6, 0.7, 0.8, 0.9] # the within-cluster connection probability (to be used in SBM)

with open(FILENAME, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Density", "NMI Score", "Method"])

    for within_cluster_prob in within_cluster_probs:
        # Making each of the 3 clusters have the same within-cluster connection probability
        result = subprocess.run(["python", "run_all.py", "-p1", str(within_cluster_prob),
                                                         "-p2", str(within_cluster_prob),
                                                         "-p3", str(within_cluster_prob)],
                                          capture_output=True, text=True, check=True)
        
        last_line = result.stdout.strip().split("\n")[-1]
        nmi_scores_fname = last_line.strip().split(" ")[-1]

        # Reading in the NMI scores for "this" SBM network
        nmi_scores = pd.read_csv(nmi_scores_fname, header=0)

        # Reading in the network
        with open("sbm_actual_labels.pkl", "rb") as f:
            sbm_data = pickle.load(f)
        G = sbm_data["Graph"]

        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()

        # The denominator is the total number of possible edges (n choose 2)
        density = n_edges / (n_nodes*(n_nodes-1)/2)

        # Writing the info to the final csv file
        for _, method in nmi_scores.iterrows():
            writer.writerow([density, method["NMI Score"], method["Method"]])