import matplotlib.pyplot as plt
import pandas as pd


FILENAME = "classification_accuracy_all_density.csv"
DENSITY_PLOT_FNAME = "classification_accuracy_all_density.png"

# Mapping embedding methods to ints, used by plt.scatter() to colour by method
method_to_int = {"Spectral Clustering": int(0),
                 "Node2vec": int(1),
                 "LINE": int(2),
                 "GraRep": int(3)}

# CSV file of NMI Scores for different network densities and different methods
# (The file created by run_all_density.py)
accuracies_density = pd.read_csv(FILENAME, header=0)

methods_as_ints = [method_to_int[method] for method in accuracies_density["Method"]]

# Plotting NMI score vs. network density for each method
fig, ax = plt.subplots()
scatterplot = plt.scatter(x=accuracies_density["Density"], y=accuracies_density["Accuracy"], c=methods_as_ints)

# Minimizing the size of the scatterplot (to make room for the legend)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

# Adding the legend
handles, labels = scatterplot.legend_elements()
plt.legend(handles, ["Spectral Clustering", "Node2vec", "LINE", "GraRep"], title="Node Embedding Method",
           bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylim(-0.03, 1.03)

# Adding axis labe;s
plt.xlabel("Network Density")
plt.ylabel("Classification Accuracy")

plt.title("Classification Accuracy vs. Network Density")

plt.savefig(DENSITY_PLOT_FNAME)

print(f"A plot of Classification Accuracies for each network density, coloured by embedding method, can be found in {DENSITY_PLOT_FNAME}")