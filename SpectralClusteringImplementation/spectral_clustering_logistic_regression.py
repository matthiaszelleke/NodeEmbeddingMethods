### Evaluating the node embedding's performance on a node classification 
### task using Logistic Regression

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EMBEDDING_FILENAME = "spectral_clustering_node_embeddings.csv"

embeddings = pd.read_csv("SpectralClusteringImplementation/" + EMBEDDING_FILENAME, header=None)

# Instantiate a logistic regression model
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression())
])

# Reading in the list of actual node labels
with open("sbm_actual_labels.pkl", "rb") as f:
    sbm_data = pickle.load(f)
node_labels = sbm_data["Block/Cluster"]

# Doing a train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings.values, node_labels,
                                                    train_size=0.8)

# Fitting the logistic regression model and getting the predicted labels (clusters) for each node
logreg.fit(X_train, y_train)

# Making predictions on the test set
y_predict = logreg.predict(X_test)

# Saving the actual and predicted labels for each node
with open("SpectralClusteringImplementation/spectral_clustering_labels.pkl", "wb") as f:
    pickle.dump({"Logreg model": logreg, "Training embeddings": X_train, "Test embeddings": X_test,
                  "Actual labels (training)": y_train, "Actual labels (test)": y_test, "Predicted labels": y_predict}, f)