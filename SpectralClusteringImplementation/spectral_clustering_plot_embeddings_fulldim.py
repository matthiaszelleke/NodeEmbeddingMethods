
""" Visualizing the performance of the Logistic Regression model by plotting
    the test embeddings coloured by their actual class along with 2D contours showing
    each point in the 2D plotting space coloured by its predicted class.
    
    Note: The predicted classes for each point in the 2D plotting space are 
          obtained using the same 3-dimensional Logistic Regression model used
          to predict the node classes. The points in the PCA-reduced 2D space
          are reprojected to the 3D space using PCA's inverse transform, and
          then their predicted classes are obtained from the 3-dimensional 
          Logistic Regression model.

          Thus, the plot produced shows the "proper/accurate" classes for each grid 
          point, but the resulting class boundaries may look weird due to information
          loss from projecting them from 3 to 2 dimensions.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA

# Reading in the trained logistic regression model
with open("SpectralClusteringImplementation/spectral_clustering_labels.pkl", "rb") as labels:
    spec_labels = pickle.load(labels)

spec_logreg = spec_labels["Logreg model"]
X_test = spec_labels["Test embeddings"]
y_test = spec_labels["Actual labels (test)"]

standard_scaler = spec_logreg.named_steps["scaler"]
embeddings = standard_scaler.transform(X_test)

pca_model = PCA(n_components=2)
pca_embeddings = pca_model.fit_transform(embeddings)

# Create a meshgrid for plotting
x_min, x_max = pca_embeddings[:, 0].min() - 1, pca_embeddings[:, 0].max() + 1
y_min, y_max = pca_embeddings[:, 1].min() - 1, pca_embeddings[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

# Re-project pca embeddings into original higher dimension space (for model prediction)
full_dim_emb = pca_model.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

# Predict class labels for each point in the meshgrid
meshgrid_labels = spec_logreg.predict(full_dim_emb)
meshgrid_labels = meshgrid_labels.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, meshgrid_labels, alpha=0.8, cmap=plt.cm.RdYlBu)

# Plot the data points
plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=y_test, cmap=plt.cm.RdYlBu, s=30, edgecolors='k')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Multiclass Logistic Regression Decision Boundaries in 2-dim space')

# Saving the plot in the proper directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "spectral_clustering_pca_fulldim.png")

plt.savefig(output_path)