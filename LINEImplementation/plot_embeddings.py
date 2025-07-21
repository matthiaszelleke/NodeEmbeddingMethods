import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reading in the trained logistic regression model
with open("LINEImplementation/LINE_labels.pkl", "rb") as labels:
    line_labels = pickle.load(labels)

line_logreg = line_labels["Logreg model"]
X_test = line_labels["Test embeddings"]
y_test = line_labels["Actual labels (test)"]

standard_scaler = line_logreg.named_steps["scaler"]
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
meshgrid_labels = line_logreg.predict(full_dim_emb)
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
output_path = os.path.join(current_dir, "line_pca.png")

plt.savefig(output_path)