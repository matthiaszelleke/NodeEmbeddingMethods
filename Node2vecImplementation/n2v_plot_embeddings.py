
""" Visualizing the performance of the Logistic Regression model by plotting
    the test embeddings and their actual class along with 2D contours showing
    the predicted class for each point in the 2D plotting space.
    
    Note: Unlike the other plotting file in this folder, the predicted class 
          for each point in the 2D plotting space is obtained by using a separate
          2-dimensional Logistic Regression model (taking 2 input features). This
          model is fitted to the training embeddings and then used to predict the
          class for each point in the 2D plotting space.

          Thus, the plot produced doesn't show what the "proper/actual" predicted 
          class would be for every grid point, but it sacrifices this to gain
          interpretability and visual cleanliness, as the plot shows clear and
          sensible 2D class boundaries.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Reading in the trained Logistic Regression model and the Training/Test data 
with open("Node2vecImplementation/n2v_labels.pkl", "rb") as labels:
    n2v_labels = pickle.load(labels)

n2v_logreg = n2v_labels["Logreg model"]

X_train = n2v_labels["Training embeddings"]
y_train = n2v_labels["Actual labels (training)"]
X_test = n2v_labels["Test embeddings"]
y_test = n2v_labels["Actual labels (test)"]

standard_scaler = n2v_logreg.named_steps["scaler"]

# The 2D Logistic Regression model to predict the class for each grid point
two_dim_logreg = Pipeline([
    ("scaler", standard_scaler),
    ("pca_2D", PCA(n_components=2)),
    ("logreg_2D", LogisticRegression())
])
standard_scaler_2D = two_dim_logreg.named_steps["scaler"]
pca_2D = two_dim_logreg.named_steps["pca_2D"]
logreg_2D = two_dim_logreg.named_steps["logreg_2D"]

# Fitting the 2D Logreg model to the training embeddings (to prevent overfitting)
X_train = standard_scaler_2D.transform(X_train)
X_train_2D = pca_2D.fit_transform(X_train)
logreg_2D.fit(X_train_2D, y_train)

# Obtaining the 2D principal components of the test embeddings
X_test = standard_scaler_2D.transform(X_test)
X_test_2D = pca_2D.transform(X_test)

# Creating a meshgrid for plotting
x_min, x_max = X_test_2D[:, 0].min() - 1, X_test_2D[:, 0].max() + 1
y_min, y_max = X_test_2D[:, 1].min() - 1, X_test_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

# Predicting class labels for each point in the meshgrid
meshgrid = np.c_[xx.ravel(), yy.ravel()]
meshgrid_labels = logreg_2D.predict(meshgrid)
meshgrid_labels = meshgrid_labels.reshape(xx.shape)

# Plotting the class decision boundaries
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, meshgrid_labels, alpha=0.8, cmap=plt.cm.RdYlBu)

# Plotting the test embeddings
plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=y_test, cmap=plt.cm.RdYlBu, s=30, edgecolors='k')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Multiclass 2-feature Logistic Regression Decision Boundaries in 2-dim space')

# Saving the plot in the proper directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "n2v_pca.png")

plt.savefig(output_path)