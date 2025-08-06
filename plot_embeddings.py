
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
import pickle
from plot_embeddings_config import args
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_model_and_labels(filename):

    with open(filename, "rb") as model_and_labels_fname:
        model_and_labels = pickle.load(model_and_labels_fname)

        model = model_and_labels["Logreg model"]

        X_train = model_and_labels["Training embeddings"]
        X_test = model_and_labels["Test embeddings"]
        y_train = model_and_labels["Actual labels (training)"]
        y_test = model_and_labels["Actual labels (test)"]

    return model, X_train, X_test, y_train, y_test

def setup_2D_logreg_model(standard_scaler, X_train, X_test, y_train):
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

    return logreg_2D, X_test_2D

def plot_contour(model, x_min, x_max, y_min, y_max):

    # Creating a meshgrid for plotting
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))

    # Predicting class labels for each point in the meshgrid
    meshgrid = np.c_[xx.ravel(), yy.ravel()]
    meshgrid_labels = model.predict(meshgrid)

    if meshgrid_labels.dtype.kind != "i":
        meshgrid_labels = to_int(meshgrid_labels)

    meshgrid_labels = meshgrid_labels.reshape(xx.shape)

    # Plotting the class decision boundaries
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, meshgrid_labels, alpha=0.8, cmap=plt.cm.RdYlBu)

def plot_embeddings(embeddings, actual_labels):
    # Plotting the test embeddings
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=actual_labels, cmap=plt.cm.RdYlBu, 
                s=30, edgecolors='k')

def plot(model_and_labels_fname, plot_fname):

    # Reading in the trained Logistic Regression model and the Training/Test data 
    logreg, X_train, X_test, y_train, y_test = load_model_and_labels(model_and_labels_fname)

    standard_scaler = logreg.named_steps["scaler"]

    logreg_2D, X_test_2D = setup_2D_logreg_model(standard_scaler, X_train, X_test, y_train)

    # Boundaries on the plot
    x_min, x_max = X_test_2D[:, 0].min() - 1, X_test_2D[:, 0].max() + 1
    y_min, y_max = X_test_2D[:, 1].min() - 1, X_test_2D[:, 1].max() + 1
    
    plot_contour(logreg_2D, x_min, x_max, y_min, y_max)
    plot_embeddings(embeddings=X_test_2D, actual_labels=y_test)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Multiclass 2-feature Logistic Regression Decision Boundaries in 2-dim space')

    plt.show()

    # Saving the plot in the proper directory
    plt.savefig(plot_fname)

if __name__ == "__main__":
    plot(args.model_and_labels_fname, args.plot_fname)