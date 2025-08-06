### Evaluating the node embedding's performance on a node classification 
### task using Logistic Regression

import csv
from logistic_regression_config import args
import networkx as nx
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import to_int


def save_model_and_labels(model, X_train, X_test, y_train, y_test, model_filename):
   
   with open(model_filename, "wb") as model_fname:
    pickle.dump({"Logreg model": model, 
                 "Training embeddings": X_train, 
                 "Test embeddings": X_test,
                 "Actual labels (training)": y_train, 
                 "Actual labels (test)": y_test}, model_fname)

def logreg_fit_predict(embeddings, node_labels, model_filename):

    # Instantiate a logistic regression model
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression())
    ])

    # Doing a train/test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, node_labels,
                                                        train_size=0.8, random_state=100)

    # Fitting the logistic regression model and getting the predicted labels (clusters) for each node
    logreg.fit(X_train, y_train)

    # Making predictions on the test set
    y_predict = logreg.predict(X_test)

    save_model_and_labels(logreg, X_train, X_test, y_train, y_test, model_filename)

    return y_predict, y_test

def get_prediction_metrics(predicted_labels, actual_labels, save_metrics=False, metrics_filename=None):

    accuracy_score_ = accuracy_score(actual_labels, predicted_labels)
    f1_score_ = f1_score(actual_labels, predicted_labels, average='macro')
    precision_score_ = precision_score(actual_labels, predicted_labels, average='macro')
    recall_score_ = recall_score(actual_labels, predicted_labels, average='macro')

    print(f"Accuracy: {accuracy_score_:.2f}")
    print(f"(Macro) F1-Score: {f1_score_:.2f}")
    print(f"(Macro) Precision: {precision_score_:.2f}")
    print(f"(Macro) Recall: {recall_score_:.2f}")

    if save_metrics: # if user chooses to save the metrics to a csv file
        if not metrics_filename:
            print("Error: Must specify a filename for the metrics if you wish to save them")
            return
        
        metrics_and_scores = [
            {"Metric": "Accuracy", "Score": accuracy_score_},
            {"Metric": "(Macro) F1-Score", "Score": f1_score_},
            {"Metric": "(Macro) Precision", "Score": precision_score_},
            {"Metric": "(Macro) Recall", "Score": recall_score_}
        ]

        with open(metrics_filename, "w", newline="") as csvfile:
            field_names = ["Metric", "Score"]
            dict_csv_writer = csv.DictWriter(csvfile, fieldnames=field_names)
            dict_csv_writer.writeheader()
            dict_csv_writer.writerows(metrics_and_scores)

if __name__ == "__main__":
    embeddings = pd.read_csv(args.embeddings_filename, header=None)

    network = nx.read_edgelist(args.network)
    node_labels = [node_label for node, node_label in network.nodes.data("label")]

    if node_labels.dtype.kind != "i":
        node_labels = to_int(node_labels)

    y_predict, y_test = logreg_fit_predict(embeddings.values, node_labels, 
                                           args.model_and_labels_fname)
    get_prediction_metrics(y_predict, y_test, args.metrics_filename)