### Evaluating the node embedding's performance on a node classification 
### task using Logistic Regression

import os
import torch
from config import args
from utils.line import LINE
from train import args
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

MODEL_FILENAME = "LINE_model.pt"
NUM_CLUSTERS = 3
NUM_NODES = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

line_ord1 = LINE(NUM_NODES, embed_dim=args.dimension).to(device)
line_ord2 = LINE(NUM_NODES, embed_dim=args.dimension).to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path_ord1 = os.path.join(script_dir, f"{MODEL_FILENAME.split('.')[0]}_ord1.pt")
model_path_ord2 = os.path.join(script_dir, f"{MODEL_FILENAME.split('.')[0]}_ord2.pt")
line_ord1.load_state_dict(torch.load(model_path_ord1))
line_ord2.load_state_dict(torch.load(model_path_ord2))

line_ord1.eval()
line_ord2.eval()

embeddings_ord1 = line_ord1.node_embeddings.weight.data
embeddings_ord2 = line_ord2.contextnode_embeddings.weight.data

final_emb = torch.cat([embeddings_ord1,
                       embeddings_ord2], dim=1)

final_emb_np = final_emb.cpu().numpy()

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
X_train, X_test, y_train, y_test = train_test_split(final_emb_np, node_labels,
                                                    train_size=0.8)

# Fitting the logistic regression model and getting the predicted labels (clusters) for each node
logreg.fit(X_train, y_train)

# Making predictions on the test set
y_predict = logreg.predict(X_test)

# Saving the actual and predicted labels for each node
with open("LINEImplementation/LINE_labels.pkl", "wb") as f:
    pickle.dump({"Logreg model": logreg, "Training embeddings": X_train, "Test embeddings": X_test,
                  "Actual labels (training)": y_train, "Actual labels (test)": y_test, "Predicted labels": y_predict}, f)