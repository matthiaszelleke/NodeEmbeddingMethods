from sklearn.preprocessing import LabelEncoder

### Helper functions used in various files 

class InvalidNetworkStructure(Exception):
    """Custom exception to handle cases when the network's
       structure is different from what is expected"""
    pass

class UnequalAttributeCounts(Exception):
    """Custom exception to handle the case when the nodes
       in a user's network don't all have the same number of
       attributes"""
    pass

def get_node_clusters(network):
    node_clusters = []

    for node, label in network.nodes.data("label"):
        if label:
            node_clusters.append(label)
        else:  
            raise InvalidNetworkStructure("Every node must have an attribute named 'label' in order to determine the node's cluster/class")
        
    return node_clusters
    
def get_num_clusters(node_clusters):
    return len(set(node_clusters))

def to_int(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels