import torch
import torch.nn as nn
import networkx as nx
from sklearn.datasets import fetch_openml
import scipy
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

class LinearRegression(nn.Module):
    def __init__(self, input_size, bias=True):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

def count_model_parameters(model):
    return sum(param.numel() for param in model.parameters())

def maximum_neighbours(adjacency_matrix):
    max_neighbors = 0
    client_with_max_neighbors = None

    num_nodes = adjacency_matrix.shape[0]
    for node in range(num_nodes):
        num_neighbors = np.sum(adjacency_matrix[node])
        if num_neighbors > max_neighbors:
            max_neighbors = num_neighbors
            client_with_max_neighbors = node
    return int(max_neighbors)

def cross_entropy_loss_with_l2(model, data, targets, l2_strength):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(data), targets)
    l2_reg = sum(param.pow(2).sum() for param in model.parameters())
    return loss + l2_strength * l2_reg

def mse_loss_with_l2(model, data, targets, l2_strength):
    criterion = nn.MSELoss()
    predictions = model(data)
    loss = criterion(predictions, targets.view(-1, 1))
    l2_reg = sum(param.pow(2).sum() for param in model.parameters())
    return loss + l2_strength * l2_reg

def evaluate_metric(model, dataset, metric_func, metric_name):
    model.eval()
    total_metric = 0
    total_count = 0
    with torch.no_grad():
        for data, targets in DataLoader(dataset, batch_size=32):
            predictions = model(data)
            if metric_name == 'Accuracy':
                metric = metric_func(targets, predictions.argmax(dim=1))
            elif metric_name == 'MSE':
                metric = metric_func(predictions, targets.view(-1, 1))
            total_metric += metric.item() * data.size(0)
            total_count += data.size(0)
    return total_metric / total_count

def generate_random_adjacency_matrix(n, edge_probability=0.2, seed=42):
    while True:
        g = nx.generators.random_graphs.binomial_graph(n, edge_probability, seed=seed)
        if nx.is_connected(g):
            return nx.adjacency_matrix(g).toarray()

def read_school_data(seed=None, bias=False, standardize=True):
    """ Read school dataset from .mat file."""
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    mat = scipy.io.loadmat('./Datasets/school.mat')  # Update the path accordingly
    raw_x, raw_y = mat['X'][0], mat['Y'][0]  # Adjust these based on actual keys in your .mat file

    num_clients = len(raw_x)

    client_train_datasets = []
    client_test_datasets = []

    for i in range(num_clients):
        features, label = raw_x[i], raw_y[i].flatten()

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        x_train, x_test, y_train, y_test = train_test_split(
            features_tensor, label_tensor, test_size=0.25, random_state=seed)

        if standardize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            # For y (scores), use min/max normalization
            min_y, max_y = 1, 70    # Hardcode stats from dataset.
            y_train = (y_train - min_y) / (max_y - min_y)
            y_test = (y_test - min_y) / (max_y - min_y)
        if bias:
            x_train = np.c_[x_train, np.ones(len(x_train))]
            x_test = np.c_[x_test, np.ones(len(x_test))]

        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        client_train_datasets.append(train_dataset)
        client_test_datasets.append(test_dataset)

    return client_train_datasets, client_test_datasets

def read_vehicle_data(seed=None, bias=False, density=1.0, standardize=False, downsample_rate=0.2):
    """Read Vehicle dataset from .mat file.    """
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    mat = scipy.io.loadmat('./Datasets/vehicle.mat')
    raw_x, raw_y = mat['X'], mat['Y']  # y in {-1, 1}

    # Additional code for downsampling
    num_clients = len(raw_x)
    downsample_clients = np.random.choice(num_clients, size=int(num_clients/2), replace=False)

    client_train_datasets = []
    client_test_datasets = []

    input_size = np.shape(raw_x[0][0])[1]
    num_classes = len(np.unique(raw_y[0][0].flatten()))

    for i in range(num_clients):
        features, label = raw_x[i][0], raw_y[i][0].flatten()
        label[label == -1] = 0
        

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Downsampling if the client is selected
        if i in downsample_clients:
            downsample_idx = np.random.choice(len(features_tensor), size=int(len(features_tensor) * downsample_rate), replace=False)
            features_tensor = features_tensor[downsample_idx]
            label_tensor = label_tensor[downsample_idx]

        x_train, x_test, y_train, y_test = train_test_split(
            features_tensor, label_tensor, test_size=0.25, random_state=seed)

        if standardize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        if bias:
            x_train = np.c_[x_train, np.ones(len(x_train))]
            x_test = np.c_[x_test, np.ones(len(x_test))]

        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        client_train_datasets.append(train_dataset)
        client_test_datasets.append(test_dataset)

    return client_train_datasets, client_test_datasets, input_size, num_classes 

def read_har_data(downsample_rate=0.2, downsample_clients=None):
    client_train_datasets = []
    client_test_datasets = []

    X_split = []
    y_split = []

    # Load dataset
    X, y = fetch_openml('har', version=1, return_X_y=True)

    # Preprocess labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Convert to PyTorch tensors
    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.long)

    task_index = np.load('./Datasets/task_index.npy')
    num_clients = len(np.unique(task_index))

    input_size = X.shape[1]
    num_classes = len(np.unique(y_encoded))

    # Decide which clients to downsample. If not provided, randomly select half of the clients.
    if downsample_clients is None:
        downsample_clients = np.random.choice(num_clients, size=int(num_clients/2), replace=False)

    for i in range(num_clients):
        index = np.where(task_index == i+1)
        min_idx = index[0][0]
        max_idx = index[0][-1] + 1
        X_client = X[min_idx:max_idx]
        y_client = y[min_idx:max_idx]

        # Downsample if this client is in the downsample list
        if i in downsample_clients:
            downsample_idx = np.random.choice(len(X_client), size=int(len(X_client) * downsample_rate), replace=False)
            X_client = X_client[downsample_idx]
            y_client = y_client[downsample_idx]

        X_split.append(X_client)
        y_split.append(y_client)

    for X_client, y_client in zip(X_split, y_split):
        X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.25)
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        client_train_datasets.append(train_dataset)
        client_test_datasets.append(test_dataset)

    return client_train_datasets, client_test_datasets, input_size, num_classes

#client_train_datasets, client_test_datasets, input_size, num_classes = read_har_data()

def read_gleam_data(seed=None, bias=False, density=1.0, standardize=False, downsample_rate=0.2):
  """Read gleam dataset from .mat file.    """
  x_trains, y_trains, x_tests, y_tests = [], [], [], []
  mat = scipy.io.loadmat('./Datasets/gleam.mat')
  raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y in {-1, 1}


  # Additional code for downsampling
  num_clients = len(raw_x)
  downsample_clients = np.random.choice(num_clients, size=int(num_clients/2), replace=False)


  client_train_datasets = []
  client_test_datasets = []

  input_size = np.shape(raw_x[0])[1]
  num_classes = len(np.unique(raw_y[0].flatten()))

  for i in range(len(raw_x)):
    features, label = raw_x[i], raw_y[i].flatten()

    label[label == -1] = 0

    features_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.long)

    # Downsampling if the client is selected
    if i in downsample_clients:
      downsample_idx = np.random.choice(len(features_tensor), size=int(len(features_tensor) * downsample_rate), replace=False)
      features_tensor = features_tensor[downsample_idx]
      label_tensor = label_tensor[downsample_idx]

    x_train, x_test, y_train, y_test = train_test_split(
        features_tensor, label_tensor, test_size=0.25, random_state=seed)


    if density != 1:
      # Randomness should be set for different workers (no explicit seed)
      num_train_examples = int(density * len(x_train))
      train_mask = np.random.permutation(len(x_train))[:num_train_examples]
      x_train = x_train[train_mask]
      y_train = y_train[train_mask]
    if standardize:
      # Preprocessing using mean/std from training examples, within each silo
      scaler = StandardScaler().fit(x_train)
      x_train = scaler.transform(x_train)
      x_test = scaler.transform(x_test)
    if bias:
      # Append a column of ones to implicitly include bias
      x_train = np.c_[x_train, np.ones(len(x_train))]
      x_test = np.c_[x_test, np.ones(len(x_test))]


    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    client_train_datasets.append(train_dataset)
    client_test_datasets.append(test_dataset)

  return client_train_datasets, client_test_datasets, input_size, num_classes


#---------------------------New Functions-------------------------------

def TwoSectionH(G): # Generate the 2-section of the proposed hypergraph, i.e., H2
    # VertexH = G.nodes()
    Hyperedge = [ tuple(sorted([node] + [n for n in G.neighbors(node)]))
                                    for node in G.nodes() ] # construct a hypergraph each of whose hyperedge consists of a node and its neighbours
    Hyperedge = list(set(Hyperedge)) # remove any repeated hyperedges
    H2Edges = [[tuple(sorted(e)) for e in nx.complete_graph(he).edges()] for he in Hyperedge] # a list of list of edges of H2, pylint: disable = undefined-variable
    temp = [] # construct the edge set of the H2
    for e in H2Edges:
        temp.extend(e) # remove the inner list delimiter "[]"
    H2Edges = list(set(temp)) # remove any repeated edges

    H2 = nx.Graph() # pylint: disable = undefined-variable
    H2.add_nodes_from(G.nodes())
    H2.add_edges_from(H2Edges)
    vertex_color_map = nx.greedy_color(H2, strategy = 'saturation_largest_first') # vertex coloring H2, pylint: disable = undefined-variable
    # Chi = max(vertex_color_map.values()) + 1 # chromatic number of the present coloring scheme

    return H2, vertex_color_map


def seq_scheduling(G):
    # A sequential list (slot's) of star toplogy-based schedule in a form of dicts
    star_schedule_list = []
    # key-value pair herein is node (n_b, n_c), where n_b is the #times for which a node transmits as a star center (BC),
    # and n_c is the #times for which a node transmits as an outer node
    Tx_times = {node: [0, 0] for node in G.nodes()}

    while G:
        _, from_node_to_color_id = TwoSectionH(G)
        color_degree = {c: sum(len(G[node]) for node, color in from_node_to_color_id.items() if color == c)
                        for c in from_node_to_color_id.values()}
        chosen_color = list(color_degree.values()).index(max(color_degree.values()))  # find arg_max(degree(color_list))

        # A dict including (star center: associated nodes) pairs that transmits or recieves in parallel at the current slot
        star_schedule_dict = {node: G[node] for node, color in from_node_to_color_id.items() if color == chosen_color}
        # Append the scheule in the current slot to the sequential schedule list
        star_schedule_list.append(star_schedule_dict)
        # Update n_b for the star center
        for node in star_schedule_dict.keys():
            Tx_times[node][0] += 1
        # Update n_c for the neighbors of the star centers
        for neighbors in star_schedule_dict.values():
            for node in neighbors:
                Tx_times[node][1] += 1

        # Update the graph
        # Remove the scheduled Rxs, i.e., the star centers
        G.remove_nodes_from(star_schedule_dict.keys())
        # Remove any standalone nodes
        current_node_list = list(G.nodes())
        for node in current_node_list:
            if not (G[node]):
                G.remove_node(node)

    return star_schedule_list, Tx_times

def dig_comp_level(G, CG, N, Chi, barP, N0 = 10 ** (-169/10) * 1e-3, b = 64, d = 7850):
    K = CG.shape[0]
    m_array = [int( N / Chi * np.log2(1 + barP * Chi / N0 * min(CG[i,[j for j in G[i]]]))) for i in range(K)]
    m_array = np.maximum(np.minimum(np.array(m_array)/b, d), 1)

    return m_array

