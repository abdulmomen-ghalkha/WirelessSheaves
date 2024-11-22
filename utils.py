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
import cvxpy as cp
from cvxpy.atoms.lambda_sum_smallest import lambda_sum_smallest

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
'''
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
'''
def read_vehicle_data(seed=None, bias=False, density=1.0, standardize=False, downsample_rate=0.2, feature_groups=4, features_to_drop=4):
    """Read Vehicle dataset from .mat file with partial feature dropping for groups."""
    import random
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    mat = scipy.io.loadmat('./Datasets/vehicle.mat')
    raw_x, raw_y = mat['X'], mat['Y']  # y in {-1, 1}

    # Additional code for downsampling
    num_clients = len(raw_x)
    downsample_clients = np.random.choice(num_clients, size=int(num_clients / 2), replace=False)

    client_train_datasets = []
    client_test_datasets = []

    input_size = np.shape(raw_x[0][0])[1]
    num_classes = len(np.unique(raw_y[0][0].flatten()))



    # Divide clients into groups
    clients_per_group = (int)(np.ceil(num_clients / feature_groups))
    group_feature_indices = []

    # Assign feature subsets to each group
    total_features = np.arange(input_size)
    for _ in range(feature_groups):
        # Drop `features_to_drop` features from the total set
        if features_to_drop > 0:
            drop_features = np.random.choice(total_features, size=np.random.choice(features_to_drop), replace=False)
            keep_features = np.setdiff1d(total_features, drop_features)
        else:
            keep_features = total_features

        group_feature_indices.append(keep_features)
    
    modified_input_sizes = []

    for i in range(num_clients):
        features, label = raw_x[i][0], raw_y[i][0].flatten()
        label[label == -1] = 0

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Assign features based on client group
        group_id = i // clients_per_group  # Determine the group for the client
        selected_features = group_feature_indices[group_id]  # Get group's feature subset
        modified_input_sizes.append(len(selected_features))
        features_tensor = features_tensor[:, selected_features]

        # Downsampling if the client is selected
        if i in downsample_clients:
            downsample_idx = np.random.choice(
                len(features_tensor), size=int(len(features_tensor) * downsample_rate), replace=False
            )
            features_tensor = features_tensor[downsample_idx]
            label_tensor = label_tensor[downsample_idx]

        x_train, x_test, y_train, y_test = train_test_split(
            features_tensor, label_tensor, test_size=0.25, random_state=seed
        )

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

    return client_train_datasets, client_test_datasets, modified_input_sizes, num_classes



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