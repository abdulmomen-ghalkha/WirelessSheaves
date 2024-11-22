import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import generate_random_adjacency_matrix, count_model_parameters, evaluate_metric, generate_random_adjacency_matrix


def run_local_training(client_train_datasets, client_test_datasets, num_rounds, alpha, eta, lambda_reg, factor,
                       adjacency_matrix, models, loss_func, metric_func, metric_name, beta, Ct, K):
    num_clients = len(client_train_datasets)
    client_models = [copy.deepcopy(models[i]) for i in range(len(client_train_datasets))]
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(num_clients)]
    max_neighbors = max(len(n) for n in neighbors)
    
    for i in range(num_clients):
        num_params = count_model_parameters(client_models[i])
        print(f"Models: {i}, Num of params: {num_params}")


    # Initialize accuracy storage
    average_test_metrics = []
    transmitted_bits_per_iteration = np.zeros(num_rounds)
    cumulative_transmitted_bits = 0
    cumulative_used_resource_blocks = 0
    resource_blocks = np.zeros(num_rounds)
    

    # Training loop
    for round in range(num_rounds):
        print()
        for i in range(num_clients):
            client_model = client_models[i]
            optimizer = torch.optim.SGD(client_model.parameters(), lr=alpha)
            train_loader = DataLoader(client_train_datasets[i], batch_size=32, shuffle=True)

            # Training the model
            client_model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = client_model(X_batch)
                loss = loss_func(client_model, X_batch, y_batch, l2_strength=0.01)
                loss.backward()
                optimizer.step()

            
        # Evaluate models
        test_metrics = [evaluate_metric(client_model, client_test_dataset, metric_func, metric_name)
                        for client_model, client_test_dataset in zip(client_models, client_test_datasets)]
        average_test_metric = sum(test_metrics) / len(test_metrics)
        average_test_metrics.append(average_test_metric)

        # Calculate communication cost
        num_params = sum([count_model_parameters(client_models[i]) for i in range(num_clients)] ) // num_clients 

        cumulative_transmitted_bits += 0
        transmitted_bits_per_iteration[round] = cumulative_transmitted_bits
        print(f"round: {round}, Average_metric: {average_test_metric}")

    return average_test_metrics, transmitted_bits_per_iteration

