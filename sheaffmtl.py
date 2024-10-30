import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import generate_random_adjacency_matrix, count_model_parameters, evaluate_metric

def run_sheaf_fmtl(client_train_datasets, client_test_datasets, num_rounds, alpha, eta, lambda_reg, factor, adjacency_matrix, model, loss_func, metric_func, metric_name):
    # Initialize model parameters (theta_i) for each client
    num_clients = len(client_train_datasets)
    client_models = [copy.deepcopy(model) for _ in client_train_datasets]
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(num_clients)]
    max_neighbors = max(len(n) for n in neighbors)

    # Initialize P_ij matrices
    P = {}
    for i, j in zip(*adjacency_matrix.nonzero()):
        num_params = sum(p.numel() for p in client_models[0].parameters())
        P[(i, j)] = torch.randn(int(factor*num_params), num_params)
        P[(j, i)] = torch.randn(int(factor*num_params), num_params)

    # Initialize accuracy storage
    average_test_metrics = []
    transmitted_bits_per_iteration = np.zeros(num_rounds)
    cumulative_transmitted_bits = 0

    # Training loop
    for round in range(num_rounds):
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

            with torch.no_grad():
                theta_i = torch.cat([param.view(-1) for param in client_model.parameters()])

                # Update theta_i based on P_ij and neighbors
                sum_P_terms = torch.zeros_like(theta_i)
                for j in range(num_clients):
                    if adjacency_matrix[i, j] == 1:
                        P_ij = P[(i, j)]
                        P_ji = P[(j, i)]
                        theta_j = torch.cat([param.view(-1) for param in client_models[j].parameters()])
                        sum_P_terms += P_ij.T @ (P_ij @ theta_i - P_ji @ theta_j)

                # Apply update to theta_i
                theta_i -= alpha * lambda_reg * sum_P_terms

                # Put updated theta_i back into the model
                idx = 0
                for param in client_model.parameters():
                    numel = param.numel()
                    param.data.copy_(theta_i[idx:idx+numel].reshape(param.size()))
                    idx += numel

                # Update P_ij matrices
                for j in range(num_clients):
                    if adjacency_matrix[i, j] == 1:
                        P_ij = P[(i, j)]
                        P_ji = P[(j, i)]
                        theta_j = torch.cat([param.view(-1) for param in client_models[j].parameters()])

                        # Update P_ij
                        P[(i, j)] -= eta * lambda_reg * torch.outer(P_ij @ theta_i - P_ji @ theta_j, theta_i)

        # Evaluate models
        test_metrics = [evaluate_metric(client_model, client_test_dataset, metric_func, metric_name)
                        for client_model, client_test_dataset in zip(client_models, client_test_datasets)]
        average_test_metric = sum(test_metrics) / len(test_metrics)
        average_test_metrics.append(average_test_metric)

        # Calculate communication cost
        num_params = count_model_parameters(client_models[0])
        cumulative_transmitted_bits += 2 * max_neighbors * 32 * int(factor*num_params)
        transmitted_bits_per_iteration[round] = cumulative_transmitted_bits

    return average_test_metrics, transmitted_bits_per_iteration