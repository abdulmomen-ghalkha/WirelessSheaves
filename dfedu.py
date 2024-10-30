import copy
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import count_model_parameters, evaluate_metric, cross_entropy_loss_with_l2, mse_loss_with_l2, MultinomialLogisticRegression, LinearRegression

def run_dfedu(client_train_datasets, client_test_datasets, num_rounds, local_iterations, local_lr, neighbors, lambda_reg, model, loss_func, metric_func, metric_name):

    global_lr = local_lr*local_iterations
    client_models = [copy.deepcopy(model) for _ in client_train_datasets]
    average_test_metrics = []
    transmitted_bits_per_iteration = np.zeros(num_rounds)
    
    # Calculate the number of bits for a single parameter transmission (assuming 32 bits per parameter)
    bits_per_param = 32

    # Count the parameters in one of the client models (assuming all client models have the same number of parameters)
    num_params = count_model_parameters(client_models[0]) 
    cumulative_transmitted_bits = 0
    max_neighbors = max(len(n) for n in neighbors)

    for round in range(num_rounds):
        # Local updates
        for client_id, client_data in enumerate(client_train_datasets):
            optimizer = torch.optim.SGD(client_models[client_id].parameters(), lr=local_lr)
            for epoch in range(local_iterations):
                for data, target in DataLoader(client_data, batch_size=32):
                    optimizer.zero_grad()
                    loss = loss_func(client_models[client_id], data, target, l2_strength=0.01)
                    loss.backward()
                    optimizer.step()

        # Global updates
        for client_id, client_model in enumerate(client_models):
            client_updates = {param: torch.zeros_like(param.data) for param in client_model.parameters()}

            for neighbor_id in neighbors[client_id]:
                weight = 1  # Define the weight between clients
                for (param, neighbor_param), update in zip(zip(client_model.parameters(), client_models[neighbor_id].parameters()), client_updates.values()):
                    update += global_lr * lambda_reg* weight * (param.data - neighbor_param.data)

            for param, update in zip(client_model.parameters(), client_updates.values()):
                param.data -= update

        # Calculate average test metric
        test_metrics = [evaluate_metric(client_model, client_test_dataset, metric_func, metric_name)
                        for client_model, client_test_dataset in zip(client_models, client_test_datasets)]
        average_test_metric = sum(test_metrics) / len(test_metrics)
        average_test_metrics.append(average_test_metric)
        
        # Calculate communication cost
        cumulative_transmitted_bits += 2 * max_neighbors * bits_per_param * num_params
        transmitted_bits_per_iteration[round] = cumulative_transmitted_bits

    return average_test_metrics, transmitted_bits_per_iteration