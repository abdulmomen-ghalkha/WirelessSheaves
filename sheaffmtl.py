import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import generate_random_adjacency_matrix, count_model_parameters, evaluate_metric, TwoSectionH, \
    seq_scheduling, generate_random_adjacency_matrix, dig_comp_level


def run_dig_sheaf_fmtl(client_train_datasets, client_test_datasets, num_rounds, alpha, eta, lambda_reg, factor,
                       adjacency_matrix, model, loss_func, metric_func, metric_name, G, K, CH_gen, PL, Chi, P_bar, N,
                       d):
    # Initialize model parameters (theta_i) for each client
    num_clients = len(client_train_datasets)
    client_models = [copy.deepcopy(model) for _ in client_train_datasets]
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(num_clients)]
    max_neighbors = max(len(n) for n in neighbors)

    # Initialize P_ij matrices
    P = {}
    for i, j in zip(*adjacency_matrix.nonzero()):
        num_params = sum(p.numel() for p in client_models[0].parameters())
        P[(i, j)] = torch.randn(int(factor * num_params), num_params)
        P[(j, i)] = torch.randn(int(factor * num_params), num_params)

    # Initialize accuracy storage
    average_test_metrics = []
    transmitted_bits_per_iteration = np.zeros(num_rounds)
    cumulative_transmitted_bits = 0
    cumulative_used_resource_blocks = 0
    resource_blocks = np.zeros(num_rounds)

    # Training loop
    for round in range(num_rounds):

        CH_gen_current = CH_gen[round]
        CH = np.ones((K, K), dtype=complex)  # Channel coefficients
        for i in range(K):
            for j in G[i]:
                if j < i:
                    CH[i, j] = CH_gen_current[i, j]
        for i in range(K):
            for j in G[i]:
                if j > i:
                    CH[i, j] = np.conjugate(CH[j, i])
        CH = np.sqrt(PL) * CH

        CG = np.abs(CH) ** 2
        # A list (device_i's) of #rows for the RLC matrix supported by the channels of each device to its neighbors
        m_array = dig_comp_level(G, CG, N, Chi, P_bar, 10 ** (-169 / 10) * 1e-3, 32, d * factor)

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
                    param.data.copy_(theta_i[idx:idx + numel].reshape(param.size()))
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
        # num_params = count_model_parameters(client_models[0])
        cumulative_transmitted_bits += 2 * max_neighbors * 32 * int(factor * d)
        transmitted_bits_per_iteration[round] = cumulative_transmitted_bits
        cumulative_used_resource_blocks += int(np.ceil(factor * d / np.min(m_array)))
        print(cumulative_used_resource_blocks)
        resource_blocks[round] = cumulative_used_resource_blocks

    return average_test_metrics, transmitted_bits_per_iteration, resource_blocks


def run_ana_sheaf_fmtl(client_train_datasets, client_test_datasets, num_rounds, alpha, eta, lambda_reg, factor,
                       adjacency_matrix, model, loss_func, metric_func, metric_name, G, K, CH_gen, PL, schedule_list,
                       Tx_times, P_bar, N, d):
    # Initialize model parameters (theta_i) for each client
    num_clients = len(client_train_datasets)
    client_models = [copy.deepcopy(model) for _ in client_train_datasets]
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(num_clients)]
    max_neighbors = max(len(n) for n in neighbors)

    # Initialize P_ij matrices
    P = {}
    for i, j in zip(*adjacency_matrix.nonzero()):
        num_params = sum(p.numel() for p in client_models[0].parameters())
        P[(i, j)] = torch.randn(int(factor * num_params), num_params)
        P[(j, i)] = torch.randn(int(factor * num_params), num_params)

    # Initialize accuracy storage
    average_test_metrics = []
    transmitted_bits_per_iteration = np.zeros(num_rounds)
    cumulative_transmitted_bits = 0
    cumulative_used_resource_blocks = 0
    resource_blocks = np.zeros(num_rounds)

    # Training loop
    for round in range(num_rounds):

        CH_gen_current = CH_gen[round]
        CH = np.ones((K, K), dtype=complex)  # Channel coefficients
        for i in range(K):
            for j in G[i]:
                if j < i:
                    CH[i, j] = CH_gen_current[i, j]
        for i in range(K):
            for j in G[i]:
                if j > i:
                    CH[i, j] = np.conjugate(CH[j, i])
        CH = np.sqrt(PL) * CH

        CG = np.abs(CH) ** 2

        gamma = {}
        alpha = {}
        #for i in range(len(schedule_list)):
        #    for star in schedule_list[i]:
        #        gamma[star] = min(P_bar * N / sum(Tx_times[j]) * CG[j, star] / (
        #                    (np.linalg.norm(model_diff[j], 2) ** 2)) for j in
        #                          schedule_list[i][star])
        #        alpha[star] = P_bar * N / sum(Tx_times[star]) / np.linalg.norm(model_diff[star], 2) ** 2
        #
        ## Calculate the #times for which device i receives, i.e., # of added AWGN
        #Rx_times = {node: sum(Tx_times[node]) for node in G.nodes()}
        ## Generate random noise a number of receiving times for device i
        ## A list (device_i's) of iterators that include a number of receiving times of received AWGN with shape (m,)
        #noise = {node: iter(
        #    np.sqrt(N0 / 2) * np.random.randn(Rx_times[node], m) + 1j * np.sqrt(N0 / 2) * np.random.randn(
        #        Rx_times[node], m))
        #         for node in G.nodes()}


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
                    param.data.copy_(theta_i[idx:idx + numel].reshape(param.size()))
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
        cumulative_transmitted_bits += 2 * max_neighbors * 32 * int(factor * num_params)
        transmitted_bits_per_iteration[round] = cumulative_transmitted_bits

    return average_test_metrics, transmitted_bits_per_iteration
