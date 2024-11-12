import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import generate_random_adjacency_matrix, count_model_parameters, evaluate_metric
import networkx as nx
import cvxpy as cp
from cvxpy.atoms.lambda_sum_smallest import lambda_sum_smallest

def run_sheaf_fmtl_subgraph(client_train_datasets, client_test_datasets, num_rounds, alpha, eta, lambda_reg, factor, adjacency_matrix, model, loss_func, metric_func, metric_name, beta, Ct):
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

    # Subgraph sampling
    G = nx.from_numpy_array(adjacency_matrix)
    F = nx.line_graph(G)
    edges_to_colors = nx.coloring.greedy_color(F, strategy="largest_first")
    color_map_nodes = []
    color_map_edges = []
    for node in F:
        color_map_nodes.append(edges_to_colors[node])

    for edge in G.edges():
        color_map_edges.append(edges_to_colors[edge])

    # Group edges by color
    edges_by_color = {}
    for edge, color in edges_to_colors.items():
        if color not in edges_by_color:
            edges_by_color[color] = []
        edges_by_color[color].append(edge)

    # Create and visualize each subgraph for each color
    subgraphs = {}
    laplacians = {}

    for color, edges in edges_by_color.items():
        # Create a new subgraph with all nodes from G but only the selected edges
        subgraph = nx.Graph()
        subgraph.add_nodes_from(G.nodes())  # Add all nodes
        subgraph.add_edges_from(edges)      # Add only edges of this color
        subgraphs[color] = subgraph
        # Calculate the Laplacian matrix for the subgraph
        laplacian_matrix = nx.laplacian_matrix(subgraph).toarray()
        laplacians[color] = laplacian_matrix
    
    # Create subgraphs fo reach matching
    sub_graphs = []
    for color, L in laplacians.items():
        A = np.diag(np.diag(L)) - L 
        G_v = nx.from_numpy_array(A)
        sub_graphs.append(G_v)
        


    K = 20
    
    # Training loop
    for round in range(num_rounds):

        if round % K == 0:

            L_gaps = []
            
            for sub_g in sub_graphs:
                L_gap = 0
                for node in range(num_clients):
                    client_model = client_models[i]
                    with torch.no_grad():
                        theta_i = torch.cat([param.view(-1) for param in client_model.parameters()])
                        for neighbor in sub_g.neighbors(node):
                            P_ij = P[(node, neighbor)]
                            P_ji = P[(neighbor, node)]
                            theta_j = torch.cat([param.view(-1) for param in client_models[j].parameters()])
                            L_gap += 0.5 * np.linalg.norm(P_ij @ theta_i - P_ji @ theta_j)
                L_gaps.append(L_gap)
        
            n = len(sub_graphs)
            x = cp.Variable(n)
            algebraic_connectivity = lambda_sum_smallest(sum(x[i] * laplacians[i] for i in range(n)), 2)
            L_gap_distance = sum(x[i] * L_gaps[i] for i in range(n))


            # Define objective
            objective = cp.Maximize(algebraic_connectivity + beta * L_gap_distance)
            # Define constraints
            constraints = [
            x >= 0,          # Elementwise inequality 0 <= x
            cp.sum(x) <= Ct * n, # Elementwise inequality x <= 1
            x <= 1
            ]

            # Formulate and solve the problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            probabilities = np.clip(x.value, 0, 1, out=None)

            # Original list of elements
            elements = np.ones(len(probabilities), np.int32)


        # Define the number of trials for each element (for example, 1 trial per element)
        n_trials = 1

        # Generate the new list with binomial sampling
        sampled_elements = [np.random.binomial(n_trials, p) * element for element, p in zip(elements, probabilities)]

        print(round)
        G_merged = nx.Graph()
        G_merged.add_nodes_from(range(num_clients))
        for p, sub_g in zip(sampled_elements, sub_graphs):
            if p:
                G_merged = nx.compose(G_merged, sub_g)



        merged_adjacency_matrix = nx.adjacency_matrix(G_merged).toarray()
        max_neighbors = max(len(n) for n in neighbors)

        



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
                    if merged_adjacency_matrix[i, j] == 1:
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
                    if merged_adjacency_matrix[i, j] == 1:
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
        edges = [np.nonzero(merged_adjacency_matrix[i])[0].tolist() for i in range(num_clients)]
        num_edges = 0.5 * sum(len(n) for n in neighbors)
        cumulative_transmitted_bits += 2 * num_edges * 32 * int(factor*num_params)
        transmitted_bits_per_iteration[round] = cumulative_transmitted_bits
        
    

 
    return average_test_metrics, transmitted_bits_per_iteration