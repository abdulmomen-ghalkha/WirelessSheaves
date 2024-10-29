import argparse
import torch
from dfedu import run_dfedu
from sheaffmtl import run_dig_sheaf_fmtl, run_ana_sheaf_fmtl
from utils import read_vehicle_data, read_school_data, read_har_data, read_gleam_data, count_model_parameters, MultinomialLogisticRegression, LinearRegression, generate_random_adjacency_matrix, cross_entropy_loss_with_l2, mse_loss_with_l2, TwoSectionH, seq_scheduling
import torch.nn as nn
import numpy as np
import networkx as nx

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Algorithms')
    parser.add_argument('--dataset', type=str, required=True, help='Input data (vehicle or school)')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name (dFedU or Sheaf-FL)')
    parser.add_argument('--alpha', type=float, default=0.005, help='Learning rate to update the model of Sheaf-FMTL')
    parser.add_argument('--eta', type=float, default=0.001, help='Learning rate to update the linear maps in Sheaf-FMTL')
    parser.add_argument('--local_iterations', type=int, default=5, help='Number of local iterations for dFedU')
    parser.add_argument('--local_lr', type=float, default=0.001, help='Local learning rate for dFedU')
    parser.add_argument('--lambda-reg', type=float, default=0.001, help='Regularization strength')
    parser.add_argument('--factor', type=float, default=0.3, help='Factor for P_ij matrix size for Sheaf-FMTL')
    parser.add_argument('--num-rounds', type=int, default=200, help='Number of training rounds')
    parser.add_argument('--scheme', type=str, default='dig', help='scheme: dig- Digital, ana-analog')
    parser.add_argument('--power', type=float, default=0.1, help='Transmission power')
    parser.add_argument('--N', type=int, default=5000, help='Number of resources in one block')
    args = parser.parse_args()

    if args.dataset == 'vehicle':
        client_train_datasets, client_test_datasets, input_size, num_classes = read_vehicle_data()
        model = MultinomialLogisticRegression(input_size, num_classes)
        loss_func = cross_entropy_loss_with_l2
        metric_name = 'Accuracy'
        metric_func = lambda targets, predictions: torch.mean((targets == predictions).float())
    elif args.dataset == 'gleam':
        client_train_datasets, client_test_datasets, input_size, num_classes = read_gleam_data()
        model = MultinomialLogisticRegression(input_size, num_classes)
        loss_func = cross_entropy_loss_with_l2
        metric_name = 'Accuracy'
        metric_func = lambda targets, predictions: torch.mean((targets == predictions).float())
    elif args.dataset == 'har':
        client_train_datasets, client_test_datasets, input_size, num_classes = read_har_data()
        model = MultinomialLogisticRegression(input_size, num_classes)
        loss_func = cross_entropy_loss_with_l2
        metric_name = 'Accuracy'
        metric_func = lambda targets, predictions: torch.mean((targets == predictions).float())
    elif args.dataset == 'school':
        client_train_datasets, client_test_datasets = read_school_data()
        input_size = client_train_datasets[0][0][0].shape[0]
        model = LinearRegression(input_size)
        loss_func = mse_loss_with_l2
        metric_name = 'MSE'
        metric_func = nn.MSELoss()
    else:
        raise ValueError("Invalid dataset type.")

    num_clients = len(client_train_datasets)
    print(f"Number of clients {num_clients}")
    print(f"Number of features {client_train_datasets[0][0][0].shape}")

    adjacency_matrix = generate_random_adjacency_matrix(num_clients)
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(num_clients)]
    G = nx.from_numpy_array(adjacency_matrix)


    # -------------------- Channels Preparations ----------------------
    # Prepare the channels
    P_bar = args.power
    N = args.N
    K = num_clients
    d_min = 50
    d_max = 200
    Tmax = args.num_rounds
    rho = d_min + (d_max - d_min) * np.random.rand(K, 1)
    theta = 2 * np.pi * np.random.rand(K, 1)
    D = np.sqrt(rho ** 2 + rho.T ** 2 - 2 * (rho @ rho.T) * np.cos(theta - theta.T))
    # Fill in D[i,i] some non-zero value to avoid Inf in PL
    for i in range(K):
        if i:
            D[i, i] = D[i, i - 1]
        else:
            D[i, i] = D[i, i + 1]
    A0 = 10 ** (-3.35)
    d0 = 1
    gamma = 3.76
    PL = A0 * ((D / d0) ** (-gamma))
    # Generate random channels for unblocked edges of the given graph
    np.random.seed(10)
    CH_gen = np.random.randn(Tmax, K, K) / np.sqrt(2) + 1j * np.random.randn(Tmax, K, K) / np.sqrt(2)
    d = count_model_parameters(model)

    if args.scheme == 'dig':
        _, from_node_to_color_id = TwoSectionH(G)
        Chi = max(from_node_to_color_id.values()) + 1
    elif args.scheme == 'ana':
        schedule_list, Tx_times = seq_scheduling(G.copy())
    else:
        raise ValueError("Invalid transmission scheme")




    if args.algorithm == 'dFedU':
        average_test_metrics, transmitted_bits_per_iteration, resource_blocks_per_learning_round = run_dfedu(
            client_train_datasets,
            client_test_datasets,
            args.num_rounds,
            args.local_iterations,
            args.local_lr,
            neighbors,
            args.lambda_reg,
            model,
            loss_func,
            metric_func,
            metric_name,
            G, K, CH_gen, PL, Chi, P_bar, N,
            d
    )
    elif args.algorithm == 'Sheaf-FMTL' and args.scheme == 'dig':
        average_test_metrics, transmitted_bits_per_iteration, resource_blocks_per_learning_round = run_dig_sheaf_fmtl(
            client_train_datasets, 
            client_test_datasets, 
            args.num_rounds, 
            args.alpha, 
            args.eta, 
            args.lambda_reg, 
            args.factor, 
            adjacency_matrix,
            model, 
            loss_func,
            metric_func,
            metric_name,
            G,
            K,
            CH_gen,
            PL,
            Chi,
            P_bar,
            N,
            d
        )
    elif args.algorithm == 'Sheaf-FMTL' and args.scheme == 'ana':
        average_test_metrics, transmitted_bits_per_iteration, resource_blocks_per_learning_round = run_ana_sheaf_fmtl(
            client_train_datasets,
            client_test_datasets,
            args.num_rounds,
            args.alpha,
            args.eta,
            args.lambda_reg,
            args.factor,
            adjacency_matrix,
            model,
            loss_func,
            metric_func,
            metric_name,
            G,
            K,
            CH_gen,
            PL,
            schedule_list,
            Tx_times,
            P_bar,
            N,
            d
        )

    else:
        raise ValueError('Invalid algorithm. Choose either "dFedU" or "Sheaf-FL".')

    print(f'Average Test {metric_name}s:', average_test_metrics)
    print('Transmitted Bits per Iteration:', transmitted_bits_per_iteration)
    print('Number of users Resource Blocks', resource_blocks_per_learning_round)

if __name__ == '__main__':
    main()