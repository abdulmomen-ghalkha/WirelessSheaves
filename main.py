import argparse
import torch
from dfedu import run_dfedu
from sheaffmtl import run_sheaf_fmtl
from sheaffmtl_subgraph import run_sheaf_fmtl_subgraph
from utils import read_vehicle_data, read_school_data, read_har_data, read_gleam_data, MultinomialLogisticRegression, LinearRegression, generate_random_adjacency_matrix, cross_entropy_loss_with_l2, mse_loss_with_l2
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

    adjacency_matrix = generate_random_adjacency_matrix(num_clients)
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(num_clients)]

    if args.algorithm == 'dFedU':
        average_test_metrics, transmitted_bits_per_iteration = run_dfedu(
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
            metric_name
    )
    elif args.algorithm == 'Sheaf-FMTL':
        average_test_metrics, transmitted_bits_per_iteration = run_sheaf_fmtl(
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
            metric_name
        )
    elif args.algorithm == 'Sheaf-FMTL-subgraph':
        average_test_metrics, transmitted_bits_per_iteration = run_sheaf_fmtl_subgraph(
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
            metric_name
        )
    else:
        raise ValueError('Invalid algorithm. Choose either "dFedU" or "Sheaf-FL".')

    print(f'Average Test {metric_name}s:', average_test_metrics)
    print('Transmitted Bits per Iteration:', transmitted_bits_per_iteration)

if __name__ == '__main__':
    main()