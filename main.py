import argparse
import torch
from dfedu import run_dfedu
from sheaffmtl import run_sheaf_fmtl
from sheaffmtl_subgraph import run_sheaf_fmtl_subgraph
from sheaffmtl_subgraph_optimized import run_sheaf_fmtl_subgraph_optimized
from local_training import run_local_training
from utils import read_vehicle_data, read_school_data, read_har_data, read_gleam_data, MultinomialLogisticRegression, LinearRegression, generate_random_adjacency_matrix, cross_entropy_loss_with_l2, mse_loss_with_l2
import torch.nn as nn
import numpy as np
import networkx as nx
import h5py

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Algorithms')
    parser.add_argument('--dataset', type=str, required=True, help='Input data (vehicle or school)')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name (dFedU, Sheaf-FMTL, Sheaf-FMTL, Sheaf-FMTL-subgraph, Local-training)')
    parser.add_argument('--alpha', type=float, default=0.005, help='Learning rate to update the model of Sheaf-FMTL')
    parser.add_argument('--eta', type=float, default=0.001, help='Learning rate to update the linear maps in Sheaf-FMTL')
    parser.add_argument('--local_iterations', type=int, default=5, help='Number of local iterations for dFedU')
    parser.add_argument('--local_lr', type=float, default=0.001, help='Local learning rate for dFedU')
    parser.add_argument('--lambda-reg', type=float, default=0.001, help='Regularization strength')
    parser.add_argument('--factor', type=float, default=0.3, help='Factor for P_ij matrix size for Sheaf-FMTL')
    parser.add_argument('--num-rounds', type=int, default=200, help='Number of training rounds')
    parser.add_argument('--beta', type=float, default=0.01, help='Connectivity and Lgap tradeoff')
    parser.add_argument('--Ct', type=float, default=1.0, help='Communication resources saveup')
    parser.add_argument('--times', type=int, default=5, help='Number of MC runs')
    parser.add_argument('--K', type=int, default=20, help='Sampling probabilities Optimization step')
    parser.add_argument('--drop', type=int, default=0, help='features to drop')
    args = parser.parse_args()

    if args.dataset == 'vehicle':
        client_train_datasets, client_test_datasets, input_sizes, num_classes = read_vehicle_data(features_to_drop=args.drop)
        models = []
        for i in range(len(client_train_datasets)):

            model = MultinomialLogisticRegression(input_sizes[i], num_classes)
            models.append(model)
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
    
    for time in range(args.times):

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
                models, 
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
                models, 
                loss_func,
                metric_func,
                metric_name,
                args.beta, 
                args.Ct,
                args.K
            )
        elif args.algorithm == 'Sheaf-FMTL-subgraph-op':
            average_test_metrics, transmitted_bits_per_iteration = run_sheaf_fmtl_subgraph_optimized(
                client_train_datasets, 
                client_test_datasets, 
                args.num_rounds, 
                args.alpha, 
                args.eta, 
                args.lambda_reg, 
                args.factor, 
                adjacency_matrix,
                models, 
                loss_func,
                metric_func,
                metric_name,
                args.beta, 
                args.Ct,
                args.K
            )
        elif args.algorithm == 'local-training':
            average_test_metrics, transmitted_bits_per_iteration = run_local_training(
                client_train_datasets, 
                client_test_datasets, 
                args.num_rounds, 
                args.alpha, 
                args.eta, 
                args.lambda_reg, 
                args.factor, 
                adjacency_matrix,
                models, 
                loss_func,
                metric_func,
                metric_name,
                args.beta, 
                args.Ct,
                args.K
            )
        else:
            raise ValueError('Invalid algorithm. Choose either "dFedU", "Sheaf-FL", Sheaf-FMTL-subgraph, or Local-training.')

        print(f'Average Test {metric_name}s:', average_test_metrics)
        print('Transmitted Bits per Iteration:', transmitted_bits_per_iteration)
        # Convert to numpy arrays
        average_test_metrics_array = np.array(average_test_metrics)
        transmitted_bits_per_iteration_array = np.array(transmitted_bits_per_iteration)
        

        # Construct the filename with parameters
        #alg = f"{args.dataset}_{args.algorithm}_{args.local_lr}_{args.alpha}_{args.eta}_{args.local_iterations}_{num_clients}u_{lambda-reg}b_{local_epochs}_{times}"
        alg = (
            f"{args.dataset}_{args.algorithm}_{args.local_lr}_{args.alpha}_"
            f"{args.eta}_{args.local_iterations}_{args.lambda_reg}_factor_{args.factor}_"
            f"{args.num_rounds}_beta_{args.beta}_Ct_{args.Ct}_time_{time}_K_{args.K}"
            f"droppedfeatures_{args.drop}"
        )

        # Convert lists to numpy arrays
        average_test_metrics_array = np.array(average_test_metrics)
        transmitted_bits_per_iteration_array = np.array(transmitted_bits_per_iteration)

        # Save to an HDF5 file
        with h5py.File(f"./results/{alg}.h5", 'w') as hf:
            hf.create_dataset('average_test_metrics', data=average_test_metrics_array)
            hf.create_dataset('transmitted_bits_per_iteration', data=transmitted_bits_per_iteration_array)       

if __name__ == '__main__':
    main()