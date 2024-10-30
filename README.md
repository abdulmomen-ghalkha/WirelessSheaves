# Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach

This repository is the official implementation of the paper: "Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach". The vehicle, gleam and school datasets are read from a .mat files. Both the .mat files and the datasets read functions are taken from [here](https://github.com/kenziyuliu/private-cross-silo-fl).

## Requirements

To install the required dependencies, run the following command:

pip install -r requirements.txt

## Usage

To run the code, use the following command:

python main.py --input-data <path_to_data> --algorithm <algorithm_name> [--alpha ALPHA] [--eta ETA] [--lambda-reg LAMBDA_REG] [--local_iterations LOCAL_ITERATIONS] [--local_lr LOCAL_LR] [--factor FACTOR] [--num-rounds NUM_ROUNDS]

- `--dataset`: Specify the input data. Choose either "vehicle", "school", "gleam", or "har".
- `--algorithm`: Specify the algorithm name. Choose either "dFedU" or "Sheaf-FMTL".
- `--alpha`: Learning rate for updating the models of Sheaf-FMTL (default: 0.005).
- `--eta`: Update rate for updating the maps of Sheaf-FMTL (default: 0.001).
- `--lambda-reg`: Regularization strength (default: 0.001).
- `--local_iterations`: Number of local iterations for dFedU (default: 5).
- `--local_lr`: Local learning rate for dFedU (default: 0.001).
- `--factor`: Factor for P_ij matrix size in Sheaf-FMTL (default: 0.1).
- `--num-rounds`: Number of training rounds (default: 200).

## Examples

To run dFedU on the vehicle dataset with default parameters:

python main.py --dataset vehicle --algorithm dFedU

To run Sheaf-FMTL on the school dataset with custom parameters:

python main.py --dataset school --algorithm Sheaf-FMTL --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.1 --num-rounds 200

## Results

The script will output the average test MSEs and the transmitted bits per iteration for the selected algorithm and dataset.