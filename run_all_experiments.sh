#!/bin/bash

# Run each Python command sequentially
python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.1
python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.4
python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.5
python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.9

python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.1 --Ct 0.4
python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.4
python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.001 --Ct 0.4


python main.py --dataset school --algorithm Sheaf-FMTL --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.8 --num-rounds 200 --times 5 --beta 0.001 --Ct 0.4

python main.py --dataset school --algorithm dFedU


wait
