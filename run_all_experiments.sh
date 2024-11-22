#!/bin/bash

# # Run each Python command sequentially
#python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.1 --Ct 0.1 --K 10--drop 0
#python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.1 --K 10 --drop 0

#python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.001 --Ct 0.1 --K 10 --drop 0


python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.1 --Ct 0.1 --K 200 --drop 4
python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.01 --Ct 0.1 --K 200 --drop 4

python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.001 --Ct 0.1 --K 200 --drop 4

# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.4 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.5 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.9 --K 10 --drop 0

# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.4 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.4 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 10 --drop 0


# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.1 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 10 --drop 0


# python main.py --dataset school --algorithm Sheaf-FMTL --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 10 --drop 0

#python main.py --dataset school --algorithm dFedU --K 10



# # Run each Python command sequentially --drop 0
# python main.py --dataset vehicle --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.1 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.4 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.5 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.9 --K 50 --drop 0

# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.4 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.4 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 50 --drop 0
 
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.1 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 50 --drop 0


python main.py --dataset vehicle --algorithm Sheaf-FMTL --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.2 --num-rounds 200 --times 5 --beta 0.001 --Ct 0.4 --K 200 --drop 4

python main.py --dataset vehicle --algorithm dFedU --K 200 --times 5 --drop 4


# # Run each Python command sequentially --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.1 --K 200 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.4 --K 200 --drop 0
# #python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.5 --K 200 --drop 0
# #python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.9 --K 200 --drop 0

# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.1 --Ct 0.4 --K 200 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.01 --Ct 0.4 --K 200 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 200 --drop 0

# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.1 --K 200 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 200 --drop 0

# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 1.0 --Ct 0.1 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 1.0 --Ct 0.4 --K 10 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 1.0 --Ct 0.1 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 1.0 --Ct 0.4 --K 50 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 1.0 --Ct 0.1 --K 200 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL-subgraph --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 1.0 --Ct 0.4 --K 200 --drop 0



 --drop 0
# python main.py --dataset school --algorithm Sheaf-FMTL --alpha 0.005 --eta 0.05 --lambda-reg 0.02 --factor 0.4 --num-rounds 200 --times 1 --beta 0.001 --Ct 0.4 --K 200 --drop 0

# python main.py --dataset school --algorithm dFedU --K 200 --drop 0


git add .

git commit -m "New results for different k and times"

git push -u origin main
