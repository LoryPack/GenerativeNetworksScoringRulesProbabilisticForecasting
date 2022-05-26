#!/bin/bash
# This scripts shows how to reproduce experiments for Lorenz63 model.

# create folders:
mkdir results
mkdir results/lorenz
mkdir results/lorenz/datasets

#  generate the training data:

python3 generate_data.py lorenz --window_size 10 --n_steps 20000

# RUN things:
# --- Deterministic ---
# run the training script
python3 train_nn.py lorenz regression  --lr 0.001 --epochs 1000 --seed 12  --batch_size 1000 --hidden_size_rnn 8
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz regression  --lr 0.001 --seed 12  --batch_size 1000 --hidden_size_rnn 8

# --- SRs ---
# run the training script for Energy Score:
python3 train_nn.py lorenz SR --scoring_rule Energy --lr 0.01 --epochs 1000 --ensemble_size 10 --seed 12  --batch_size 1000 --hidden_size_rnn 8
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz SR --scoring_rule Energy --lr 0.01  --training_ensemble_size 10 --prediction_ensemble_size 200 --seed 12  --batch_size 1000 --hidden_size_rnn 8

# --- GANs ---
# run the training script for GAN:
python3 train_nn.py lorenz GAN --lr 0.0001 --lr_c 0.001 --epochs 1000 --seed 12  --batch_size 1000 --hidden_size_rnn 8
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz GAN --lr 0.0001 --lr_c 0.001  --prediction_ensemble_size 200 --seed 12 --hidden_size_rnn 8  --batch_size 1000

# run the training script for WGAN-GP:
python3 train_nn.py lorenz WGAN_GP --lr 0.0003 --lr_c 0.03 --epochs 1000 --seed 12  --batch_size 1000 --hidden_size_rnn 8 --critic_steps 5
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz WGAN_GP --lr 0.0003 --lr_c 0.03  --prediction_ensemble_size 200 --seed 12  --batch_size 1000 --hidden_size_rnn 8 --critic_steps 5

# run python3 train_nn.py -h to see all possible arguments; similar for predict_test_plot.py

# Create the comparison figure (Figure 2a in the paper)
python predict_test_plot_comparison.py lorenz SR --training_ensemble_size 10 --prediction_ensemble_size 1000
