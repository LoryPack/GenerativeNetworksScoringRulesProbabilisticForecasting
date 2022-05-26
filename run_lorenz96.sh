#!/bin/bash
# This scripts shows how to reproduce experiments for lorenz96 model.

# create folders:
mkdir results
mkdir results/lorenz96
mkdir results/lorenz96/datasets

#  generate the training data:

python3 generate_data.py lorenz96 --window_size 10 --n_steps 20000

# RUN things:
# --- Deterministic ---
# run the training script
python3 train_nn.py lorenz96 regression  --lr 0.003 --epochs 1000 --seed 12  --batch_size 1000 --hidden_size_rnn 8
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz96 regression  --lr 0.003 --seed 12  --batch_size 1000 --hidden_size_rnn 8

# --- SRs ---
# run the training script for EnergyKernel Score:
python3 train_nn.py lorenz96 SR --scoring_rule EnergyKernel --lr 0.001 --epochs 1000 --ensemble_size 10 --seed 12  --batch_size 1000 --hidden_size_rnn 32
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz96 SR  --scoring_rule EnergyKernel --lr 0.001  --training_ensemble_size 10 --prediction_ensemble_size 200 --seed 12  --batch_size 1000 --hidden_size_rnn 32

# --- GANs ---
# run the training script for GAN:
python3 train_nn.py lorenz96 GAN --lr 0.0001 --lr_c 0.001 --epochs 1000 --seed 12  --batch_size 1000 --hidden_size_rnn 64
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz96 GAN --lr 0.0001 --lr_c 0.001  --prediction_ensemble_size 200 --seed 12 --hidden_size_rnn 64  --batch_size 1000

# run the training script for WGAN-GP:
python3 train_nn.py lorenz96 WGAN_GP --lr 0.0001 --lr_c 0.01 --epochs 1000 --seed 12  --batch_size 1000 --hidden_size_rnn 64 --critic_steps 5
# compute performance metrics and produce individual figures
python3 predict_test_plot.py lorenz96 WGAN_GP --lr 0.0001 --lr_c 0.01  --prediction_ensemble_size 200 --seed 12  --batch_size 1000 --hidden_size_rnn 64 --critic_steps 5

# run python3 train_nn.py -h to see all possible arguments; similar for predict_test_plot.py

# Create the comparison figure (Figure 2b in the paper)
python predict_test_plot_comparison.py lorenz96 SR --training_ensemble_size 10 --prediction_ensemble_size 1000
