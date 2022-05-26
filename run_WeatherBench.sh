#!/bin/bash
# This scripts shows how to reproduce experiments for WeatherBench model. Running these will take a long time and
# required GPUs.

# create folders:
mkdir results
mkdir results/WeatherBench

WEATHERBENCH_DATA_FOLDER=""  # need to put here the exact path to WeatherBench data folder; see download instructions in README.md

COMMON_ARGUMENTS=" --weatherbench_data_folder ${WEATHERBENCH_DATA_FOLDER} --load_all_data_GPU --seed 12  --batch_size 128 --cuda" 

# adding argument --weatherbench_small runs on small patches of the WeatherBench data; in this way it is faster

# RUN things:
# --- Deterministic ---
# run the training script
python3 train_nn.py WeatherBench regression --lr 0.01 --epochs 1000 --weight_decay 0.001 --scheduler_gamma 0.8 ${COMMON_ARGUMENTS}
# compute performance metrics and produce individual figures
python3 predict_test_plot.py WeatherBench regression   --lr 0.01 ${COMMON_ARGUMENTS}
python plot_weatherbench.py  regression --lr 0.01  --training_ensemble_size 10 ${COMMON_ARGUMENTS}

# --- SRs ---
# run the training script for EnergyKernel Score:
python3 train_nn.py WeatherBench SR   --scoring_rule Energy --patched --patch_size 8 --lr 0.0001 --epochs 1000 --ensemble_size 10 ${COMMON_ARGUMENTS} --weatherbench_small
# compute performance metrics and produce individual figures
python3 predict_test_plot.py WeatherBench SR  --scoring_rule Energy --patched --patch_size 8 --lr 0.0001  --training_ensemble_size 10 --prediction_ensemble_size 200 ${COMMON_ARGUMENTS}
python plot_weatherbench.py SR --scoring_rule Energy --patched --patch_size 8 --lr 0.0001 --training_ensemble_size 10 --prediction_ensemble_size 100 ${COMMON_ARGUMENTS}

# --- GANs ---
# run the training script for GAN:
python3 train_nn.py WeatherBench    GAN --lr 0.00001 --lr_c 0.00001 --epochs 1000  ${COMMON_ARGUMENTS} --weatherbench_small
# compute performance metrics and produce individual figures
python3 predict_test_plot.py WeatherBench GAN --lr 0.00001 --lr_c 0.00001  --prediction_ensemble_size 200 --seed 12   --batch_size 128 ${COMMON_ARGUMENTS}  # 200
python plot_weatherbench.py  GAN --lr 0.00001 --lr_c 0.00001  --training_ensemble_size 10 --prediction_ensemble_size 100 ${COMMON_ARGUMENTS}

# run the training script for WGAN-GP:
python3 train_nn.py WeatherBench WGAN_GP  --lr 0.00001 --lr_c 0.01 --epochs 1000   --critic_steps 5 ${COMMON_ARGUMENTS}
# compute performance metrics and produce individual figures
python3 predict_test_plot.py WeatherBench WGAN_GP  --lr 0.00001 --lr_c 0.01  --prediction_ensemble_size 200   --critic_steps 5 ${COMMON_ARGUMENTS}
python plot_weatherbench.py  WGAN_GP --lr 0.00001 --lr_c 0.01  --training_ensemble_size 10 --prediction_ensemble_size 100 --critic_steps 5 ${COMMON_ARGUMENTS}

# run python3 train_nn.py -h to see all possible arguments; similar for predict_test_plot.py
