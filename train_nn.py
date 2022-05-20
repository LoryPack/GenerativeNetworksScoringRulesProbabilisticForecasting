import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset
# import and set up the typeguard
from typeguard.importhook import install_import_hook

# comment these out when deploying:
install_import_hook('src.nn')
install_import_hook('src.scoring_rules')
install_import_hook('src.utils')
install_import_hook('src.weatherbench_utils')
install_import_hook('src.unet_utils')

from src.nn import InputTargetDataset, UNet2D, fit, fit_adversarial, \
    ConditionalGenerativeModel, createGenerativeFCNN, createCriticFCNN, test_epoch, PatchGANDiscriminator, \
    DiscardWindowSizeDim, get_target, LayerNormMine, createGenerativeGRUNN, createCriticGRUNN, \
    DiscardNumberGenerationsInOutput, createGRUNN, createFCNN
from src.scoring_rules import EnergyScore, KernelScore, VariogramScore, PatchedScoringRule, SumScoringRules, \
    ScoringRulesForWeatherBench, ScoringRulesForWeatherBenchPatched, LossForWeatherBenchPatched
from src.utils import plot_losses, save_net, save_dict_to_json, estimate_bandwidth_timeseries, lorenz96_variogram, \
    def_loader_kwargs, load_net, weight_for_summed_score, weatherbench_variogram_haversine
from src.parsers import parser_train_net, define_masks, nonlinearities_dict, setup
from src.weatherbench_utils import load_weatherbench_data

# --- parser ---
parser = parser_train_net()
args = parser.parse_args()

model = args.model
method = args.method
scoring_rule = args.scoring_rule
kernel = args.kernel
patched = args.patched
base_measure = args.base_measure
epochs = args.epochs
# no_scheduler = args.no_scheduler
root_folder = args.root_folder
model_folder = args.model_folder
datasets_folder = args.datasets_folder
weatherbench_data_folder = args.weatherbench_data_folder
weatherbench_small = args.weatherbench_small
unet_noise_method = args.unet_noise_method
unet_large = args.unet_large
lr = args.lr
lr_c = args.lr_c
batch_size = args.batch_size
early_stopping = not args.no_early_stop
start_epoch_early_stopping = args.epochs_before_early_stopping
epochs_early_stopping_interval = args.epochs_test_interval
critic_steps_every_generator_step = args.critic_steps_every_generator_step
use_tqdm = not args.no_tqdm
save_net_flag = not args.no_save_net
continue_training_net_if_available = args.continue_training_net_if_available
cuda = args.cuda
load_all_data_GPU = args.load_all_data_GPU
ensemble_size = args.ensemble_size
nonlinearity = args.nonlinearity
data_size = args.data_size
auxiliary_var_size = args.auxiliary_var_size
seed = args.seed
lambda_gp = args.lambda_gp
gamma = args.gamma_kernel_score
notrain_if_done_before = args.notrain_if_done_before
patch_size = args.patch_size
no_RNN = args.no_RNN
hidden_size_rnn = args.hidden_size_rnn
weight_decay = args.weight_decay
scheduler_gamma = args.scheduler_gamma
args_dict = args.__dict__

model_is_weatherbench = model == "WeatherBench"

nn_model = "unet" if model_is_weatherbench else ("fcnn" if no_RNN else "rnn")

datasets_folder, nets_folder, data_size, auxiliary_var_size, name_postfix, unet_depths, patch_size, method_is_gan, hidden_size_rnn = \
    setup(model, root_folder, model_folder, datasets_folder, data_size, method, scoring_rule, kernel, patched,
          patch_size, ensemble_size, auxiliary_var_size, critic_steps_every_generator_step, base_measure, lr, lr_c,
          batch_size, args.no_early_stop, unet_noise_method, unet_large, nn_model, hidden_size_rnn)

# stop if the net exists already:
if notrain_if_done_before and os.path.exists(nets_folder + f"net{name_postfix}.pth"):
    print("Stopping as net with this setup was trained before.")
    exit(0)

# if continue_training_net_if_available, check if the net is available and in that case start training from that
continue_training_net = continue_training_net_if_available and os.path.exists(nets_folder + f"net{name_postfix}.pth")

# create the nets folder:
os.makedirs(nets_folder, exist_ok=True)

# --- data handling ---
if not model_is_weatherbench:
    input_data_train = torch.load(datasets_folder + "train_x.pty")
    target_data_train = torch.load(datasets_folder + "train_y.pty")
    input_data_val = torch.load(datasets_folder + "val_x.pty")
    target_data_val = torch.load(datasets_folder + "val_y.pty")

    window_size = input_data_train.shape[1]

    # create the train and val loaders:
    dataset_train = InputTargetDataset(input_data_train, target_data_train,
                                       "cuda" if cuda and load_all_data_GPU else "cpu")
    dataset_val = InputTargetDataset(input_data_val, target_data_val, "cuda" if cuda and load_all_data_GPU else "cpu")
else:
    dataset_train, dataset_val = load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU,
                                                        weatherbench_small=weatherbench_small)
    len_dataset_train = len(dataset_train)
    len_dataset_val = len(dataset_val)
    print("Training set size:", len_dataset_train)
    print("Validation set size:", len_dataset_val)
    args_dict["len_dataset_train"] = len_dataset_train
    args_dict["len_dataset_val"] = len_dataset_val

loader_kwargs = def_loader_kwargs(cuda, load_all_data_GPU)

# loader_kwargs.update(loader_kwargs_2)  # if you want to add other loader arguments

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, **loader_kwargs)
if len(dataset_val) > 0:
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **loader_kwargs)
    if model_is_weatherbench:
        # obtain the target tensor to estimate the gamma for kernel SR:
        target_data_val = get_target(data_loader_val, cuda).flatten(1, -1)
else:
    data_loader_val = None

# --- losses ---
# instantiate the loss according to the chosen SR; each SR takes as input: (net_output, target)
if not method_is_gan and not method == "regression":
    if patched and not model_is_weatherbench:
        masks = define_masks[model](data_size=data_size, patch_size=patch_size)
    if scoring_rule in ["Kernel", "KernelVariogram", "EnergyKernel"]:
        # estimate the gamma value
        if gamma is None:
            if patched and not model_is_weatherbench:
                # determine the gamma using the first patch only. This assumes that the values of the variables
                # are roughly the same in the different patches.
                gamma = estimate_bandwidth_timeseries(target_data_val[:, masks[0]], return_values=["median"])
            else:
                gamma = estimate_bandwidth_timeseries(target_data_val, return_values=["median"])
            print(f"Estimated gamma: {gamma:.4f}")
            args_dict["gamma_kernel_score"] = float(gamma.cpu().numpy())
        if kernel == "gaussian":
            sr_instance = KernelScore(sigma=gamma)
        else:
            sr_instance = KernelScore(kernel="rational_quadratic", alpha=gamma ** 2)
    elif scoring_rule == "Energy":
        sr_instance = EnergyScore()
    if scoring_rule in ["Variogram", "EnergyVariogram", "KernelVariogram"]:
        variogram = None

        if model in ["lorenz96", ]:
            variogram = lorenz96_variogram(data_size)
        elif model == "WeatherBench":
            # variogram = weatherbench_variogram(weatherbench_small=weatherbench_small)  # this is the old one
            variogram = weatherbench_variogram_haversine(weatherbench_small=weatherbench_small)

        if variogram is not None and cuda:
            variogram = variogram.cuda()

        if scoring_rule == "Variogram":
            sr_instance = VariogramScore(variogram=variogram, max_batch_size=16 if model_is_weatherbench else None)
        else:
            if scoring_rule == "EnergyVariogram":
                sr1_instance = EnergyScore()
            elif scoring_rule == "KernelVariogram":
                if kernel == "gaussian":
                    sr1_instance = KernelScore(sigma=gamma)
                else:
                    sr1_instance = KernelScore(kernel="rational_quadratic", alpha=gamma ** 2)
            sr2_instance = VariogramScore(variogram=variogram, max_batch_size=16 if model_is_weatherbench else None)
            weight_list = weight_for_summed_score(scoring_rule, model_is_weatherbench)
            args_dict["weight_list"] = weight_list
            sr_instance = SumScoringRules((sr1_instance, sr2_instance), weight_list=weight_list)

    if scoring_rule == "EnergyKernel":
        sr1_instance = EnergyScore()
        if kernel == "gaussian":
            sr2_instance = KernelScore(sigma=gamma)
        else:
            sr2_instance = KernelScore(kernel="rational_quadratic", alpha=gamma ** 2)
        weight_list = weight_for_summed_score(scoring_rule, model_is_weatherbench)
        args_dict["weight_list"] = weight_list
        sr_instance = SumScoringRules((sr1_instance, sr2_instance), weight_list=weight_list)

    if model_is_weatherbench:
        if patched:
            # wrap the scoring rule in a new one for the weatherbench dataset, as you need to flatten somehow the data:
            sr1_instance = ScoringRulesForWeatherBenchPatched(sr_instance, patch_step=patch_size // 2,
                                                              patch_size=patch_size)
            sr2_instance = ScoringRulesForWeatherBench(sr_instance)
            # the patched version is proper but not strictly; add it therefore to the original SR on the full data to
            # make it strictly proper.
            weight_list = weight_for_summed_score(scoring_rule, model_is_weatherbench, patch_size=patch_size)
            args_dict["weight_list"] = weight_list
            sr_instance = SumScoringRules((sr1_instance, sr2_instance), weight_list=weight_list)
        else:
            # wrap the scoring rule in a new one for the weatherbench dataset, as you need to flatten somehow the data:
            sr_instance = ScoringRulesForWeatherBench(sr_instance)

    if patched and not model_is_weatherbench:
        patched_sr = PatchedScoringRule(sr_instance, masks)
        loss_fn = patched_sr.estimate_score_batch
    else:
        loss_fn = sr_instance.estimate_score_batch
elif method == "regression":
    # use the RMSE loss function
    loss_fn = nn.MSELoss()
    # add a patched version for WeatherBench
    if patched:
        if model_is_weatherbench:
            loss_fn = LossForWeatherBenchPatched(loss_fn)
        else:
            raise NotImplementedError

# --- networks ---
if seed is not None:  # set seed for network instantiating
    torch.manual_seed(seed)
if method == "regression":
    if nn_model == "unet":
        # NOTE: make sure that a channels dimension exists
        net_class = UNet2D
        unet_kwargs = {"in_channels": data_size[0], "out_channels": 1,
                       "noise_method": "no noise", "conv_depths": unet_depths}
        net = DiscardWindowSizeDim(net_class(**unet_kwargs))
    elif nn_model == "rnn":
        output_size = data_size
        gru_layers = 1
        gru_hidden_size = hidden_size_rnn
        net = createGRUNN(data_size=data_size, gru_hidden_size=gru_hidden_size,
                          output_size=output_size, hidden_sizes=None, gru_layers=gru_layers,
                          nonlinearity=nonlinearities_dict[nonlinearity])()
    else:
        net = createFCNN(input_size=window_size * data_size, output_size=data_size, unsqueeze_output=True)()

    if continue_training_net:
        net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardNumberGenerationsInOutput, net)
    else:
        # create net
        net = DiscardNumberGenerationsInOutput(net)
else:  # generative-SR and GAN

    wrap_net = True
    number_generations_per_forward_call = ensemble_size if (method == "generative" or method == "Energy_SR_GAN") else 1
    # create generative net:
    if nn_model == "fcnn":
        input_size = window_size * data_size + auxiliary_var_size
        output_size = data_size
        hidden_sizes_list = [int(input_size * 1.5), int(input_size * 3), int(input_size * 3),
                             int(input_size * 0.75 + output_size * 3), int(output_size * 5)]
        inner_net = createGenerativeFCNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes_list,
                                         nonlinearity=nonlinearities_dict[nonlinearity])()
    elif nn_model == "rnn":
        output_size = data_size
        gru_layers = 1
        gru_hidden_size = hidden_size_rnn
        inner_net = createGenerativeGRUNN(data_size=data_size, gru_hidden_size=gru_hidden_size,
                                          noise_size=auxiliary_var_size,
                                          output_size=output_size, hidden_sizes=None, gru_layers=gru_layers,
                                          nonlinearity=nonlinearities_dict[nonlinearity])()
    elif nn_model == "unet":
        # select the noise method here:
        inner_net = UNet2D(in_channels=data_size[0], out_channels=1, noise_method=unet_noise_method,
                           number_generations_per_forward_call=number_generations_per_forward_call,
                           conv_depths=unet_depths)
        if unet_noise_method in ["sum", "concat"]:
            # here we overwrite the auxiliary_var_size above, as there is a precise constraint
            downsampling_factor, n_channels = inner_net.calculate_downsampling_factor()
            if weatherbench_small:
                auxiliary_var_size = torch.Size(
                    [n_channels, 16 // downsampling_factor, 16 // downsampling_factor])
            else:
                auxiliary_var_size = torch.Size(
                    [n_channels, data_size[1] // downsampling_factor, data_size[2] // downsampling_factor])
        elif unet_noise_method == "dropout":
            wrap_net = False  # do not wrap in the conditional generative model

    if wrap_net:
        # the following wraps the nets above and takes care of generating the auxiliary variables at each forward call
        if continue_training_net:
            net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
                           size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                           number_generations_per_forward_call=number_generations_per_forward_call, seed=seed + 1)
        else:
            net = ConditionalGenerativeModel(inner_net, size_auxiliary_variable=auxiliary_var_size, seed=seed + 1,
                                             number_generations_per_forward_call=number_generations_per_forward_call,
                                             base_measure=base_measure)
    else:
        if continue_training_net:
            net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardWindowSizeDim, inner_net)
        else:
            net = DiscardWindowSizeDim(inner_net)

    if method_is_gan:
        if nn_model == "fcnn":
            model_class = createCriticFCNN(input_size=(window_size + 1) * data_size,
                                           end_sigmoid=method == "GAN")
            if continue_training_net:
                critic = load_net(nets_folder + f"critic{name_postfix}.pth", model_class)
            else:
                critic = model_class()
        elif nn_model == "rnn":
            gru_layers_critic = 1
            gru_hidden_size = hidden_size_rnn
            model_class = createCriticGRUNN(data_size, gru_hidden_size, gru_layers=gru_layers_critic,
                                            end_sigmoid=method == "GAN")
            if continue_training_net:
                critic = load_net(nets_folder + f"critic{name_postfix}.pth", model_class)
            else:
                critic = model_class()
        elif nn_model == "unet":
            # Using PatchGanDiscriminator model popularised in the following work:https://arxiv.org/abs/1611.07004v3
            # This discriminator seems to be popular for image based GAN work
            if continue_training_net:
                critic = load_net(nets_folder + f"critic{name_postfix}.pth",
                                  PatchGANDiscriminator, in_channels=1 + data_size[0], last_layer_filters=32,
                                  n_layers=3, end_sigmoid=method == "GAN",
                                  norm_layer=nn.BatchNorm2d if method != "WGAN_GP" else LayerNormMine)
            else:
                critic = PatchGANDiscriminator(in_channels=1 + data_size[0], first_layer_filters=32, n_layers=3,
                                               end_sigmoid=method == "GAN",
                                               norm_layer=nn.BatchNorm2d if method != "WGAN_GP" else LayerNormMine)
                # in general, we have to put in_channels = (window_size * fields_context) + fields_target

# --- network tools ---
if cuda:
    net.cuda()

# optimizer
optimizer_kwargs = {"weight_decay": weight_decay}  # l2 regularization
args_dict["weight_decay"] = optimizer_kwargs["weight_decay"]
optimizer = Adam(net.parameters(), lr=lr, **optimizer_kwargs)

# scheduler
scheduler_steps = 10
scheduler_gamma = scheduler_gamma
scheduler = lr_scheduler.StepLR(optimizer, scheduler_steps, gamma=scheduler_gamma, last_epoch=-1)
args_dict["scheduler_steps"] = scheduler_steps
args_dict["scheduler_gamma"] = scheduler_gamma

if method_is_gan:
    if cuda:
        critic.cuda()
    optimizer_kwargs = {}
    optimizer_c = Adam(critic.parameters(), lr=lr_c, **optimizer_kwargs)
    # dummy scheduler:
    scheduler_c = lr_scheduler.StepLR(optimizer_c, 8, gamma=1, last_epoch=-1)

string = f"Train {method} network for {model} model with lr {lr} "
if not method_is_gan:
    string += f"using {scoring_rule} scoring rule"
else:
    string += f"and critic lr {lr_c}"
print(string)

# --- train ---
start = time()
if method_is_gan:
    # load the previous losses if available:
    if continue_training_net:
        train_loss_list_g = np.load(nets_folder + f"train_loss_g{name_postfix}.npy").tolist()
        train_loss_list_c = np.load(nets_folder + f"train_loss_c{name_postfix}.npy").tolist()
    else:
        train_loss_list_g = train_loss_list_c = None
    kwargs = {}
    if method == "WGAN_GP":
        kwargs["lambda_gp"] = lambda_gp
    train_loss_list_g, train_loss_list_c = fit_adversarial(method, data_loader_train, net, critic, optimizer, scheduler,
                                                           optimizer_c, scheduler_c, epochs, cuda,
                                                           start_epoch_training=0, use_tqdm=use_tqdm,
                                                           critic_steps_every_generator_step=
                                                           critic_steps_every_generator_step,
                                                           train_loss_list_g=train_loss_list_g,
                                                           train_loss_list_c=train_loss_list_c, **kwargs)
else:
    if continue_training_net:
        train_loss_list = np.load(nets_folder + f"train_loss{name_postfix}.npy").tolist()
        val_loss_list = np.load(nets_folder + f"val_loss{name_postfix}.npy").tolist()
    else:
        train_loss_list = val_loss_list = None
    train_loss_list, val_loss_list = fit(data_loader_train, net, loss_fn, optimizer, scheduler, epochs, cuda,
                                         val_loader=data_loader_val, early_stopping=early_stopping,
                                         start_epoch_early_stopping=0 if continue_training_net else start_epoch_early_stopping,
                                         epochs_early_stopping_interval=epochs_early_stopping_interval,
                                         start_epoch_training=0, use_tqdm=use_tqdm, train_loss_list=train_loss_list,
                                         test_loss_list=val_loss_list)
    # compute now the final validation loss achieved by the model; it is repetition from what done before but easier
    # to do this way
    final_validation_loss = test_epoch(data_loader_val, net, loss_fn, cuda)

training_time = time() - start

if save_net_flag:

    save_dict_to_json(args_dict, nets_folder + f'config{name_postfix}.json')

    save_net(nets_folder + f"net{name_postfix}.pth", net)
    if method_is_gan:
        save_net(nets_folder + f"critic{name_postfix}.pth", critic)
        plot_losses(train_loss_list_g, train_loss_list_c, GAN=True)
        plt.savefig(nets_folder + f"losses{name_postfix}.png")
        np.save(nets_folder + f"train_loss_g{name_postfix}.npy", np.array(train_loss_list_g))
        np.save(nets_folder + f"train_loss_c{name_postfix}.npy", np.array(train_loss_list_c))

    else:
        plot_losses(train_loss_list, val_loss_list)
        plt.savefig(nets_folder + f"losses{name_postfix}.png")
        np.save(nets_folder + f"train_loss{name_postfix}.npy", np.array(train_loss_list))
        np.save(nets_folder + f"val_loss{name_postfix}.npy", np.array(val_loss_list))
        text_file = open(nets_folder + f"final_validation_loss{name_postfix}.txt", "w")
        string = f"Final validation loss (with the loss used for training): {final_validation_loss:.2f}"
        text_file.write(string + "\n")
        text_file.close()

    text_file = open(nets_folder + f"training_time{name_postfix}.txt", "w")
    string = "Training time: {:.2f} seconds.".format(training_time)
    text_file.write(string + "\n")
    text_file.close()
