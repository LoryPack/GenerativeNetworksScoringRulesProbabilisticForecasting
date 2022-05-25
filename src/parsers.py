import argparse

import torch

from src.utils import lorenz96_mask, return_raise_not_implemented, lorenz_mask

default_root_folder = "results"

default_model_folder = {"lorenz": "/lorenz/",
                        "lorenz96": "/lorenz96/",
                        "WeatherBench": "/WeatherBench/"}

default_data_size = {"lorenz": 1, "lorenz96": 8,
                     "WeatherBench": torch.Size([1, 32, 64])}

default_patch_size = {"lorenz": 1, "lorenz96": 4,
                      "WeatherBench": 16}

define_masks = {"lorenz": lorenz_mask,
                "lorenz96": lorenz96_mask,
                "WeatherBench": return_raise_not_implemented, }

allowed_srs = ["Energy", "Kernel", "EnergyKernel", "Variogram", "EnergyVariogram", "KernelVariogram"]
allowed_kernels = ["gaussian", "rational_quadratic"]
allowed_methods = ['SR', 'GAN', 'WGAN_GP', 'regression']
allowed_base_measures = ["normal", "laplace", "cauchy"]
allowed_unet_noises = ["sum", "dropout", "concat"]
nonlinearities_dict = {"relu": torch.nn.functional.relu, "tanhshrink": torch.nn.functional.tanhshrink,
                       "leaky_relu": torch.nn.functional.leaky_relu}


def parser_generate_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="The statistical model to consider.", choices=list(default_model_folder.keys()))
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--datasets_folder', type=str, default='datasets')
    parser.add_argument('--n_steps', type=int, default=30000, help='')
    parser.add_argument('--spinup_steps', type=int, default=1000, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--window_size', type=int, default=2, help='')
    parser.add_argument('--not_save_observations', action="store_true", help='')

    return parser


def parser_train_net():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="The statistical model to consider.", choices=list(default_model_folder.keys()))
    parser.add_argument('method', type=str, choices=allowed_methods, help="Which method to use.")
    parser.add_argument('--scoring_rule', type=str, default="Energy", choices=allowed_srs,
                        help="The scoring rule to consider; Energy Score is default. Ignored if method is GAN.")
    parser.add_argument('--kernel', type=str, default="gaussian", choices=allowed_kernels,
                        help="The kernel used in the kernel SR. Ignored if other SRs are used.")
    parser.add_argument('--patched', action="store_true", help="Whether to use a patched SR or not.")
    parser.add_argument('--base_measure', type=str, choices=allowed_base_measures,
                        help="Base measure for the generative network. 'normal' is default.", default="normal")
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default 1000).')
    # parser.add_argument('--no_scheduler', action="store_true", help="Disable scheduler")
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--datasets_folder', type=str, default='datasets')
    parser.add_argument('--weatherbench_data_folder', type=str, default=None, help="Only relevant with WeatherBench.")
    parser.add_argument('--weatherbench_small', action="store_true", help="Whether to use a 16x16 weathebench patch"
                                                                          " rather than the full one.")
    parser.add_argument('--unet_noise_method', type=str, default="sum", choices=allowed_unet_noises,
                        help="Only relevant with WeatherBench and SR or GAN method.")
    parser.add_argument('--unet_large', action="store_true", help="Only relevant with WeatherBench.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_c', type=float, default=1e-3,
                        help='Learning rate for the critic network. Relevant only for GAN')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size (default 1000).')
    parser.add_argument('--no_early_stop', action="store_true")
    parser.add_argument('--epochs_before_early_stopping', type=int, default=100)
    parser.add_argument('--epochs_test_interval', type=int, default=25)
    parser.add_argument('--critic_steps_every_generator_step', type=int, default=1)
    parser.add_argument('--no_tqdm', action="store_true")
    parser.add_argument('--no_save_net', action="store_true")
    parser.add_argument('--continue_training_net_if_available', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--load_all_data_GPU', action="store_true")
    parser.add_argument('--ensemble_size', type=int, default=50,
                        help='Number of generations for each context (default 50)')
    parser.add_argument('--nonlinearity', type=str, default='leaky_relu', choices=list(nonlinearities_dict.keys()))
    parser.add_argument('--data_size', type=int, default=None, help="Size of a single time series instant.")
    parser.add_argument('--auxiliary_var_size', type=int, default=None,
                        help="Size of each realization of the auxiliary data for the generative model approach; "
                             "if None, it uses data_size.")
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Weight for gradient penalty for the WGAN_GP')
    parser.add_argument('--gamma_kernel_score', type=float, default=None,
                        help='The value of bandwidth used in the kernel SR.'
                             'If not provided, it is determined from the observations in the validation window.')
    parser.add_argument('--notrain_if_done_before', action="store_true",
                        help="Do not perform training if the net exists "
                             "before.")
    parser.add_argument('--patch_size', type=int, default=None, help='Patch size for the masks')
    parser.add_argument('--no_RNN', action="store_true", help="Use FCNN in place of RNN for the Lorenz63 and Lorenz96 "
                                                              "models; ignored otherwise.")
    parser.add_argument('--hidden_size_rnn', type=int, default=None, help='Hidden size for the RNN')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight l2 penalization for the optimizer')
    parser.add_argument('--scheduler_gamma', type=float, default=1, help='gamma parameter for scheduler; defaults to '
                                                                         '1 which corresponds to no scheduler used')

    return parser


def parser_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="The statistical model to consider.", choices=list(default_model_folder.keys()))
    # common arguments
    parser = add_parser_arguments_predict_plot(parser)

    parser.add_argument('--plot_start_timestep', type=int, default=100, help='')
    parser.add_argument('--plot_end_timestep', type=int, default=200, help='')
    parser.add_argument('--gamma_kernel_score', type=float, default=None,
                        help='The value of bandwidth used in the kernel SR.'
                             'If not provided, it is determined from the observations in the validation window.')
    parser.add_argument('--gamma_kernel_score_patched', type=float, default=None,
                        help='The value of bandwidth used in the kernel SR in the patched framework.'
                             'If not provided, it is determined from the observations in the validation window.')
    parser.add_argument('--no_RNN', action="store_true", help="Use FCNN in place of RNN for the Lorenz63 and Lorenz96 "
                                                              "models; ignored otherwise.")
    parser.add_argument('--hidden_size_rnn', type=int, default=None, help='Hidden size for the RNN')

    return parser


def parser_plot_weatherbench():
    parser = argparse.ArgumentParser()
    # common arguments
    parser = add_parser_arguments_predict_plot(parser)

    parser.add_argument('--date', type=str, default="2017-08-12", help='Date to consider for the plot. It has to be in '
                                                                       'the format "yyyy-mm-dd".')

    return parser


def add_parser_arguments_predict_plot(parser):
    parser.add_argument('method', type=str, choices=allowed_methods, help="Which method to use.")
    parser.add_argument('--scoring_rule', type=str, default="Energy", choices=allowed_srs,
                        help="The scoring rule to consider; Energy Score is default. Ignored if method is GAN.")
    parser.add_argument('--kernel', type=str, default="gaussian", choices=allowed_kernels,
                        help="The kernel used in the kernel SR for training the NN. Ignored if other SRs are used.")
    parser.add_argument('--patched', action="store_true", help="Whether to network was trained with patched SR or not.")
    parser.add_argument('--base_measure', type=str, choices=allowed_base_measures,
                        help="Base measure for the generative network. 'normal' is default.", default="normal")
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--datasets_folder', type=str, default='datasets')
    parser.add_argument('--weatherbench_data_folder', type=str, default=None, help="Only relevant with WeatherBench.")
    parser.add_argument('--weatherbench_small', action="store_true", help="Whether to use a 16x16 weathebench patch"
                                                                          " rather than the full one.")
    parser.add_argument('--unet_noise_method', type=str, default="sum", choices=allowed_unet_noises,
                        help="Only relevant with WeatherBench and SR or GAN method.")
    parser.add_argument('--unet_large', action="store_true", help="Only relevant with WeatherBench.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_c', type=float, default=1e-3,
                        help='Learning rate for the critic network. Relevant only for GAN')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size (default 1000).')
    parser.add_argument('--no_early_stop', action="store_true")
    parser.add_argument('--critic_steps_every_generator_step', type=int, default=1)
    parser.add_argument('--no_save_plots', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--load_all_data_GPU', action="store_true")
    parser.add_argument('--training_ensemble_size', type=int, default=50,
                        help='Number of generations for each context (default 50) used during training')
    parser.add_argument('--prediction_ensemble_size', type=int, default=200,
                        help='Number of generations for each context (default 100) used for prediction. '
                             'It does not have to be the same as --training_ensemble_size for the Generative model.')
    parser.add_argument('--nonlinearity', type=str, default='leaky_relu', choices=list(nonlinearities_dict.keys()))
    parser.add_argument('--data_size', type=int, default=None, help="Size of a single time series instant.")
    parser.add_argument('--auxiliary_var_size', type=int, default=None,
                        help="Size of each realization of the auxiliary data for the generative model approach; "
                             "if None, it uses data_size.")
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--patch_size', type=int, default=None, help='Patch size for the masks')
    return parser


def obtain_name_folder(model, method, scoring_rule, kernel, patched, patch_size, ensemble_size, auxiliary_var_size,
                       critic_steps_every_generator_step, base_measure, unet_large, method_is_gan, nn_model,
                       hidden_size_rnn):
    if method_is_gan:
        nets_folder_name = method
        if critic_steps_every_generator_step != 1:
            nets_folder_name += f"_critic_steps_{critic_steps_every_generator_step}"
    else:
        nets_folder_name = f"{method}_{scoring_rule}{kernel if scoring_rule == 'Kernel' else ''}_" \
                           f"{ensemble_size}"

    if model != "WeatherBench":
        nets_folder_name += f"_auxdatasize_{auxiliary_var_size}"
    if base_measure != "normal":
        nets_folder_name += "_" + base_measure

    # this overrides the name
    if method == "regression":
        nets_folder_name = method
    if model == "WeatherBench" and unet_large:
        nets_folder_name += "_large_net"
    if nn_model == "rnn":
        nets_folder_name += f"_rnn_{hidden_size_rnn}"
    if not method_is_gan and patched:
        nets_folder_name += f"_patched_{patch_size}"

    return nets_folder_name


def setup(model, root_folder, model_folder, datasets_folder, data_size, method, scoring_rule, kernel, patched,
          patch_size, ensemble_size, auxiliary_var_size, critic_steps_every_generator_step, base_measure, lr, lr_c,
          batch_size, no_early_stop, noise_type, unet_large, nn_model, hidden_size_rnn):
    if root_folder is None:
        root_folder = default_root_folder

    if model_folder is None:
        model_folder = default_model_folder[model]

    if data_size is None:
        data_size = default_data_size[model]

    if auxiliary_var_size is None:
        auxiliary_var_size = data_size

    if hidden_size_rnn is None:
        hidden_size_rnn = data_size

    if patch_size is None:
        patch_size = default_patch_size[model]

    method_is_gan = method in ["GAN", "WGAN_GP"]

    datasets_folder = root_folder + '/' + model_folder + '/' + datasets_folder + '/'

    nets_folder_name = obtain_name_folder(model, method, scoring_rule, kernel, patched, patch_size, ensemble_size,
                                          auxiliary_var_size, critic_steps_every_generator_step, base_measure,
                                          unet_large, method_is_gan, nn_model, hidden_size_rnn)

    nets_folder = root_folder + '/' + model_folder + '/' + nets_folder_name + '/'

    if method_is_gan:
        name_postfix = f"_lr_{lr}_lrc_{lr_c}"
    else:
        name_postfix = f"_lr_{lr}"
    name_postfix += f"_batchsize_{batch_size}{'_noes' if no_early_stop else ''}"

    if model == "WeatherBench":
        name_postfix += f"_{noise_type}"
        if patched and 32 % patch_size != 0:
            raise RuntimeError("patch_size must divide 32.")

    unet_depths = (32, 64, 128, 256) if unet_large else (32, 64, 128)

    return datasets_folder, nets_folder, data_size, auxiliary_var_size, name_postfix, unet_depths, patch_size, method_is_gan, hidden_size_rnn
