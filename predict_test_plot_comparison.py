import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.transforms import Bbox
# import and set up the typeguard
from typeguard.importhook import install_import_hook

# comment these out when deploying:
install_import_hook('src.nn')
install_import_hook('src.scoring_rules')
install_import_hook('src.utils')
install_import_hook('src.parsers')
install_import_hook('src.calibration')
install_import_hook('src.weatherbench_utils')
install_import_hook('src.unet_utils')

from src.nn import ConditionalGenerativeModel, createGenerativeFCNN, InputTargetDataset, \
    UNet2D, DiscardWindowSizeDim, get_predictions_and_target
from src.utils import load_net, def_loader_kwargs
from src.parsers import parser_predict, nonlinearities_dict, setup, default_model_folder, default_root_folder
from src.weatherbench_utils import load_weatherbench_data

# --- parser ---
parser = parser_predict()
args = parser.parse_args()

model = args.model
# method = args.method
# scoring_rule = args.scoring_rule
kernel = args.kernel
patched = args.patched
base_measure = args.base_measure
root_folder = args.root_folder
model_folder = args.model_folder
datasets_folder = args.datasets_folder
weatherbench_data_folder = args.weatherbench_data_folder
weatherbench_small = args.weatherbench_small
unet_noise_method = args.unet_noise_method
unet_large = args.unet_large
# lr = args.lr
# lr_c = args.lr_c
batch_size = args.batch_size
no_early_stop = args.no_early_stop
critic_steps_every_generator_step = args.critic_steps_every_generator_step
save_plots = not args.no_save_plots
cuda = args.cuda
load_all_data_GPU = args.load_all_data_GPU
training_ensemble_size = args.training_ensemble_size
prediction_ensemble_size = args.prediction_ensemble_size
nonlinearity = args.nonlinearity
data_size = args.data_size
auxiliary_var_size = args.auxiliary_var_size
seed = args.seed
plot_start_timestep = args.plot_start_timestep
plot_end_timestep = args.plot_end_timestep
gamma = args.gamma_kernel_score
gamma_patched = args.gamma_kernel_score_patched
patch_size = args.patch_size

save_pdf = True
plot_start_timestep = 0
plot_end_timestep = 30

compute_patched = model in ["lorenz96", ]

if model == "lorenz":
    # define the 3 things which we consider for that
    method1 = "SR"
    scoring_rule1 = "Energy"
    lr1 = 0.01
    lr_c1 = None
    critic_steps_every_generator_step1 = 1

    method2 = "GAN"
    scoring_rule2 = None
    lr2 = 0.03
    lr_c2 = 0.001
    critic_steps_every_generator_step2 = 1

    method3 = "WGAN_GP"
    scoring_rule3 = None
    lr3 = 0.0001
    lr_c3 = 0.1
    critic_steps_every_generator_step3 = 5

    methods_list = ["Energy", "GAN", "WGAN-GP"]

elif model == "lorenz96":
    # define the 3 things which we consider for that
    method1 = "SR"
    scoring_rule1 = "EnergyKernel"
    lr1 = 0.001
    lr_c1 = None
    critic_steps_every_generator_step1 = 1

    method2 = "GAN"
    scoring_rule2 = None
    lr2 = 0.01
    lr_c2 = 0.001
    critic_steps_every_generator_step2 = 1

    method3 = "WGAN_GP"
    scoring_rule3 = None
    lr3 = 0.0001
    lr_c3 = 0.003
    critic_steps_every_generator_step3 = 5

    methods_list = ["Energy-Kernel", "GAN", "WGAN-GP"]
else:
    raise NotImplementedError

datasets_folder, nets_folder1, data_size, auxiliary_var_size, name_postfix1, unet_depths, patch_size, method_is_gan1 = \
    setup(model, root_folder, model_folder, args.datasets_folder, data_size, method1, scoring_rule1, kernel, patched,
          patch_size, training_ensemble_size, auxiliary_var_size, critic_steps_every_generator_step1, base_measure, lr1,
          lr_c1, batch_size, no_early_stop, unet_noise_method, unet_large)

datasets_folder, nets_folder2, data_size, auxiliary_var_size, name_postfix2, unet_depths, patch_size, method_is_gan2 = \
    setup(model, root_folder, model_folder, args.datasets_folder, data_size, method2, scoring_rule2, kernel, patched,
          patch_size, training_ensemble_size, auxiliary_var_size, critic_steps_every_generator_step2, base_measure, lr2,
          lr_c2, batch_size, no_early_stop, unet_noise_method, unet_large)

datasets_folder, nets_folder3, data_size, auxiliary_var_size, name_postfix3, unet_depths, patch_size, method_is_gan3 = \
    setup(model, root_folder, model_folder, args.datasets_folder, data_size, method3, scoring_rule3, kernel, patched,
          patch_size, training_ensemble_size, auxiliary_var_size, critic_steps_every_generator_step3, base_measure, lr3,
          lr_c3, batch_size, no_early_stop, unet_noise_method, unet_large)

model_is_weatherbench = model == "WeatherBench"

nn_model = "unet" if model_is_weatherbench else "fcnn"

# --- data handling ---
if not model_is_weatherbench:
    input_data_test = torch.load(datasets_folder + "test_x.pty")
    target_data_test = torch.load(datasets_folder + "test_y.pty")
    input_data_val = torch.load(datasets_folder + "val_x.pty")
    target_data_val = torch.load(datasets_folder + "val_y.pty")

    window_size = input_data_test.shape[1]

    # create the test loaders; these are unused for the moment.
    dataset_val = InputTargetDataset(input_data_val, target_data_val, "cuda" if cuda and load_all_data_GPU else "cpu")
    dataset_test = InputTargetDataset(input_data_test, target_data_test,
                                      "cuda" if cuda and load_all_data_GPU else "cpu")
else:
    print("Load weatherbench dataset...")
    dataset_train, dataset_val, dataset_test = load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU,
                                                                      return_test=True,
                                                                      weatherbench_small=weatherbench_small)
    print("Loaded")
    print("Validation set size:", len(dataset_val))
    print("Test set size:", len(dataset_test))

loader_kwargs = def_loader_kwargs(cuda, load_all_data_GPU)

# loader_kwargs.update(loader_kwargs_2)  # if you want to add other loader arguments

data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **loader_kwargs)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **loader_kwargs)

# --- networks ---
nets_list = []
nets_folder_list = [nets_folder1, nets_folder2, nets_folder3]
name_postfix_list = [name_postfix1, name_postfix2, name_postfix3]
for i in range(3):
    wrap_net = True
    # create generative net:
    if nn_model == "fcnn":
        input_size = window_size * data_size + auxiliary_var_size
        output_size = data_size
        hidden_sizes_list = [int(input_size * 1.5), int(input_size * 3), int(input_size * 3),
                             int(input_size * 0.75 + output_size * 3), int(output_size * 5)]
        inner_net = createGenerativeFCNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes_list,
                                         nonlinearity=nonlinearities_dict[nonlinearity])()
    elif nn_model == "unet":
        # select the noise method here:
        inner_net = UNet2D(in_channels=data_size[0], out_channels=1, noise_method=unet_noise_method,
                           number_generations_per_forward_call=prediction_ensemble_size, conv_depths=unet_depths)
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
        net = load_net(nets_folder_list[i] + f"net{name_postfix_list[i]}.pth", ConditionalGenerativeModel, inner_net,
                       size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                       number_generations_per_forward_call=prediction_ensemble_size, seed=seed + 1)
        nets_list.append(net)
    else:
        net = load_net(nets_folder_list[i] + f"net{name_postfix_list[i]}.pth", DiscardWindowSizeDim, inner_net)
        nets_list.append(net)

    if cuda:
        net.cuda()

# --- predictions ---
# predict all the different elements of the test set and create plots.
# can directly feed through the whole test set for now; if it does not work well then, I will batch it.
predictions_val_list = []
predictions_test_list = []
for i in range(3):
    with torch.no_grad():
        if model_is_weatherbench:
            # shape (n_val, ensemble_size, lon, lat, n_fields)
            predictions_val, target_data_val = get_predictions_and_target(data_loader_val, nets_list[i], cuda)
            predictions_test, target_data_test = get_predictions_and_target(data_loader_test, nets_list[i], cuda)
            # _map is with the original shape. The following instead is flattened:
            predictions_val = predictions_val.flatten(2, -1)
            target_data_val = target_data_val.flatten(1, -1)
            predictions_test = predictions_test.flatten(2, -1)
            target_data_test = target_data_test.flatten(1, -1)
        else:
            predictions_val = nets_list[i](input_data_val)  # shape (n_val, ensemble_size, data_size)
            predictions_test = nets_list[i](input_data_test)  # shape (n_test, ensemble_size, data_size)
    predictions_val_list.append(predictions_val.cpu().detach().numpy())
    predictions_test_list.append(predictions_test.cpu().detach().numpy())

# -- plots --
with torch.no_grad():
    if model_is_weatherbench:
        # we visualize only the first 8 variables.
        variable_list = np.linspace(0, target_data_test.shape[-1] - 1, 8, dtype=int)
        predictions_test = predictions_test[:, :, variable_list]
        target_data_test = target_data_test[:, variable_list]

    target_data_test_for_plot = target_data_test.cpu()
    time_vec = torch.arange(len(predictions_test)).cpu()
    data_size = 1

    if model == "lorenz":
        var_name = r"$y$"
    elif model == "WeatherBench":
        # todo write here the correct lon and lat coordinates!
        var_name = r"$x_{}$".format(1)
    else:
        var_name = r"$x_{}$".format(1)

    # predictions: median and 99% quantile region
    fig, ax = plt.subplots(nrows=data_size, ncols=1, sharex="col", figsize=(6, 4.5) if data_size == 1 else None)
    label_size = 13

    # add the target values:
    ax.plot(time_vec[plot_start_timestep:plot_end_timestep],
            target_data_test_for_plot[plot_start_timestep:plot_end_timestep, 0], ls="--", color="black",
            label="True")

    size = 99
    for i in range(3):
        predictions_median = np.median(predictions_test_list[i], axis=1)
        predictions_lower = np.percentile(predictions_test_list[i], 50 - size / 2, axis=1)
        predictions_upper = np.percentile(predictions_test_list[i], 50 + size / 2, axis=1)

        ax.plot(time_vec[plot_start_timestep:plot_end_timestep],
                predictions_median[plot_start_timestep:plot_end_timestep, 0], ls="-", color=f"C{i}",
                label=methods_list[i], alpha=0.6)
        ax.fill_between(
            time_vec[plot_start_timestep:plot_end_timestep], alpha=0.2, color=f"C{i}",
            y1=predictions_lower[plot_start_timestep:plot_end_timestep, 0],
            y2=predictions_upper[plot_start_timestep:plot_end_timestep, 0])
        ax.set_ylabel(var_name, size=label_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)

    ax.legend(fontsize=label_size)

    ax.set_xlabel(r"$t$", size=label_size)
    # fig.suptitle(f"Median and {size}% credible region, " + model_name_for_plot, size=title_size)
    # plt.show()

    if save_plots:
        if root_folder is None:
            root_folder = default_root_folder
        if model_folder is None:
            model_folder = root_folder + '/' + default_model_folder[model]
        bbox = Bbox(np.array([[0, 0], [5.8, 4]]))
        plt.savefig(model_folder + "prediction_median_comparison." + ("pdf" if save_pdf else "png"), dpi=400,
                    bbox_inches=bbox)
    plt.close()
