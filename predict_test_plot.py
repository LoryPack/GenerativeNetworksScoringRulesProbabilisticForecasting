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
    UNet2D, DiscardWindowSizeDim, get_predictions_and_target, createGenerativeGRUNN, DiscardNumberGenerationsInOutput, \
    createGRUNN, createFCNN
from src.scoring_rules import EnergyScore, KernelScore, VariogramScore, PatchedScoringRule, estimate_score_chunks
from src.utils import load_net, estimate_bandwidth_timeseries, lorenz96_variogram, def_loader_kwargs, \
    weatherbench_variogram_haversine
from src.parsers import parser_predict, define_masks, nonlinearities_dict, setup
from src.calibration import calibration_error, R2, rmse, plot_metrics_params
from src.weatherbench_utils import load_weatherbench_data

# --- parser ---
parser = parser_predict()
args = parser.parse_args()

model = args.model
method = args.method
scoring_rule = args.scoring_rule
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
lr = args.lr
lr_c = args.lr_c
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
no_RNN = args.no_RNN
hidden_size_rnn = args.hidden_size_rnn

save_pdf = True

compute_patched = model in ["lorenz96", ]

model_is_weatherbench = model == "WeatherBench"

nn_model = "unet" if model_is_weatherbench else ("fcnn" if no_RNN else "rnn")

datasets_folder, nets_folder, data_size, auxiliary_var_size, name_postfix, unet_depths, patch_size, method_is_gan, hidden_size_rnn = \
    setup(model, root_folder, model_folder, datasets_folder, data_size, method, scoring_rule, kernel, patched,
          patch_size, training_ensemble_size, auxiliary_var_size, critic_steps_every_generator_step, base_measure, lr,
          lr_c, batch_size, args.no_early_stop, unet_noise_method, unet_large, nn_model, hidden_size_rnn)

model_name_for_plot = {"lorenz": "Lorenz63",
                       "lorenz96": "Lorenz96",
                       "WeatherBench": "WeatherBench"}[model]

string = f"Test {method} network for {model} model"
if not method_is_gan:
    string += f" using {scoring_rule} scoring rule"
print(string)

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

    net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardNumberGenerationsInOutput, net).net
    # wrap by this to discard the window size dimension and the number of simulations in the output
    # net = DiscardNumberGenerationsInOutput(DiscardWindowSizeDim(net))
else:
    wrap_net = True
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
        net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
                       size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                       number_generations_per_forward_call=prediction_ensemble_size, seed=seed + 1)
    else:
        net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardWindowSizeDim, inner_net)

if cuda:
    net.cuda()

# --- predictions ---
# predict all the different elements of the test set and create plots.
# can directly feed through the whole test set for now; if it does not work well then, I will batch it.
with torch.no_grad():
    if model_is_weatherbench:
        # shape (n_val, ensemble_size, lon, lat, n_fields)
        predictions_val, target_data_val = get_predictions_and_target(data_loader_val, net, cuda)
        predictions_test, target_data_test = get_predictions_and_target(data_loader_test, net, cuda)
        # _map is with the original shape. The following instead is flattened:
        predictions_val = predictions_val.flatten(2, -1)
        target_data_val = target_data_val.flatten(1, -1)
        predictions_test = predictions_test.flatten(2, -1)
        target_data_test = target_data_test.flatten(1, -1)
    else:
        predictions_val = net(input_data_val)  # shape (n_val, ensemble_size, data_size)
        predictions_test = net(input_data_test)  # shape (n_test, ensemble_size, data_size)

if method != "regression":
    # --- scoring rules ---
    if compute_patched:
        # mask for patched SRs:
        masks = define_masks[model](data_size=data_size)

    if gamma is None:
        print("Compute gamma...")
        gamma = estimate_bandwidth_timeseries(target_data_val, return_values=["median"])
        print(f"Estimated gamma: {gamma:.4f}")
    if gamma_patched is None and compute_patched:
        # determine the gamma using the first patch only. This assumes that the values of the variables
        # are roughly the same in the different patches.
        gamma_patched = estimate_bandwidth_timeseries(target_data_val[:, masks[0]], return_values=["median"])
        print(f"Estimated gamma patched: {gamma_patched:.4f}")

    # instantiate SRs; each SR takes as input: (net_output, target)
    kernel_gaussian_sr = KernelScore(sigma=gamma)
    kernel_rat_quad_sr = KernelScore(kernel="rational_quadratic", alpha=gamma ** 2)
    energy_sr = EnergyScore()

    variogram = None
    if model in ["lorenz96", ]:
        variogram = lorenz96_variogram(data_size)
    elif model == "WeatherBench":
        # variogram = weatherbench_variogram(weatherbench_small=weatherbench_small)
        variogram = weatherbench_variogram_haversine(weatherbench_small=weatherbench_small)
    if variogram is not None and cuda:
        variogram = variogram.cuda()

    variogram_sr = VariogramScore(variogram=variogram)

    if compute_patched:
        # patched SRs:
        kernel_gaussian_sr_patched = PatchedScoringRule(KernelScore(sigma=gamma_patched), masks)
        kernel_rat_quad_sr_patched = PatchedScoringRule(
            KernelScore(kernel="rational_quadratic", alpha=gamma_patched ** 2),
            masks)
        energy_sr_patched = PatchedScoringRule(energy_sr, masks)

    # -- out of sample score --
    with torch.no_grad():
        string = ""
        for name, predictions, target in zip(["VALIDATION", "TEST"], [predictions_val, predictions_test],
                                             [target_data_val, target_data_test]):
            string += name + "\n"
            kernel_gaussian_score = estimate_score_chunks(kernel_gaussian_sr, predictions, target)
            kernel_rat_quad_score = estimate_score_chunks(kernel_rat_quad_sr, predictions, target)
            energy_score = estimate_score_chunks(energy_sr, predictions, target)
            variogram_score = estimate_score_chunks(variogram_sr, predictions, target, chunk_size=8)

            string += f"Whole data scores: \nEnergy score: {energy_score:.2f}, " \
                      f"Gaussian Kernel score {kernel_gaussian_score:.2f}," \
                      f" Rational quadratic Kernel score {kernel_rat_quad_score:.2f}, " \
                      f"Variogram score {variogram_score:.2f}\n"

            if compute_patched:
                kernel_gaussian_score_patched = estimate_score_chunks(kernel_gaussian_sr_patched, predictions, target)
                kernel_rat_quad_score_patched = estimate_score_chunks(kernel_rat_quad_sr_patched, predictions, target)
                energy_score_patched = estimate_score_chunks(energy_sr_patched, predictions, target)
                string += f"\nPatched data scores: \nEnergy score: {energy_score_patched:.2f}, " \
                          f"Gaussian Kernel score {kernel_gaussian_score_patched:.2f}," \
                          f" Rational quadratic Kernel score {kernel_rat_quad_score_patched:.2f}\n"

        print(string)

with torch.no_grad():
    # -- calibration metrics --
    # target_data_test shape (n_test, data_size)
    # predictions_test shape (n_test, ensemble_size, data_size)
    data_size = predictions_test.shape[-1]
    predictions_for_calibration = predictions_test.transpose(1, 0).cpu().detach().numpy()
    target_data_test_for_calibration = target_data_test.cpu().detach().numpy()
    cal_err_values = calibration_error(predictions_for_calibration, target_data_test_for_calibration)
    rmse_values = rmse(predictions_for_calibration, target_data_test_for_calibration)
    r2_values = R2(predictions_for_calibration, target_data_test_for_calibration)

    string2 = f"Calibration metrics:\n"
    for i in range(data_size):
        string2 += f"x{i}: Cal. error {cal_err_values[i]:.4f}, RMSE {rmse_values[i]:.4f}, R2 {r2_values[i]:.4f}\n"
    string2 += f"\nAverage values: Cal. error {cal_err_values.mean():.4f}, RMSE {rmse_values.mean():.4f}, R2 {r2_values.mean():.4f}\n"
    string2 += f"\nStandard deviation: Cal. error {cal_err_values.std():.4f}, RMSE {rmse_values.std():.4f}, R2 {r2_values.std():.4f}\n\n"

    string2 += f"\nAverage values: Cal. error, RMSE, R2 \n"
    string2 += f"\n\t\t {cal_err_values.mean():.4f} $ \pm$ {cal_err_values.std():.4f} & {rmse_values.mean():.4f}  $ \pm$ {rmse_values.std():.4f} &  {r2_values.mean():.4f} $ \pm$ {r2_values.std():.4f} \\\\ \n"
    string2 += f"\n\t\t {cal_err_values.mean():.4f}  & {rmse_values.mean():.4f}  &  {r2_values.mean():.4f}  \\\\ \n"
    print(string2)

    # -- plots --
with torch.no_grad():
    if model_is_weatherbench:
        # we visualize only the first 8 variables.
        variable_list = np.linspace(0, target_data_test.shape[-1] - 1, 8, dtype=int)
        predictions_test = predictions_test[:, :, variable_list]
        target_data_test = target_data_test[:, variable_list]
        predictions_for_calibration = predictions_for_calibration[:, :, variable_list]
        target_data_test_for_calibration = target_data_test_for_calibration[:, variable_list]

    predictions_test_for_plot = predictions_test.cpu()
    target_data_test_for_plot = target_data_test.cpu()
    time_vec = torch.arange(len(predictions_test)).cpu()
    data_size = predictions_test_for_plot.shape[-1]

    if model == "lorenz":
        var_names = [r"$y$"]
    elif model == "WeatherBench":
        # todo write here the correct lon and lat coordinates!
        var_names = [r"$x_{}$".format(i + 1) for i in range(data_size)]
    else:
        var_names = [r"$x_{}$".format(i + 1) for i in range(data_size)]

    # predictions: mean +- std
    label_size = 13
    if method != "regression":
        predictions_mean = torch.mean(predictions_test_for_plot, dim=1).detach().numpy()
        predictions_std = torch.std(predictions_test_for_plot, dim=1).detach().numpy()

        fig, ax = plt.subplots(nrows=data_size, ncols=1, sharex="col", figsize=(6.4, 3) if data_size == 1 else None)
        if data_size == 1:
            ax = [ax]
        for var in range(data_size):
            ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                         target_data_test_for_plot[plot_start_timestep:plot_end_timestep, var], ls="--",
                         color=f"C{var}")
            ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                         predictions_mean[plot_start_timestep:plot_end_timestep, var], ls="-", color=f"C{var}")
            ax[var].fill_between(
                time_vec[plot_start_timestep:plot_end_timestep], alpha=0.3, color=f"C{var}",
                y1=predictions_mean[plot_start_timestep:plot_end_timestep, var] -
                   predictions_std[plot_start_timestep:plot_end_timestep, var],
                y2=predictions_mean[plot_start_timestep:plot_end_timestep, var] +
                   predictions_std[plot_start_timestep:plot_end_timestep, var])
            ax[var].set_ylabel(var_names[var], size=label_size)

        ax[-1].set_xlabel("Integration time index")
        fig.suptitle(r"Mean $\pm$ std, " + model)
        # plt.show()
        if save_plots:
            plt.savefig(nets_folder + f"prediction{name_postfix}.png")
        plt.close()

    # predictions: median and 99% quantile region
    np_predictions = predictions_test_for_plot.detach().numpy()
    size = 99
    predictions_median = np.median(np_predictions, axis=1)
    if method != "regression":
        predictions_lower = np.percentile(np_predictions, 50 - size / 2, axis=1)
        predictions_upper = np.percentile(np_predictions, 50 + size / 2, axis=1)

    fig, ax = plt.subplots(nrows=data_size, ncols=1, sharex="col", figsize=(6.4, 3) if data_size == 1 else None)
    if data_size == 1:
        ax = [ax]
    for var in range(data_size):
        ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                     target_data_test_for_plot[plot_start_timestep:plot_end_timestep, var], ls="--", color=f"C{var}",
                     label="True")
        ax[var].plot(time_vec[plot_start_timestep:plot_end_timestep],
                     predictions_median[plot_start_timestep:plot_end_timestep, var], ls="-", color=f"C{var}",
                     label="Median forecast" if method != "regression" else "Forecast")
        if method != "regression":
            ax[var].fill_between(
                time_vec[plot_start_timestep:plot_end_timestep], alpha=0.3, color=f"C{var}",
                y1=predictions_lower[plot_start_timestep:plot_end_timestep, var],
                y2=predictions_upper[plot_start_timestep:plot_end_timestep, var], label="99% credible region")
        ax[var].set_ylabel(var_names[var], size=label_size)
        ax[var].tick_params(axis='both', which='major', labelsize=label_size)

    if data_size == 1:
        ax[0].legend(fontsize=label_size)

    ax[-1].set_xlabel(r"$t$", size=label_size)
    # fig.suptitle(f"Median and {size}% credible region, " + model_name_for_plot, size=title_size)
    # plt.show()

    if save_plots:
        # save the metrics in file
        text_file = open(nets_folder + f"test_losses{name_postfix}.txt", "w")
        text_file.write(string + "\n")
        text_file.write(string2 + "\n")
        text_file.close()
        # save the plot:

        if data_size == 1:
            bbox = Bbox(np.array([[0, -0.2], [6.1, 3]]))
        else:
            bbox = Bbox(np.array([[0, -0.2], [6.0, 4.8]]))
        plt.savefig(nets_folder + f"prediction_median{name_postfix}." + ("pdf" if save_pdf else "png"), dpi=400,
                    bbox_inches=bbox)
    plt.close()

    if not model_is_weatherbench:
        # metrics plots
        plot_metrics_params(cal_err_values, rmse_values, r2_values,
                            filename=nets_folder + f"metrics{name_postfix}.png" if save_plots else None)
