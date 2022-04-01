import matplotlib.pyplot as plt
import numpy as np
import torch
# import and set up the typeguard
from typeguard.importhook import install_import_hook

# comment these out when deploying:
install_import_hook('src.models')
install_import_hook('src.parsers')

from src.models import run_lorenz96_truth
from src.parsers import parser_generate_data, default_root_folder, default_model_folder

parser = parser_generate_data()
args = parser.parse_args()

model = args.model
root_folder = args.root_folder
model_folder = args.model_folder
datasets_folder = args.datasets_folder
n_steps = args.n_steps
spinup_steps = args.spinup_steps
seed = args.seed
window_size = args.window_size
save_observations = not args.not_save_observations

if root_folder is None:
    root_folder = default_root_folder

if model_folder is None:
    model_folder = default_model_folder[model]

datasets_folder = root_folder + '/' + model_folder + '/' + args.datasets_folder + '/'
rng = np.random.RandomState(seed)

print(f"Generate data for {model} model.")

timeseries_stacked = None

if model == "lorenz":

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        """
        Given:
           x, y, z: a point of interest in three dimensional space
           s, r, b: parameters defining the lorenz attractor
        Returns:
           x_dot, y_dot, z_dot: values of the lorenz attractor's partial
               derivatives at the point x, y, z
        """
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot


    discard_interval = 30
    integration_steps = (n_steps + spinup_steps) * discard_interval  # then keep one every 30 for the dataset

    dt = 0.01

    # Need one more for the initial values
    xs = np.empty(integration_steps + 1)
    ys = np.empty(integration_steps + 1)
    zs = np.empty(integration_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point. That is simple Euler integration
    for i in range(integration_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    timeseries = ys[::discard_interval][spinup_steps:].reshape(-1, 1)

elif model == "lorenz96":
    # full Lorenz96; we observe the x's only.

    dt_observation = 0.2
    dt_integration = 0.001
    discard_interval = int(dt_observation / dt_integration)
    K = 8  # number of observed variables
    J = 32  # number of unobserved variable for each observed one
    X_init = np.zeros(K)
    Y_init = np.zeros(J * K)
    X_init[0] = 1
    Y_init[0] = 1
    h = 1
    b = 10.0
    c = 10.0
    F = 20.0

    burnin_steps = int(2 / dt_integration)  # discard two time units of burn-in
    total_integration_steps = n_steps * discard_interval + burnin_steps

    timeseries, Y_out, times, steps = run_lorenz96_truth(X_init, Y_init, dt_integration, total_integration_steps,
                                                         burn_in=burnin_steps, skip=discard_interval, h=h, F=F, b=b,
                                                         c=c)

print("Timeseries shape", timeseries.shape)
len_timeseries, n_vars = timeseries.shape
if n_vars == 1:
    plt.plot(timeseries, alpha=0.5)
    plt.savefig(datasets_folder + "timeseries.pdf")
else:
    plt.figure(figsize=(8, 10))
    plt.pcolormesh(np.arange(n_vars + 1), np.arange(len_timeseries + 1), timeseries, cmap="RdBu_r")
    plt.title("Timeseries output")
    plt.colorbar()
    plt.xlabel("Variable")
    plt.ylabel("Timestep")
    plt.savefig(datasets_folder + "timeseries.pdf")

# now transform the data as needed.
# the one below is a sliding window array with shape (n_windows, window_size + 1) array.
if timeseries_stacked is None:  # if that is not created above (as for the Lorenz96)
    timeseries_stacked = np.lib.stride_tricks.sliding_window_view(timeseries, window_shape=window_size + 1,
                                                                  axis=0).transpose(0, 2, 1)

# Then, we can split in input, target as follows:
input_data = timeseries_stacked[:, 0:-1]
target_data = timeseries_stacked[:, -1]
n_samples = input_data.shape[0]
print("Total number of samples:", n_samples)

# train-test split: take 70/30, shuffled
# indices_test = rng.choice(np.arange(n_samples), int(0.3 * n_samples), replace=False)
#
# input_data_train = np.delete(input_data, indices_test, axis=0)
# target_data_train = np.delete(target_data, indices_test, axis=0)
# input_data_test = input_data[indices_test]
# target_data_test = target_data[indices_test]

# train/validate/test (60-20-20) split, in order (not shuffling):
n_train = int(0.6 * n_samples)
n_val = int(0.2 * n_samples)
n_test = n_samples - n_val - n_train
input_data_train = input_data[0:n_train]
target_data_train = target_data[0:n_train]
input_data_val = input_data[n_train:n_train + n_val]
target_data_val = target_data[n_train:n_train + n_val]
input_data_test = input_data[n_train + n_val:n_samples]
target_data_test = target_data[n_train + n_val:n_samples]

# convert to torch:
input_data_train = torch.from_numpy(np.float32(input_data_train))
target_data_train = torch.from_numpy(np.float32(target_data_train))
input_data_val = torch.from_numpy(np.float32(input_data_val))
target_data_val = torch.from_numpy(np.float32(target_data_val))
input_data_test = torch.from_numpy(np.float32(input_data_test))
target_data_test = torch.from_numpy(np.float32(target_data_test))

if save_observations:
    torch.save(input_data_train, datasets_folder + "train_x.pty")
    torch.save(target_data_train, datasets_folder + "train_y.pty")
    torch.save(input_data_val, datasets_folder + "val_x.pty")
    torch.save(target_data_val, datasets_folder + "val_y.pty")
    torch.save(input_data_test, datasets_folder + "test_x.pty")
    torch.save(target_data_test, datasets_folder + "test_y.pty")
