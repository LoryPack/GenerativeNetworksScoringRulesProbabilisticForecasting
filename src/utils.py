import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import haversine_distances
from torchtyping import TensorType, patch_typeguard

patch_typeguard()  # use before @typechecked


def plot_losses(losses_1, losses_2, GAN=False):
    """
    Plot losses vs training epochs after the NN have been trained.

    Parameters
    ----------


    Returns
    -------

    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(np.arange(len(losses_1)) + 1, losses_1, color="C0",
            label="Generator train loss" if GAN else "Train loss")
    if len(losses_2) != len(losses_1):
        raise RuntimeError("Length of train and test losses list should be the same.")
    ax.plot(np.arange(len(losses_1)) + 1, losses_2, color="C1",
            label="Critic train loss" if GAN else "Test loss")

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    return fig, ax


def save_net(path, net):
    """Function to save the Pytorch state_dict of a network to a file."""
    torch.save(net.state_dict(), path)


def load_net(path, network_class, *network_args, **network_kwargs):
    """Function to load a network from a Pytorch state_dict, given the corresponding network_class."""
    net = network_class(*network_args, **network_kwargs)
    net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return net.eval()  # call the network to eval model. Needed with batch normalization and dropout layers.


def save_dict_to_json(dictionary, file):
    with open(file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def estimate_bandwidth_timeseries(timeseries: TensorType["batch", "data_size"], return_values=["median"]):
    """Estimate the bandwidth for the gaussian kernel in KernelSR. """

    timesteps, data_size = timeseries.shape
    distances = torch.cdist(timeseries.reshape(timesteps, data_size), timeseries.reshape(timesteps, data_size))
    # discard the diagonal elements, as they are 0:
    distances = distances[~torch.eye(timesteps, dtype=bool)].flatten()

    return_list = []
    if "median" in return_values:
        return_list.append(torch.median(distances))
    if "mean" in return_values:
        return_list.append(torch.mean(distances))

    return return_list[0] if len(return_list) == 1 else return_list


def weight_for_summed_score(scoring_rule, model_is_weatherbench, patch_size=16):
    if scoring_rule == "EnergyVariogram":
        if model_is_weatherbench:
            weight_list = (1, 1 / (4800.0 * 30))
        else:
            weight_list = (1, 1 / 10.0)
    elif scoring_rule == "KernelVariogram":
        if model_is_weatherbench:
            weight_list = (1, 1 / (250000.0 * 30))
        else:
            weight_list = (1, 1 / 10.0)
    elif scoring_rule == "EnergyKernel":
        if model_is_weatherbench:
            weight_list = (1 / 70.0, 1)
        else:
            weight_list = (1, 1)
    elif scoring_rule == "Energy":
        if model_is_weatherbench:
            # this supposes to be in the case of patched SR, the first element being the patched one and the second
            # the regular one.
            # With patch_size = 16 (and patch_step=8), you get 32 patches. As the EnergyScore scales
            # as d, each of the 16x16=256 patches has relative magnitude 256/(32x64=2048)=0.125. However there are 32 of
            # them, so that the magnitude of the patched part is 4 times the overall one.
            #
            # With patch_size = 8 (and patch_step=4), you get 128 patches. As the EnergyScore scales
            # as d, each of the 8x8=64 patches has relative magnitude 64/(32x64=2048)=0.03125. However there are 128 of
            # them, so that the magnitude of the patched part is 4 times the overall one.
            #
            # In both cases, I leave therefore them with equal weights:
            weight_list = (1, 1)
        else:
            raise NotImplementedError
    return weight_list


# variograms
def lorenz96_variogram(data_size=8):
    # implement the variogram matrix for Lorenz96, inversely proportional to the distance in the circle
    variogram = torch.zeros(data_size, data_size)
    for j in range(data_size):
        for i in range(j + 1, data_size):
            variogram[i, j] = variogram[j, i] = 1 / (torch.min(torch.abs((i - j) * torch.ones(1)),
                                                               torch.abs((j + 8 - i) * torch.ones(1))))
    return variogram


def weatherbench_variogram(weatherbench_small=False) -> TensorType["data_size", "data_size"]:
    # implement the variogram matrix for weatherbench, inversely proportional to the grid distance. Consider that
    # longitudinal direction is a loop. In principle, the different grid points are not exactly at the same distance.
    # Discard that for now.
    n_lat = 16 if weatherbench_small else 32
    n_lon = 16 if weatherbench_small else 64

    # distance_lat = torch.zeros(n_lat, n_lat)
    # for j in range(n_lat):
    #     for i in range(j + 1, n_lat):
    #         distance_lat[i, j] = distance_lat[j, i] = torch.abs((i - j) * torch.ones(1))

    distance_lat = torch.abs(torch.arange(n_lat).reshape(1, -1) - torch.arange(n_lat).reshape(-1, 1) * torch.ones(1))

    # could optimize the one below as well but improvement is small.
    distance_lon = torch.zeros(n_lon, n_lon)
    for j in range(n_lon):
        for i in range(j + 1, n_lon):
            distance_lon[i, j] = distance_lon[j, i] = (torch.min(torch.abs((i - j) * torch.ones(1)),
                                                                 torch.abs((j + n_lon - i) * torch.ones(1))))

    # assert torch.allclose(distance_lat_2, distance_lat)

    distance_lat_squared = distance_lat * distance_lat
    distance_lon_squared = distance_lon * distance_lon

    # very inefficient loop
    # distance = torch.zeros(n_lat, n_lon, n_lat, n_lon)
    # for j in range(n_lat):
    #     for i in range(n_lat):
    #         for k in range(n_lon):
    #             for l in range(n_lon):
    #                 distance[i, k, j, l] = distance_lat_squared[i, j] + distance_lon_squared[k, l]

    # smarter way:
    distance_lat_squared = distance_lat_squared.reshape(n_lat, 1, n_lat, 1)
    distance_lon_squared = distance_lon_squared.reshape(1, n_lon, 1, n_lon)

    distance = distance_lat_squared + distance_lon_squared

    distance = torch.sqrt(distance)

    variogram = 1 / distance
    variogram = variogram.flatten(2, 3).flatten(0, 1)

    # there are some infinity values here, on the diagonal. Remove them
    for i in range(n_lon * n_lat):
        variogram[i, i] = 0

    return variogram  # , distance.flatten(2, 3).flatten(0, 1)


def weatherbench_variogram_haversine(weatherbench_small=False) -> TensorType["data_size", "data_size"]:
    # implements the variogram matrix for weatherbench, inversely proportional to the geophysical distance (in km),
    # computed from latitude and longitude using the Haversine formula, implemented in sklearn.
    if weatherbench_small:
        raise NotImplementedError

    n_lat = 32
    n_lon = 64

    lat = np.array([-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625, -53.4375,
                    -47.8125, -42.1875, -36.5625, -30.9375, -25.3125, -19.6875, -14.0625,
                    -8.4375, -2.8125, 2.8125, 8.4375, 14.0625, 19.6875, 25.3125,
                    30.9375, 36.5625, 42.1875, 47.8125, 53.4375, 59.0625, 64.6875,
                    70.3125, 75.9375, 81.5625, 87.1875])

    lon = np.array([0., 5.625, 11.25, 16.875, 22.5, 28.125, 33.75, 39.375,
                    45., 50.625, 56.25, 61.875, 67.5, 73.125, 78.75, 84.375,
                    90., 95.625, 101.25, 106.875, 112.5, 118.125, 123.75, 129.375,
                    135., 140.625, 146.25, 151.875, 157.5, 163.125, 168.75, 174.375,
                    180., 185.625, 191.25, 196.875, 202.5, 208.125, 213.75, 219.375,
                    225., 230.625, 236.25, 241.875, 247.5, 253.125, 258.75, 264.375,
                    270., 275.625, 281.25, 286.875, 292.5, 298.125, 303.75, 309.375,
                    315., 320.625, 326.25, 331.875, 337.5, 343.125, 348.75, 354.375])

    X = np.meshgrid(lon, lat)  # create the mesh from the above two
    lat_lon_vector = np.stack((X[1], X[0]), axis=2)  # 32x64x2
    lat_lon_vector = lat_lon_vector.reshape(-1, 2)

    # the above is in degree, convert in radiants
    lat_lon_vector = np.deg2rad(lat_lon_vector)

    distance_angular = haversine_distances(lat_lon_vector)
    # distance_km = distance_angular * 6371  # multiply by Earth radius to get kilometers

    variogram = 1 / distance_angular

    # there are some infinity values here, on the diagonal. Remove them
    for i in range(n_lon * n_lat):
        variogram[i, i] = 0

    return torch.from_numpy(variogram.astype(np.float32))  # , distance_angular


# masks
def lorenz96_mask(data_size=8, patch_size=4) -> TensorType["n_patches", "data_size", bool]:
    # for the case data_size=8, we define 4 patches: 0-3, 2-5, 4-7, 6-1
    if data_size != 8:
        raise NotImplementedError  # for now

    if patch_size == 1:
        masks = torch.eye(8, dtype=bool)
    elif patch_size == 2:
        n_patches = 8
        masks = torch.zeros(n_patches, data_size, dtype=bool)
        for i in range(7):
            masks[i, i:i + 2] = True
        masks[7, 7] = masks[7, 0] = True
    elif patch_size in [3, 4]:
        n_patches = 4
        masks = torch.zeros(n_patches, data_size, dtype=bool)
        masks[0, 0:0 + patch_size] = True
        masks[1, 2:2 + patch_size] = True
        masks[2, 4:4 + patch_size] = True
        masks[3, 6:8] = masks[3, 0:+patch_size - 2] = True
    else:
        raise NotImplementedError

    return masks


def lorenz_mask(data_size=1) -> TensorType["n_patches", "data_size", bool]:
    # for the case data_size=8, we define 4 patches: 0-3, 2-5, 4-7, 6-1
    if data_size != 1:
        raise NotImplementedError  # for now

    masks = torch.ones(1, 1, dtype=bool)

    return masks


def return_raise_not_implemented(*args, **kwargs):
    raise NotImplementedError


def def_loader_kwargs(cuda, load_all_data_GPU):
    if cuda:
        if load_all_data_GPU:
            loader_kwargs = {'num_workers': 0, 'pin_memory': False}
        else:
            loader_kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        loader_kwargs = {}
    return loader_kwargs
