try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    pass
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchtyping import TensorType, patch_typeguard

patch_typeguard()  # use before @typechecked


# The next class is adapted from the original WeatherBench code


class WeatherBenchDataset(Dataset):
    def __init__(self, ds, var_dict, lead_time, observation_window=1, daily=True, load=True, cuda=False, mean=None,
                 std=None, small_patch=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: xarray Dataset containing all variables.
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time; in days if daily=True, in hours otherwise.
                It is the time from the start of the observation window to the prediction.
            observation_window: Length of the observation window, in number of time frames.
                It is the number of frames used to predict at lead time. Setting observation_window=1 uses
                a single frame. Frames are spaced by one day if daily=True and by one hour otherwise.
            load: bool. If True, dataset is loaded into RAM.
            cuda: bool. If True, the full dataset is moved to the GPU if it was loaded. That may reduce training time by
                reducing data transfer, but may not work if the dataset is too large to fit in GPU memory.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """

        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.observation_window = observation_window
        self.lead_plus_observation_minus_1 = lead_time + observation_window - 1
        self.load = load

        data = []
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except (KeyError, ValueError):  # I fixed this from ValueError
                generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        if daily:
            self.data = self.data.isel(time=(self.data.time.dt.hour == 12))
        if small_patch is not None:
            self.data = self.data.isel(lon=slice(0, small_patch)).isel(lat=slice(0, small_patch))
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -self.lead_plus_observation_minus_1)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -self.lead_plus_observation_minus_1)).time
        self.valid_time = self.data.isel(time=slice(self.lead_plus_observation_minus_1, None)).time

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load:
            print('Loading data into RAM')
            self.data.load()
            self.data_torch = torch.from_numpy(
                self.data.values.astype("float32"))  # convert to torch if they were loaded.
            print("CPU memory used (MB)", self.data_torch.element_size() * self.data_torch.nelement() // 1024 ** 2)
            # move to the correct device if they were loaded:
            if cuda:
                print('Loading data into GPU')
                self.data_torch = self.data_torch.cuda()
                print("GPU memory used (MB)", self.data_torch.element_size() * self.data_torch.nelement() // 1024 ** 2)

    def __len__(self) -> int:
        """Total number of samples in the dataset"""
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[
        TensorType["window_size", "lat", "lon", "n_fields"], TensorType["lat", "lon", "n_fields"]]:
        """Generate one single data element"""
        # here we return the correct elements, converted to torch
        if self.load:
            # we have already transformed to torch.
            context = self.data_torch[index:index + self.observation_window]
            target = self.data_torch[index + self.lead_plus_observation_minus_1]
        else:
            context = self.data.isel(time=slice(index, index + self.observation_window)).values
            target = self.data.isel(time=index + self.lead_plus_observation_minus_1).values
            context = torch.from_numpy(context.astype("float32"))
            target = torch.from_numpy(target.astype("float32"))
        return context, target

    def select_time(self, timestring):
        """Returns the context and target at a given timestring. The context is returned as torch (to be input in a
        net), while the target is returned as a xarray.DataArray"""
        where_result = np.where(self.data.time == np.datetime64(timestring))
        if len(where_result[0]) == 0:
            raise RuntimeError("No data corresponding to that timestring.")
        target = self.data.sel(time=timestring)
        # find the corresponding index for that timestring
        index = np.where(self.data.time == np.datetime64(timestring))[0][0] - self.lead_plus_observation_minus_1
        print("corresponding index", index)
        if index < 0:
            raise RuntimeError(
                "You want an observation target which is not available with the "
                "considered observation window and forecast lead time")
        if self.load:
            context = self.data_torch[index:index + self.observation_window]
        else:
            context = self.data.isel(time=slice(index, index + self.observation_window)).values
            context = torch.from_numpy(context.astype("float32"))
        return context, target


def load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU, return_test=False,
                           weatherbench_small=False):
    window_size = 1
    lead_time = 3 - window_size + 1  # 3 days
    # geopotential height:
    data = xr.open_mfdataset(weatherbench_data_folder + '/geopotential_500/*.nc', combine='by_coords')
    var_dict = {'z': None}
    # temperature:
    # data = xr.open_mfdataset(weatherbench_data_folder + 'temperature_850/*.nc', combine='by_coords')
    # var_dict = {'t': None}
    small_patch = 16 if weatherbench_small else None
    dataset_train = WeatherBenchDataset(data.sel(time=slice('1979', '2006')), var_dict, lead_time, window_size,
                                        load=True, cuda=cuda and load_all_data_GPU, daily=True, small_patch=small_patch)
    # validation and test sets use the training mean and std
    dataset_val = WeatherBenchDataset(data.sel(time=slice('2007', '2016')), var_dict, lead_time, window_size,
                                      load=True, cuda=cuda and load_all_data_GPU, mean=dataset_train.mean,
                                      std=dataset_train.std, daily=True, small_patch=small_patch)
    return_list = [dataset_train, dataset_val]
    if return_test:
        dataset_test = WeatherBenchDataset(data.sel(time=slice('2017', '2018')), var_dict, lead_time, window_size,
                                           load=True, cuda=cuda and load_all_data_GPU, mean=dataset_train.mean,
                                           std=dataset_train.std, daily=True, small_patch=small_patch)
        return_list += [dataset_test]
    return return_list


def convert_tensor_to_da(tensor, reference_da):
    tensor_numpy = tensor.numpy()
    return reference_da.copy(data=tensor_numpy)


def single_map_plot(da, title=None, **kwargs):
    p = da.plot(
        subplot_kws=dict(projection=ccrs.PlateCarree(), facecolor="gray"),
        transform=ccrs.PlateCarree(), **kwargs)

    p.axes.set_global()
    p.axes.coastlines()
    p.axes.gridlines()

    if title is not None:
        p.axes.set_title(title)
    return p


def plot_map_ax(da, title=None, ax=None, global_projection=True, **kwargs):
    p = xr.plot.pcolormesh(da, transform=ccrs.PlateCarree(), ax=ax, add_colorbar=False, **kwargs)

    if global_projection:
        p.axes.set_global()
    p.axes.coastlines()
    p.axes.gridlines()

    if title is not None:
        p.axes.set_title(title)
    return p
