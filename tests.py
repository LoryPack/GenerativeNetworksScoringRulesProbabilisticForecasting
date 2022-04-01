import unittest

import numpy as np
import torch
import xarray as xr
# import and set up the typeguard
from typeguard.importhook import install_import_hook

install_import_hook('src.nn')
install_import_hook('src.scoring_rules')
install_import_hook('src.utils')
install_import_hook('src.weatherbench_utils')
install_import_hook('src.unet_utils')

from src.scoring_rules import EnergyScore, KernelScore, \
    VariogramScore, SumScoringRules, PatchedScoringRule
from src.nn import createFCNN, ConditionalGenerativeModel, createGenerativeFCNN, LayerNormMine
from src.utils import estimate_bandwidth_timeseries
from src.parsers import allowed_base_measures
from src.weatherbench_utils import WeatherBenchDataset


class EnergyScoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(2)
        self.forecast = self.rng.randn(2, 5, 3).astype("float32")
        self.verification = self.rng.randn(2, 3).astype("float32")
        self.forecast_torch = torch.from_numpy(self.forecast)
        self.verification_torch = torch.from_numpy(self.verification)
        self.sr = EnergyScore(beta=1.7)
        self.sr_no_mean = EnergyScore(beta=1.7, mean=False)

    def test_numpy_torch_match(self):
        # you can test their accordance only in case of 1 single observation (ie batch element) due to the
        # different way they are computed

        numpy_value = self.sr.score(self.verification[0].reshape(1, -1), self.forecast[0])
        torch_value = self.sr.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                   self.verification_torch[0].reshape(1, -1))
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

    def test_additive_batch_torch(self):
        score_1 = self.sr.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                               self.verification_torch[0].reshape(1, -1))
        score_2 = self.sr.estimate_score_batch(self.forecast_torch[1].reshape(1, 5, 3),
                                               self.verification_torch[1].reshape(1, -1))
        score_joint = self.sr.estimate_score_batch(self.forecast_torch, self.verification_torch)

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

    def test_mean(self):
        score_mean = self.sr.estimate_score_batch(self.forecast_torch, self.verification_torch)
        score_no_mean = self.sr_no_mean.estimate_score_batch(self.forecast_torch, self.verification_torch)
        self.assertTrue(torch.allclose(score_mean * 2, score_no_mean))


class KernelScoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.forecast = self.rng.randn(2, 5, 3).astype("float32")
        self.verification = self.rng.randn(2, 3).astype("float32")
        self.forecast_torch = torch.from_numpy(self.forecast)
        self.verification_torch = torch.from_numpy(self.verification)
        self.sr_unbiased_numpy_gaussian = KernelScore(torch=False, sigma=1.5)
        self.sr_biased_numpy_gaussian = KernelScore(biased_estimator=True, torch=False, sigma=1.5)
        self.sr_unbiased_torch_gaussian = KernelScore(sigma=1.5)
        self.sr_biased_torch_gaussian = KernelScore(biased_estimator=True, sigma=1.5)
        self.sr_unbiased_numpy_rational_quadratic = KernelScore(torch=False, kernel="rational_quadratic", alpha=0.3)
        self.sr_biased_numpy_rational_quadratic = KernelScore(biased_estimator=True, torch=False,
                                                              kernel="rational_quadratic", alpha=0.3)
        self.sr_unbiased_torch_rational_quadratic = KernelScore(kernel="rational_quadratic", alpha=0.3)
        self.sr_biased_torch_rational_quadratic = KernelScore(biased_estimator=True, kernel="rational_quadratic",
                                                              alpha=0.3)
        self.sr_unbiased_torch_gaussian_no_mean = KernelScore(sigma=1.5, mean=False)

        def def_negative_Euclidean_distance(beta=1.0):
            if beta <= 0 or beta > 2:
                raise RuntimeError("'beta' not in the right range (0,2]")

            if beta == 1:
                def Euclidean_distance(x, y):
                    return - torch.norm(x - y)
            else:
                def Euclidean_distance(x, y):
                    return - torch.norm(x - y) ** beta

            return Euclidean_distance

        self.sr_unbiased_torch_kernel_energy = KernelScore(kernel=def_negative_Euclidean_distance(beta=1.4))
        self.sr_energy_torch = EnergyScore(beta=1.4)

    def test_numpy_torch_match(self):
        # you can test their accordance only in case of 1 single observation (ie batch element) due to the
        # different way they are computed

        # unbiased:
        numpy_value = self.sr_unbiased_numpy_gaussian.score(self.verification[0].reshape(1, -1), self.forecast[0])
        torch_value = self.sr_unbiased_torch_gaussian.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                                           self.verification_torch[0].reshape(1, -1))
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

        # biased:
        numpy_value = self.sr_biased_numpy_gaussian.score(self.verification[0].reshape(1, -1), self.forecast[0])
        torch_value = self.sr_biased_torch_gaussian.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                                         self.verification_torch[0].reshape(1,
                                                                                                            -1))
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

        # unbiased:
        numpy_value = self.sr_unbiased_numpy_rational_quadratic.score(self.verification[0].reshape(1, -1),
                                                                      self.forecast[0])
        torch_value = self.sr_unbiased_torch_rational_quadratic.estimate_score_batch(
            self.forecast_torch[0].reshape(1, 5, 3), self.verification_torch[0].reshape(1, -1))
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

        # biased:
        numpy_value = self.sr_biased_numpy_rational_quadratic.score(self.verification[0].reshape(1, -1),
                                                                    self.forecast[0])
        torch_value = self.sr_biased_torch_rational_quadratic.estimate_score_batch(
            self.forecast_torch[0].reshape(1, 5, 3), self.verification_torch[0].reshape(1, -1))
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

    def test_additive_batch_torch(self):
        # unbiased:
        score_1 = self.sr_unbiased_torch_gaussian.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                                       self.verification_torch[0].reshape(1, -1))
        score_2 = self.sr_unbiased_torch_gaussian.estimate_score_batch(self.forecast_torch[1].reshape(1, 5, 3),
                                                                       self.verification_torch[1].reshape(1, -1))
        score_joint = self.sr_unbiased_torch_gaussian.estimate_score_batch(self.forecast_torch, self.verification_torch)

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

        # biased:
        score_1 = self.sr_biased_torch_gaussian.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                                     self.verification_torch[0].reshape(1, -1))
        score_2 = self.sr_biased_torch_gaussian.estimate_score_batch(self.forecast_torch[1].reshape(1, 5, 3),
                                                                     self.verification_torch[1].reshape(1, -1))
        score_joint = self.sr_biased_torch_gaussian.estimate_score_batch(self.forecast_torch, self.verification_torch)

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

        # unbiased:
        score_1 = self.sr_unbiased_torch_rational_quadratic.estimate_score_batch(
            self.forecast_torch[0].reshape(1, 5, 3), self.verification_torch[0].reshape(1, -1))
        score_2 = self.sr_unbiased_torch_rational_quadratic.estimate_score_batch(
            self.forecast_torch[1].reshape(1, 5, 3), self.verification_torch[1].reshape(1, -1))
        score_joint = self.sr_unbiased_torch_rational_quadratic.estimate_score_batch(self.forecast_torch,
                                                                                     self.verification_torch)

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

        # biased:
        score_1 = self.sr_biased_torch_rational_quadratic.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                                               self.verification_torch[0].reshape(1,
                                                                                                                  -1))
        score_2 = self.sr_biased_torch_rational_quadratic.estimate_score_batch(self.forecast_torch[1].reshape(1, 5, 3),
                                                                               self.verification_torch[1].reshape(1,
                                                                                                                  -1))
        score_joint = self.sr_biased_torch_rational_quadratic.estimate_score_batch(self.forecast_torch,
                                                                                   self.verification_torch)

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

        # hand defined kernel:
        score_1 = self.sr_unbiased_torch_kernel_energy.estimate_score_batch(self.forecast_torch[0].reshape(1, 5, 3),
                                                                            self.verification_torch[0].reshape(1, -1))
        score_2 = self.sr_unbiased_torch_kernel_energy.estimate_score_batch(self.forecast_torch[1].reshape(1, 5, 3),
                                                                            self.verification_torch[1].reshape(1, -1))
        score_joint = self.sr_unbiased_torch_kernel_energy.estimate_score_batch(self.forecast_torch,
                                                                                self.verification_torch)

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

    def test_match_energy_score(self):
        score_1 = self.sr_unbiased_torch_kernel_energy.estimate_score_batch(self.forecast_torch,
                                                                            self.verification_torch)
        score_2 = self.sr_energy_torch.estimate_score_batch(self.forecast_torch, self.verification_torch)

        self.assertTrue(torch.allclose(score_2, score_1))

    def test_mean(self):
        score_mean = self.sr_unbiased_torch_gaussian.estimate_score_batch(self.forecast_torch, self.verification_torch)
        score_no_mean = self.sr_unbiased_torch_gaussian_no_mean.estimate_score_batch(self.forecast_torch,
                                                                                     self.verification_torch)
        self.assertTrue(torch.allclose(score_mean * 2, score_no_mean))


class VariogramScoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.forecast = torch.from_numpy(self.rng.randn(7, 5, 3).astype("float32"))
        self.verification = torch.from_numpy(self.rng.randn(7, 3).astype("float32"))
        variogram = torch.from_numpy(self.rng.uniform(0, 1, (3, 3)).astype("float32"))
        self.sr = VariogramScore(p=1.3, variogram=variogram)
        self.sr_no_mean = VariogramScore(p=1.3, variogram=variogram, mean=False)
        self.sr_max_batch_size = VariogramScore(p=1.3, variogram=variogram, max_batch_size=3)

    def test(self):
        self.sr.estimate_score_batch(self.forecast, self.verification)

    def test_additive(self):
        score_1 = self.sr.estimate_score_batch(self.forecast[0].reshape(1, 5, 3), self.verification[0].reshape(1, -1))
        score_2 = self.sr.estimate_score_batch(self.forecast[1].reshape(1, 5, 3), self.verification[1].reshape(1, -1))
        score_joint = self.sr.estimate_score_batch(self.forecast[0:2], self.verification[0:2])

        self.assertTrue(torch.allclose(score_joint, (score_2 + score_1) / 2))

    def test_mean(self):
        score_mean = self.sr.estimate_score_batch(self.forecast, self.verification)
        score_no_mean = self.sr_no_mean.estimate_score_batch(self.forecast, self.verification)
        self.assertTrue(torch.allclose(score_mean * 7, score_no_mean))

    def test_max_batch_size(self):
        score_mean = self.sr.estimate_score_batch(self.forecast, self.verification)
        score_max_batch_size = self.sr_max_batch_size.estimate_score_batch(self.forecast, self.verification)
        self.assertTrue(torch.allclose(score_mean, score_max_batch_size))


class SumScoringRulesTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.forecast = torch.from_numpy(self.rng.randn(20, 50, 3).astype("float32"))
        self.verification = torch.from_numpy(self.rng.randn(20, 3).astype("float32"))
        variogram = torch.from_numpy(self.rng.uniform(0, 1, (3, 3)).astype("float32"))
        self.variogram_sr = VariogramScore(p=1.3, variogram=variogram)
        self.energy_sr = EnergyScore(beta=1.7)
        self.energy_sr_2 = EnergyScore(beta=1.3)
        self.weights = (1.0, 2.0, 3.0)
        self.sum_sr = SumScoringRules((self.energy_sr, self.variogram_sr, self.energy_sr_2), self.weights)

    def test(self):
        total_sr_val = self.sum_sr.estimate_score_batch(self.forecast, self.verification)
        var_sr_val = self.variogram_sr.estimate_score_batch(self.forecast, self.verification)
        eng_sr_val = self.energy_sr.estimate_score_batch(self.forecast, self.verification)
        eng_sr2_val = self.energy_sr_2.estimate_score_batch(self.forecast, self.verification)

        self.assertTrue(torch.allclose(
            total_sr_val, self.weights[0] * eng_sr_val + self.weights[1] * var_sr_val + self.weights[2] * eng_sr2_val))

    def test_additive(self):
        score_1 = self.sum_sr.estimate_score_batch(self.forecast[0].reshape(1, 50, 3),
                                                   self.verification[0].reshape(1, -1))
        score_2 = self.sum_sr.estimate_score_batch(self.forecast[1].reshape(1, 50, 3),
                                                   self.verification[1].reshape(1, -1))
        score_joint = self.sum_sr.estimate_score_batch(self.forecast[0:2], self.verification[0:2])

        self.assertTrue(torch.allclose(score_joint, score_2 + score_1))


class PatchedScoringRuleTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.forecast = torch.from_numpy(self.rng.randn(2, 5, 3).astype("float32"))
        self.verification = torch.from_numpy(self.rng.randn(2, 3).astype("float32"))
        self.energy_sr = EnergyScore(beta=1.7)

        self.masks = torch.from_numpy(self.rng.randint(2, size=(5, 3), dtype=bool))
        self.patched_sr = PatchedScoringRule(self.energy_sr, self.masks)

        self.nn = createFCNN(3, 3)()

    def test(self):
        total_sr_val = self.patched_sr.estimate_score_batch(self.forecast, self.verification)

        sr_tot = 0
        for i in range(self.masks.shape[0]):
            sr_tot += self.energy_sr.estimate_score_batch(self.forecast[:, :, self.masks[i]],
                                                          self.verification[:, self.masks[i]])
        self.assertTrue(torch.allclose(total_sr_val, sr_tot))

    def test_additive(self):
        score_1 = self.patched_sr.estimate_score_batch(self.forecast[0].reshape(1, 5, 3),
                                                       self.verification[0].reshape(1, -1))
        score_2 = self.patched_sr.estimate_score_batch(self.forecast[1].reshape(1, 5, 3),
                                                       self.verification[1].reshape(1, -1))
        score_joint = self.patched_sr.estimate_score_batch(self.forecast, self.verification)

        self.assertTrue(torch.allclose(score_joint, score_2 + score_1))

    def test_autograd(self):
        # autograd still works even with the masking
        nn_output = self.nn(self.forecast)  # just a way to get a nn output of the correct size
        score = self.patched_sr.estimate_score_batch(nn_output, self.verification)
        score.backward()


class FCNNTests(unittest.TestCase):
    """Tests whether it gives same output with 2d and 3d tensors"""

    def setUp(self):
        self.net = createFCNN(5, 2, nonlinearity=torch.nn.Softplus())()
        self.tensor_2d = torch.randn((12, 5), requires_grad=True)
        self.tensor_3d = self.tensor_2d.reshape(3, 4, 5)

    def test(self):
        out_2d = self.net(self.tensor_2d)
        out_3d = self.net(self.tensor_3d)

        self.assertTrue(torch.allclose(out_2d.reshape(3, 4, -1), out_3d, rtol=0, atol=0))
        self.assertTrue(torch.allclose(out_2d, out_3d.reshape(12, -1), rtol=0, atol=0))


class GenerativeFCNNTests(unittest.TestCase):
    """Tests whether it gives same output with 2d and 3d tensors"""

    def setUp(self):
        batch_size = 3
        auxiliary_var_size = 5
        window_size = 2
        data_size = 5
        input_size = window_size * data_size + auxiliary_var_size
        self.net = createGenerativeFCNN(input_size, 2, nonlinearity=torch.nn.Softplus())()
        self.tensor_3d = torch.randn((batch_size, window_size, data_size))
        self.z = torch.randn((batch_size, 6, auxiliary_var_size))

    def test(self):
        out_3d = self.net(self.tensor_3d, self.z)


class ConditionalGenerativeModelTests(unittest.TestCase):
    def setUp(self):
        self.auxiliary_var_size = 5
        window_size = 2
        self.data_size = 5
        self.input_size = window_size * self.data_size + self.auxiliary_var_size
        self.tensor_3d = torch.randn((3, window_size, self.data_size))

    def test(self):
        for measure in allowed_base_measures:
            self.net = ConditionalGenerativeModel(
                createGenerativeFCNN(self.input_size, self.data_size, nonlinearity=torch.nn.Softplus())(),
                size_auxiliary_variable=self.auxiliary_var_size, number_generations_per_forward_call=6,
                base_measure=measure)
            self.net(self.tensor_3d)


class EstimateBandwidthTests(unittest.TestCase):
    """Tests whether it gives same output with 2d and 3d tensors"""

    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.timeseries = torch.from_numpy(self.rng.randn(20, 30).astype("float32"))

    def test(self):
        estimate_bandwidth_timeseries(self.timeseries)


class WeatherBenchDatasetTests(unittest.TestCase):
    """This works only if the WeatherBench dataset is on the computer in the correct folder"""

    def setUp(self):
        folder = "/"  # todo add folder where you stored the Weatherbench data!
        z500 = xr.open_mfdataset(folder + 'geopotential_500/*.nc', combine='by_coords')
        var_dict = {'z': None}
        observation_window = 3
        lead_time = 10
        self.dataset_hourly_load = WeatherBenchDataset(z500.sel(time=slice('1981', '1981')), var_dict, lead_time,
                                                       observation_window, daily=False)
        self.dataset_hourly_noload = WeatherBenchDataset(z500.sel(time=slice('1981', '1981')), var_dict, lead_time,
                                                         observation_window, load=False, daily=False)
        self.dataset_daily_load = WeatherBenchDataset(z500.sel(time=slice('1981', '1981')), var_dict, lead_time,
                                                      observation_window)
        self.dataset_daily_noload = WeatherBenchDataset(z500.sel(time=slice('1981', '1981')), var_dict, lead_time,
                                                        observation_window, load=False)

    def test_same_load_noload(self):
        x_noload, y_noload = self.dataset_hourly_noload[1]
        x, y = self.dataset_hourly_load[1]
        self.assertTrue(torch.all(x == x_noload))
        self.assertTrue(torch.all(y == y_noload))

        x_noload, y_noload = self.dataset_daily_noload[1]
        x, y = self.dataset_daily_load[1]
        self.assertTrue(torch.all(x == x_noload))
        self.assertTrue(torch.all(y == y_noload))

    def test_same_index_timestring(self):
        timestring = "1981-12-01T12:00:00.000000000"

        x_time, y_time = self.dataset_hourly_load.select_time(timestring)
        x, y = self.dataset_hourly_load[8016]
        assert torch.all(x == x_time)
        assert torch.all(y == torch.from_numpy(y_time.values))

        x_time, y_time = self.dataset_hourly_noload.select_time(timestring)
        x, y = self.dataset_hourly_noload[8016]
        assert torch.all(x == x_time)
        assert torch.all(y == torch.from_numpy(y_time.values))

        x_time, y_time = self.dataset_daily_load.select_time(timestring)
        x, y = self.dataset_daily_load[322]
        assert torch.all(x == x_time)
        assert torch.all(y == torch.from_numpy(y_time.values))

        x_time, y_time = self.dataset_daily_noload.select_time(timestring)
        x, y = self.dataset_daily_noload[322]
        assert torch.all(x == x_time)
        assert torch.all(y == torch.from_numpy(y_time.values))


class LayerNormMineTests(unittest.TestCase):
    """This works only if the WeatherBench dataset is on the computer in the correct folder"""

    def setUp(self):
        self.layer_norm_mine = LayerNormMine()

        N1, C1, H1, W1 = 20, 5, 10, 10
        self.input1 = torch.randn(N1, C1, H1, W1)
        self.layer_norm_1 = torch.nn.LayerNorm([C1, H1, W1], elementwise_affine=False)

        N2, C2, H2, W2 = 20, 6, 20, 20
        self.input2 = torch.randn(N2, C2, H2, W2)
        self.layer_norm_2 = torch.nn.LayerNorm([C2, H2, W2], elementwise_affine=False)

    def test_same(self):
        out_mine_1 = self.layer_norm_mine(self.input1)
        out_mine_2 = self.layer_norm_mine(self.input2)

        out1 = self.layer_norm_1(self.input1)
        out2 = self.layer_norm_2(self.input2)

        self.assertTrue(torch.allclose(out_mine_1, out1))
        self.assertTrue(torch.allclose(out_mine_2, out2))
