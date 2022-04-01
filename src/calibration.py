import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


# --- the following 3 are adapted from the Bayesflow verification metrics ---

def calibration_error(theta_samples, theta_test, alpha_resolution=100):
    """
    Computes the calibration error of an approximate posterior per parameters.
    The calibration error is given as the median of the absolute deviation
    between alpha (0 - 1) (credibility level) and the relative number of inliers from
    theta test.

    ----------

    Arguments:
    theta_samples       : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test          : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    alpha_resolution    : int -- the number of intervals to consider

    ----------

    Returns:

    cal_errs  : np.ndarray of shape (n_params, ) -- the calibration errors per parameter
    """

    n_params = theta_test.shape[1]
    n_test = theta_test.shape[0]
    alphas = np.linspace(0.01, 1.0, alpha_resolution)
    cal_errs = np.zeros(n_params)

    # Loop for each parameter
    for k in range(n_params):
        alphas_in = np.zeros(len(alphas))
        # Loop for each alpha
        for i, alpha in enumerate(alphas):
            # Find lower and upper bounds of posterior distribution
            region = 1 - alpha
            lower = np.round(region / 2, 3)
            upper = np.round(1 - (region / 2), 3)

            # Compute quantiles for given alpha using the entire sample
            quantiles = np.quantile(theta_samples[:, :, k], [lower, upper], axis=0).T

            # Compute the relative number of inliers
            inlier_id = (theta_test[:, k] > quantiles[:, 0]) & (theta_test[:, k] < quantiles[:, 1])
            inliers_alpha = np.sum(inlier_id) / n_test
            alphas_in[i] = inliers_alpha

        # Compute calibration error for k-th parameter
        diff_alphas = np.abs(alphas - alphas_in)
        cal_err = np.round(np.median(diff_alphas), 3)
        cal_errs[k] = cal_err

    return cal_errs


def rmse(theta_samples, theta_test, normalized=True):
    """
    Computes the RMSE or normalized RMSE (NRMSE) between posterior means
    and true parameter values for each parameter

    ----------

    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    normalized      : boolean -- whether to compute nrmse or rmse (default True)

    ----------

    Returns:

    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy()
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()

    theta_approx_means = theta_samples.mean(0)
    rmse = np.sqrt(np.mean((theta_approx_means - theta_test) ** 2, axis=0))

    if normalized:
        rmse = rmse / (theta_test.max(axis=0) - theta_test.min(axis=0))
    return rmse


def R2(theta_samples, theta_test):
    """
    Computes the R^2 score as a measure of reconstruction (percentage of variance
    in true parameters captured by estimated parameters)

    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values

    ----------
    Returns:

    r2s  : np.ndarray of shape (n_params, ) -- the r2s per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy()
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()

    theta_approx_means = theta_samples.mean(0)
    return r2_score(theta_test, theta_approx_means, multioutput='raw_values')


def plot_metrics_params(cal_error_values, rmse_values, r2_values, show=False, filename=None, font_size=12):
    """Plots R2 and NRMSE side by side for all parameters over a test set."""

    # Plot initialization
    plt.rcParams['font.size'] = font_size
    f, axarr = plt.subplots(1, 3, figsize=(15, 4))

    n_params = rmse_values.shape[0]

    # Plot calibration error
    axarr[0].plot(np.arange(n_params) + 1, cal_error_values, "o")
    # Plot NRMSE
    axarr[1].plot(np.arange(n_params) + 1, rmse_values, "o")
    # Plot R2
    axarr[2].plot(np.arange(n_params) + 1, r2_values, "o")

    names = ['Cal error', 'NRMSE', '$R^2$']
    # Tweak plots
    for i, name in enumerate(names):
        axarr[i].set_xlabel('Parameter #')
        axarr[i].set_ylabel(name)
        axarr[i].set_title('Test ' + name)
        axarr[i].spines['right'].set_visible(False)
        axarr[i].spines['top'].set_visible(False)

    f.tight_layout()

    if show:
        plt.show()

    if filename is not None:
        f.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
