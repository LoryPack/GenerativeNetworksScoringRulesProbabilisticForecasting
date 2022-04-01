import numpy as np
from numba import jit


# following code adapted from the Lorenz96 parametrization with GAN paper

@jit(nopython=True, cache=True)
def l96_truth_step(X, Y, h, F, b, c):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Y[k * J: (k + 1) * J])
    for j in range(J * K):
        dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j - 1]) - c * Y[j] + h * c / b * X[
            int(j / J)]
    return dXdt, dYdt


@jit(nopython=True, cache=True)
def run_lorenz96_truth(x_initial, y_initial, time_step, num_steps, burn_in, skip, h=1, b=10.0, c=10.0, F=20.0):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        F (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival

    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    # Calculate total Y forcing over archive period using trapezoidal rule
    y_trap = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    y_trap[:] = y_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            print(n, n * time_step)
        k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, F, b, c)
        k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                                y + k1_dydt * time_step / 2,
                                                h, F, b, c)
        k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                                y + k2_dydt * time_step / 2,
                                                h, F, b, c)
        k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                                y + k3_dydt * time_step,
                                                h, F, b, c)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = (y + y_trap) / skip
            i += 1
        elif n % skip == 1:
            y_trap[:] = y
        else:
            y_trap[:] += y
    return x_out, y_out, times, steps
