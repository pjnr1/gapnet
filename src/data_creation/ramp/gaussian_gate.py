import numpy as np


def beta_gamma_t(gamma_t: float, sigma: float = 0.0005) -> float:
    """

    @param gamma_t:
    @param sigma:
    @return:
    """
    return np.sqrt(1 / (1 + np.exp(-np.power(gamma_t, 2) / (4 * np.power(sigma, 2)))))


def generate_gaussian_gate(t: np.ndarray,
                           sigma: float = 0.0005,
                           gamma_t: float = 0.0) -> np.ndarray:
    """

    @param t:
        time vector
    @param sigma:
    @param gamma_t:
    @return:
    """
    x = np.exp(-((t - gamma_t) ** 2) / (2 * np.power(sigma, 2)))
    return x


def generate_gaussian_gate_length(t: np.ndarray,
                                  sigma: float = 0.0005,
                                  gamma_t: float = 0.0,
                                  length: float = 0.001) -> np.ndarray:
    """

    @param t:
        time vector
    @param sigma:
    @param gamma_t:
    @param length:
    @return:
    """
    x = np.zeros(t.shape)
    for i in range(int(np.floor(length / sigma)) + 1):
        x += generate_gaussian_gate(t, sigma, gamma_t + sigma * i)
    x /= max(x)
    x *= beta_gamma_t(gamma_t=length, sigma=sigma)
    return x
