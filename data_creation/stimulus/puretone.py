import numpy as np


def beta_gamma_t(gamma_t: float, sigma: float = 0.0005) -> float:
    return np.sqrt(1 / (1 + np.exp(-np.power(gamma_t, 2) / (4 * np.power(sigma, 2)))))


def generate_gaussian_gate(t, sigma=0.0005, gamma_t=0.0):
    return np.exp(-((t - gamma_t) ** 2) / (2 * np.power(sigma, 2)))


def generate_gaussian_gate_length(t, sigma=0.0005, gamma_t=0.0, length=0.001):
    x = np.zeros(t.shape)
    for i in range(int(np.floor(length / sigma)) + 1):
        x += beta_gamma_t(gamma_t=sigma, sigma=sigma) * generate_gaussian_gate(t, sigma, gamma_t + sigma * i)
    return x
