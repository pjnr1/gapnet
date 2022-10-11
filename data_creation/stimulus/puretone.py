import numpy as np


def beta_gamma_t(gamma_t: float, sigma: float = 0.0005) -> float:
    """
    
    @arg gamma_t: 
    @arg sigma: 
    @return: 
    """
    return np.sqrt(1 / (1 + np.exp(-np.power(gamma_t, 2) / (4 * np.power(sigma, 2)))))


def generate_gaussian_gate(t, sigma=0.0005, gamma_t=0.0):
    """
    
    @arg t: 
        time vector
    @arg sigma: 
    @arg gamma_t: 
    @return: 
    """
    return np.exp(-((t - gamma_t) ** 2) / (2 * np.power(sigma, 2)))


def generate_gaussian_gate_length(t, sigma=0.0005, gamma_t=0.0, length=0.001):
    """
    
    @arg t: 
        time vector
    @arg sigma: 
    @arg gamma_t: 
    @arg length: 
    @return: 
    """
    x = np.zeros(t.shape)
    for i in range(int(np.floor(length / sigma)) + 1):
        x += beta_gamma_t(gamma_t=sigma, sigma=sigma) * generate_gaussian_gate(t, sigma, gamma_t + sigma * i)
    return x


def puretone_randomphase(t, frequency) -> np.ndarray:
    """
    
    @arg t: 
        time vector
    @arg frequency:
        frequency of the puretone
    @return:
        the puretone signal
    """
    phase = 2 * np.pi * np.random.rand(1)
    return np.sin(2 * np.pi * frequency * t + phase)
