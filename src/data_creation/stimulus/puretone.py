import numpy as np


def puretone_randomphase(t, frequency) -> np.ndarray:
    """
    
    @param t: 
        time vector
    @param frequency:
        frequency of the puretone
    @return:
        the puretone signal
    """
    phase = 2 * np.pi * np.random.rand(1)
    return np.sin(2 * np.pi * frequency * t + phase)


def puretone_syncedphase(t, start_time, frequency) -> np.ndarray:
    """
    @param t:
        time vector
    @param start_time:
        start time of the puretone
    @param frequency:
        frequency of the puretone
    @return:
        the puretone signal
    """
    return np.sin(2 * np.pi * frequency * (t + start_time))
