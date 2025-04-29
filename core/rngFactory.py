import numpy as np

def rand_uniform(low = 0.0, high = 1.0, shape = None):
    if shape is None:
        return np.random.uniform(low, high)
    else:
        return np.random.uniform(low, high, shape)

def rand_positive(shape = None, scale = 1.0):
    if shape is None:
        return np.random.uniform(0.0, 1.0)
    else:
        return np.random.uniform(0.0, 1.0, shape) * scale



