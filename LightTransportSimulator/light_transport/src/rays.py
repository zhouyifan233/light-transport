import numpy as np
import numba
from .vectors import normalize


c_1d_vec = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")

@numba.experimental.jitclass([
    ('origin', c_1d_vec),
    ('direction', c_1d_vec)
])
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.asarray([origin[0], origin[1], origin[2], 1])
        self.direction = np.asarray([direction[0], direction[1], direction[2], 0])