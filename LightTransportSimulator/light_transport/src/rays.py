import numpy as np
import numba
from .vectors import normalize
from .constants import Medium


c_1d_vec = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")

@numba.experimental.jitclass([
    ('origin', numba.float64[:]),
    ('direction', numba.float64[:]),
    ('inv_dir', numba.float64[:]),
    ('tmin', numba.float64),
    ('tmax', numba.float64)
])
class Ray:
    def __init__(self, origin, direction, tmin=0.0):
        self.origin = origin
        self.direction = direction
        self.tmin = tmin
        self.tmax = np.inf
