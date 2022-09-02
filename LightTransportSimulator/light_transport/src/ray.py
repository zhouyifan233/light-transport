import numpy as np
import numba
from .vectors import normalize


c_1d_vec = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")

@numba.experimental.jitclass([
    ('origin', c_1d_vec),
    ('end', c_1d_vec),
    ('direction', c_1d_vec),
    ('magnitude', numba.float64)
])
class Ray:
    def __init__(self, origin, end):
        self.origin = np.asarray([origin[0], origin[1], origin[2], 1])
        self.end = np.asarray([end[0], end[1], end[2], 1])
        self.direction = normalize(self.end - self.origin)
        self.magnitude = np.linalg.norm(self.end - self.origin)