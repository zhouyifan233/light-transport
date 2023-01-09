import numpy as np
import numba
from .vectors import normalize
from .constants import Medium


c_1d_vec = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")

@numba.experimental.jitclass([
    ('origin', numba.float64[:]),
    ('direction', numba.float64[:]),
    ('color', numba.float64[:]),
    ('g_norm', numba.float64[:]),
    ('pdf_pos', numba.float64),
    ('pdf_dir', numba.float64),
    ('importance', numba.float64),
    ('rev_pdf', numba.float64),
    ('fwd_pdf', numba.float64),
    ('hit_light', numba.boolean),
    ('medium', numba.intp),
    ('throughput', numba.float64[:]),
    ('geometry_term', numba.float64[:])
])
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.g_norm = np.zeros((3), dtype=np.float64)
        self.color = np.zeros((3), dtype=np.float64)
        self.pdf_pos = 0.0
        self.pdf_dir = 0.0
        self.importance = 0.0
        self.rev_pdf = 0.0
        self.fwd_pdf = 0.0
        self.hit_light = False
        self.medium = Medium.NONE.value
        self.throughput = np.ones((3), dtype=np.float64)
        self.geometry_term = np.zeros((3), dtype=np.float64)