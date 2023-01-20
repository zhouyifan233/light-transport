import numpy as np
import numba
from .vectors import normalize
from .constants import Medium




@numba.experimental.jitclass([
    ('point', numba.float64[:]),
    ('color', numba.float64[:]),
    ('g_norm', numba.float64[:]),
    ('pdf_pos', numba.float64),
    ('pdf_dir', numba.float64),
    ('importance', numba.float64),
    ('pdf_fwd', numba.float64),
    ('pdf_rev', numba.float64),
    ('hit_light', numba.boolean),
    ('medium', numba.intp),
    ('throughput', numba.float64[:]),
    ('geometry_term', numba.float64[:]),
    ('is_delta', numba.boolean)
])
class Vertex:
    def __init__(self, point):
        self.point = point
        self.g_norm = np.zeros((3), dtype=np.float64)
        self.color = np.zeros((3), dtype=np.float64)
        self.pdf_pos = 0.0
        self.pdf_dir = 0.0
        self.importance = 0.0
        self.pdf_fwd = 0.0
        self.pdf_rev = 0.0
        self.hit_light = False
        self.medium = Medium.NONE.value
        self.throughput = np.ones((3), dtype=np.float64)
        self.geometry_term = np.zeros((3), dtype=np.float64)
        self.is_delta = False