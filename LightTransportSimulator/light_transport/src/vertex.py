import numpy as np
import numba

from .bdpt_utils import convert_density
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


@numba.njit
def create_camera_vertex(point, normal, pdf_pos, pdf_dir, throughput):
    camera = Vertex(point)
    camera.medium = Medium.CAMERA.value
    camera.g_norm = normal
    camera.pdf_pos = pdf_pos
    camera.pdf_dir = pdf_dir
    camera.throughput = throughput
    return camera


@numba.njit
def create_light_vertex(light, pdf_dir, pdf_fwd):
    light_v = Vertex(light.source)
    light_v.medium = Medium.LIGHT.value
    light_v.g_norm = light.normal
    light_v.pdf_pos = 1/light.total_area
    light_v.pdf_dir = pdf_dir
    light_v.pdf_fwd = pdf_fwd
    light_v.color = light.material.color
    return light_v


@numba.njit
def create_surface_vertex(point, throughput, pdf_fwd, prev_v):
    surface = Vertex(point)
    surface.medium = Medium.SURFACE.value
    surface.throughput = throughput
    surface.pdf_fwd = convert_density(pdf_fwd, prev_v, surface)
    return surface




