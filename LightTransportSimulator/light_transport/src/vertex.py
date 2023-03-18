import numpy as np
import numba

from .vectors import normalize
from .constants import Medium, ONES


@numba.experimental.jitclass([
    ('point', numba.float64[:]),
    ('color', numba.float64[:]),
    ('g_norm', numba.float64[:]),
    ('min_distance', numba.float64),
    ('ray_direction', numba.float64[:]),
    ('pdf_pos', numba.float64),
    ('pdf_dir', numba.float64),
    ('importance', numba.float64),
    ('pdf_fwd', numba.float64),
    ('pdf_rev', numba.float64),
    ('hit_light', numba.boolean),
    ('medium', numba.intp),
    ('intr_type', numba.intp),
    ('throughput', numba.float64),
    ('geometry_term', numba.float64[:]),
    ('is_delta', numba.boolean)
])
class Vertex:
    def __init__(self, point, ray_direction, g_norm, min_distance):
        self.point = point
        self.g_norm = g_norm
        self.min_distance = min_distance
        self.ray_direction = ray_direction
        self.color = np.zeros((3), dtype=np.float64)
        self.pdf_pos = 0.0
        self.pdf_dir = 0.0
        self.importance = 0.0
        self.pdf_fwd = 0.0
        self.pdf_rev = 0.0
        self.hit_light = False
        self.medium = Medium.NONE.value
        self.intr_type = Medium.NONE.value
        self.throughput = 1.0
        self.geometry_term = np.zeros((3), dtype=np.float64)
        self.is_delta = False


@numba.njit
def create_camera_vertex(point, ray_direction, normal, min_distance, pdf_pos, pdf_dir, throughput):
    camera = Vertex(point, ray_direction, normal, min_distance)
    camera.medium = Medium.CAMERA.value
    camera.g_norm = normal
    camera.pdf_pos = pdf_pos
    camera.pdf_dir = pdf_dir
    camera.throughput = throughput
    return camera


@numba.njit
def create_light_vertex(light, ray_direction, min_distance, pdf_dir, pdf_fwd):
    light_v = Vertex(light.source, ray_direction, light.normal, min_distance)
    light_v.medium = Medium.LIGHT.value
    light_v.pdf_pos = 1/light.total_area
    light_v.pdf_dir = pdf_dir
    light_v.pdf_fwd = pdf_fwd
    light_v.color = light.material.color.diffuse
    return light_v


@numba.njit
def create_surface_vertex(point, ray_direction, normal, min_distance, throughput, color, pdf_fwd, prev_v):
    surface = Vertex(point, ray_direction, normal, min_distance)
    surface.medium = Medium.SURFACE.value
    surface.throughput = throughput
    surface.color = color
    surface.pdf_fwd = convert_density(pdf_fwd, prev_v, surface)
    return surface


@numba.njit
def convert_density(pdf, prev_v, next_v):
    '''
    converts pdf to solid angle density
    :param pdf: pdf
    :param prev_v: previous vertex
    :param next_v: next or current vertex
    :return: solid angle density
    '''
    path = normalize(next_v.point - prev_v.point)
    path_magnitude = np.linalg.norm(next_v.point - prev_v.point)
    if path_magnitude==0:
        return 0
    inv_dist_sqr = 1/(path_magnitude*path_magnitude)
    if next_v.medium==Medium.SURFACE.value:
        pdf *= np.abs(np.dot(next_v.g_norm, path*np.sqrt(inv_dist_sqr)))
    return pdf * inv_dist_sqr


@numba.njit
def is_on_surface(vx):
    return vx.medium==Medium.SURFACE.value


@numba.njit
def is_connectible(vx):
    if vx.medium==Medium.LIGHT.value or vx.medium==Medium.CAMERA.value:
        # Assuming all lights are area lights, hence no delta light
        return True
    elif vx.medium==Medium.SURFACE.value and not vx.is_delta:
        return True
    else:
        return False


@numba.njit
def get_vertex_color(vx):
    if vx.medium==Medium.SURFACE.value or vx.medium==Medium.LIGHT.value:
        return vx.color
    else:
        return ONES