from .material import Material
from .vectors import normalize
import numpy as np
import numba
import enum


class ShapeOptions(enum.Enum):
    TRIANGLE = 1
    PLANE = 2
    SPHERE = 3



@numba.experimental.jitclass([
    ('type', numba.intp),
    ('vertex_1', numba.float64[:]),
    ('vertex_2', numba.float64[:]),
    ('vertex_3', numba.float64[:]),
    ('material', Material.class_type.instance_type),
    ('normal', numba.float64[:])
])
class Triangle:
    def __init__(self, vertex_1, vertex_2, vertex_3, material):
        self.type = ShapeOptions.TRIANGLE.value
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.vertex_3 = vertex_3
        self.material = material
        self.normal = normalize(np.cross(vertex_2-vertex_1, vertex_3-vertex_1))


@numba.experimental.jitclass([
    ('type', numba.intp),
    ('center', numba.float64[:]),
    ('radius', numba.float64[:]),
    ('material', Material.class_type.instance_type)
])
class Sphere:
    def __init__(self, center, radius, material):
        self.type = ShapeOptions.SPHERE.value
        self.center = center
        self.radius = radius
        self.material = material


@numba.experimental.jitclass([
    ('type', numba.intp),
    ('point', numba.float64[:]),
    ('normal', numba.float64[:]),
    ('material', Material.class_type.instance_type)
])
class Plane:
    def __init__(self, point, normal, material):
        self.type = ShapeOptions.PLANE.value
        self.point = point
        self.normal = normal
        self.material = material
