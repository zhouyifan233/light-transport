import numpy as np
import numba
from .material import Material


@numba.experimental.jitclass([
    ('source', numba.float64[:]),
    ('material', Material.class_type.instance_type),
    ('normal', numba.float64[:]),
    ('total_area', numba.float64)
])
class Light:
    def __init__(self, source, material, normal, total_area):
        self.source = source
        self.material = material
        self.normal = normal
        self.total_area = total_area


@numba.experimental.jitclass([
    ('position', numba.float64[:]),
    ('focal_length', numba.intp)
])
class Camera:
    def __init__(self, position, focal_length):
        self.position = position
        self.focal_length = focal_length


@numba.experimental.jitclass([
    ('camera', numba.float64[:]),
    ('lights', numba.types.ListType(Light.class_type.instance_type)),
    # ('lights', numba.types.ListType(numba.float64[::1])),
    ('width', numba.uintp),
    ('height', numba.uintp),
    ('max_depth', numba.uintp),
    ('aspect_ratio', numba.float64),
    ('left', numba.float64),
    ('top', numba.float64),
    ('right', numba.float64),
    ('bottom', numba.float64),
    ('f_distance', numba.float64),
    ('number_of_samples', numba.uintp),
    ('image', numba.float64[:,:,:]),
    ('rand_0', numba.float64[:,:,:]),
    ('rand_1', numba.float64[:,:,:])
])
class Scene:
    def __init__(self, camera, lights, width=400, height=400, max_depth=3, f_distance=5, number_of_samples=8):
        self.camera = camera
        self.lights = lights
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.aspect_ratio = width/height
        self.left = -1
        self.top = 1/self.aspect_ratio
        self.right = 1
        self.bottom = -1/self.aspect_ratio
        self.f_distance = f_distance
        self.image = np.zeros((height, width, 3), dtype=np.float64)
        self.number_of_samples = number_of_samples
        self.rand_0 = np.random.rand(height, width, number_of_samples)
        self.rand_1 = np.random.rand(height, width, number_of_samples)