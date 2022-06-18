import numpy as np
import numba
from .material import Material


@numba.experimental.jitclass([
    ('source', numba.float64[:]),
    ('material', Material.class_type.instance_type)
])
class Light:
    def __init__(self, source, material):
        self.source = source
        self.material = material


@numba.experimental.jitclass([
    ('camera', numba.float64[:]),
    ('light', Light.class_type.instance_type),
    ('width', numba.int64),
    ('height', numba.int64),
    ('max_depth', numba.int64),
    ('aspect_ratio', numba.float64),
    ('left', numba.float64),
    ('top', numba.float64),
    ('right', numba.float64),
    ('bottom', numba.float64),
    ('image', numba.float64[:,:,:])
])
class Scene:
    def __init__(self, camera, light, width=400, height=400, max_depth=3):
        self.camera = camera
        self.light = light
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.aspect_ratio = width/height
        self.left = -1
        self.top = 1/self.aspect_ratio
        self.right = 1
        self.bottom = -1/self.aspect_ratio
        self.image = np.zeros((height, width, 3), dtype=np.float64)