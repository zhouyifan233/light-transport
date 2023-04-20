import numpy as np
import numba
from .material import Material
from .primitives import Sphere
from .vectors import normalize


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
    ('look_at', numba.float64[:]),
    ('scene_normal', numba.float64[:]),
    ('fov', numba.float64),
    ('focal_length', numba.optional(numba.intp)),
    ('normal', numba.optional(numba.float64[:])),
    ('screen_area', numba.optional(numba.float64))
])
class Camera:
    def __init__(self, position):
        self.position = position
        self.look_at = normalize(np.array([0, -0.042612, -1], dtype=np.float64))
        self.scene_normal = np.array([0.0, 1.0, 0.0], np.float64)
        self.fov = np.deg2rad(30) #0.5135
        self.focal_length = None
        self.normal = None
        self.screen_area = None


def generate_custom_camera(width, height, fov):
    aspect_ratio = width / height
    near = 0.1
    far = 100.0
    gaze = np.array([0, -0.042612, -1])
    gaze /= np.linalg.norm(gaze)
    right = np.cross(np.array([0, 1, 0]), gaze)
    right /= np.linalg.norm(right)
    up = np.cross(gaze, right)
    up /= np.linalg.norm(up)

    h = 2 * near * np.tan(fov / 2)
    w = h * aspect_ratio

    M = np.zeros((4,4))
    M[0][0] = w / width
    M[1][1] = h / height
    M[2][2] = -(far + near) / (far - near)
    M[2][3] = -(2 * far * near) / (far - near)
    M[3][2] = -1

    R = np.identity(4)
    R[:3, :3] = np.column_stack((right, up, -gaze))

    T = np.identity(4)
    T[:3, 3] = [0, 0, 0]

    camera_matrix = np.dot(np.dot(M, T), R)

    return camera_matrix



@numba.experimental.jitclass([
    ('camera', Camera.class_type.instance_type),
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
    ('rand_0', numba.float64[:,:,:,:]),
    ('rand_1', numba.float64[:,:,:,:])
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
        self.rand_0 = np.random.rand(height, width, number_of_samples, max_depth)
        self.rand_1 = np.random.rand(height, width, number_of_samples, max_depth)



@numba.experimental.jitclass([
    ('camera', Camera.class_type.instance_type),
    # ('lights', Sphere.class_type.instance_type),
    ('lights', numba.types.ListType(Light.class_type.instance_type)),
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
    ('rand_0', numba.float64[:,:,:,:]),
    ('rand_1', numba.float64[:,:,:,:]),
    ('t_matrix', numba.float64[:, :])
])
class SphereScene:
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
        self.rand_0 = np.random.rand(width, height, number_of_samples, max_depth)
        self.rand_1 = np.random.rand(width, height, number_of_samples, max_depth)
        self.t_matrix = np.identity(4) # required transformations


