import numba


@numba.experimental.jitclass([
    ('ambient', numba.float64[:]),
    ('diffuse', numba.float64[:]),
    ('specular', numba.float64[:])
])
class Color:
    def __init__(self, ambient, diffuse, specular):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular


# color_type = numba.deferred_type()

@numba.experimental.jitclass([
    ('color', Color.class_type.instance_type),
    ('shininess', numba.float64),
    ('reflectance', numba.float64),
    ('ior', numba.float64),
    ('is_diffuse', numba.boolean),
    ('is_mirror', numba.boolean),
    ('emission', numba.float64),
    ('transmittance', numba.float64),
    ('roughness', numba.float64),
    ('albedo', numba.float64)
])
class Material:
    def __init__(self, color, shininess, reflectance, ior, emission=0.0, transmittance=0.0, is_diffuse=True, is_mirror=False):
        self.color = color
        self.shininess = shininess
        self.reflectance = reflectance
        self.is_diffuse = is_diffuse
        self.is_mirror = is_mirror
        self.ior = ior
        self.emission = emission
        self.transmittance = transmittance
        self.roughness = 0.0
        self.albedo = 1.0


# color_type.define(Color.class_type.instance_type)