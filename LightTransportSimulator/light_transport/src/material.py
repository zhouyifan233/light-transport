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
    ('ior', numba.float64),
    ('type', numba.intp),
    ('emission', numba.float64),
    ('roughness', numba.float64),
    ('albedo', numba.float64)
])
class Material:
    def __init__(self, color, shininess, ior, type, emission=0.0):
        self.color = color
        self.shininess = shininess
        self.ior = ior
        self.type = type
        self.emission = emission
        self.roughness = 0.0
        self.albedo = 1.0


# color_type.define(Color.class_type.instance_type)