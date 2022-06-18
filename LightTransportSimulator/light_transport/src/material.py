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
    ('reflection', numba.float64)
])
class Material:
    def __init__(self, color, shininess, reflection):
        self.color = color
        self.shininess = shininess
        self.reflection = reflection


# color_type.define(Color.class_type.instance_type)