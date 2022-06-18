import numpy as np
import numba

from .intersects import sphere_intersect, triangle_intersect, plane_intersect
from .shapes import Triangle, Sphere, Plane, ShapeOptions


@numba.njit
def nearest_intersected_object(objects, ray_origin, ray_end):
    """
    returns the nearest object that a ray intersects, if it exists
    :param objects: list of all objects
    :param ray_origin: origin of the ray (eg. camera)
    :param ray_end: pixel on the object
    :return: nearest object and the minimum distance to that object
    """
    distances = [find_intersect(obj, ray_origin, ray_end) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance is not None and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


@numba.njit
def find_intersect(obj, ray_origin, ray_end):
    # if isinstance(obj, Plane):
    #     return plane_intersect(ray_origin, ray_end, obj)
    # elif isinstance(obj, Triangle):
    #     return triangle_intersect(ray_origin, ray_end, obj)
    # elif isinstance(obj, Sphere):
    #     return sphere_intersect(ray_origin, ray_end, obj)
    # else:
    #     return []
    # if obj.type == 3:
    #     return plane_intersect(ray_origin, ray_end, obj)
    if obj.type == ShapeOptions.TRIANGLE.value:
        return triangle_intersect(ray_origin, ray_end, obj)
    # elif obj.type == 2:
    #     return sphere_intersect(ray_origin, ray_end, obj)
    # else:
    #     return []


# def load_obj(file_path):
