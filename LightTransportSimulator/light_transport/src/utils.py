import numpy as np
import numba

from .intersects import sphere_intersect, triangle_intersect, plane_intersect, __triangle_intersect, pc_triangle_intersect
from .primitives import Triangle, Sphere, Plane, ShapeOptions


@numba.njit
def nearest_intersected_object(objects, ray_origin, ray_direction, t0=0.0, t1=np.inf):
    """
    returns the nearest object that a ray intersects, if it exists
    :param objects: list of all objects
    :param ray_origin: origin of the ray (eg. camera)
    :param ray_end: pixel on the object
    :return: nearest object and the minimum distance to that object
    """
    # distances = numba.typed.List()
    #
    # for obj in objects:
    #     distances.append(triangle_intersect(ray_origin, ray_end, obj))

    distances = [triangle_intersect(ray_origin, ray_direction, obj) for obj in objects]

    # print(distances)

    nearest_object = None
    min_distance = np.inf #t1

    for index, distance in enumerate(distances):
        if distance is not None and distance < min_distance:
            if t0<distance<(t1-0.00001):
            # if not objects[index].is_light:
                min_distance = distance
                nearest_object = objects[index]

    return nearest_object, min_distance