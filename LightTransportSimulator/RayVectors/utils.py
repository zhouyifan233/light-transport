import numpy as np

from LightTransportSimulator.RayVectors.intersects import sphere_intersect, triangle_intersect, plane_intersect


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
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def find_intersect(obj, ray_origin, ray_end):
    if obj['type'] == 'sphere':
        return sphere_intersect(obj['geom_props']['center'], obj['geom_props']['radius'], ray_origin, ray_end)
    elif obj['type'] == 'triangle':
        return triangle_intersect(ray_origin, ray_end, obj['geom_props']['a'], obj['geom_props']['b'], obj['geom_props']['c'])
    elif obj['type'] == 'plane':
        ab = obj['geom_props']['b'] - obj['geom_props']['a']
        ac = obj['geom_props']['c'] - obj['geom_props']['a']
        plane_normal = np.cross(ab, ac)
        return plane_intersect(ray_origin, ray_end, obj['geom_props']['a'], plane_normal)
    else:
        return []