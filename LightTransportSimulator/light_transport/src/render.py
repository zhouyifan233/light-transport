import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .primitives import ShapeOptions
from .ray import Ray
from .utils import nearest_intersected_object
from .vectors import normalize


# from LightTransportSimulator.light_transport.src.ray import Ray
# from LightTransportSimulator.light_transport.src.utils import nearest_intersected_object
# from LightTransportSimulator.light_transport.src.brdf import *
# from LightTransportSimulator.light_transport.src.bvh import traverse_bvh


@numba.njit
def hit_object(bvh, ray_origin, ray_direction):
    # get hittable objects
    objects = traverse_bvh(bvh, ray_origin, ray_direction)
    # check for intersections
    nearest_object, min_distance = nearest_intersected_object(objects, ray_origin, ray_direction)

    if nearest_object is None:
        # no object was hit
        return None, None, None, None

    intersection = ray_origin + min_distance * ray_direction
    surface_normal = nearest_object.normal

    return nearest_object, min_distance, intersection, surface_normal

@numba.njit
def random_unit_vector_from_hemisphere(normal_at_intersection):

    # random sample point on hemisphere
    r1 = np.random.rand()
    r2 = np.random.rand()

    theta = math.sqrt(max((0.0, 1.0-r1**2)))
    phi = 2 * np.pi * r2

    _point = [theta * np.cos(phi), theta * np.sin(phi), r1]
    random_point = np.array(_point, dtype=np.float64)


    if abs(normal_at_intersection[0]) > abs(normal_at_intersection[1]):
        inv_len = 1.0 / math.sqrt(normal_at_intersection[0]**2 + normal_at_intersection[2]**2)
        v2 = np.array([-normal_at_intersection[2] * inv_len, 0.0, normal_at_intersection[0] * inv_len], dtype=np.float64)
    else:
        inv_len = 1.0 / math.sqrt(normal_at_intersection[1]**2 + normal_at_intersection[2]**2)
        v2 = np.array([0.0, normal_at_intersection[2] * inv_len, -normal_at_intersection[1] * inv_len], dtype=np.float64)

    v3 = np.cross(normal_at_intersection[:-1], v2)

    rot_x = np.dot(np.array([v2[0], v3[0], normal_at_intersection[0]], dtype=np.float64), random_point)
    rot_y = np.dot(np.array([v2[1], v3[1], normal_at_intersection[1]], dtype=np.float64), random_point)
    rot_z = np.dot(np.array([v2[2], v3[2], normal_at_intersection[2]], dtype=np.float64), random_point)

    global_ray_dir = np.array([rot_x, rot_y, rot_z, 0], dtype=np.float64)

    return global_ray_dir



@numba.njit
def trace_path(scene, bvh, ray_origin, ray_direction, depth):
    # set the defaults
    color = np.zeros((3), dtype=np.float64)
    reflection = 1.0

    if depth>scene.max_depth:
        # reached max bounce
        return color

    r_r = 1.0 # russian roulette factor
    if depth >= 5:
        rr_stop = 0.1
        if np.random.rand() <= rr_stop:
            return color
        r_r = 1.0 / (1.0 - rr_stop)

    # cast a ray
    nearest_object, min_distance, intersection, surface_normal = hit_object(bvh, ray_origin, ray_direction)

    if nearest_object is None:
        # no object was hit
        return color

    ray_inside_object = False
    if np.dot(surface_normal, ray_direction) > 0:
        # print('Flipped')
        surface_normal = -surface_normal # normal facing opposite direction, hence flipped
        ray_inside_object = True
    # else:
    #     print('Not Flipped')

    # color += nearest_object.material.color.ambient # add ambient color

    if nearest_object.is_light:
        color += nearest_object.material.emission * r_r
        return color

    new_ray_origin = intersection + 1e-5 * surface_normal

    if nearest_object.material.is_diffuse:
        # diffuse color
        new_ray_direction = random_unit_vector_from_hemisphere(surface_normal)

        # _prob = 1/(2*np.pi)
        cos_theta = np.dot(new_ray_direction, surface_normal)

        incoming = trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1)

        color += (nearest_object.material.color.diffuse*incoming)*cos_theta*2*r_r

    elif nearest_object.material.is_mirror:
        # specular color
        new_ray_direction = normalize(reflected_ray(ray_direction, surface_normal))
        cos_theta = np.dot(ray_direction, surface_normal)
        reflection *= nearest_object.material.reflection
        color += trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1)*reflection*r_r

    else:
        # compute reflection and refraction
        # use Fresnel
        if ray_inside_object:
            n1 = nearest_object.material.ior
            n2 = 1
        else:
            n1 = 1
            n2 = nearest_object.material.ior

        R0 = ((n1 - n2)/(n1 + n2))**2
        cos_theta = np.dot(ray_direction, surface_normal)
        reflection *= R0 + (1 - R0) * (1 - np.cos(cos_theta))**5

        # reflection
        new_ray_direction = normalize(reflected_ray(ray_direction, surface_normal))
        color += trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1)*reflection*r_r

        Nr = nearest_object.material.ior
        if np.dot(ray_direction, surface_normal)>0:
            Nr = 1/Nr
        Nr = 1/Nr
        cos_theta = -(np.dot(ray_direction, surface_normal))
        _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)

        if _sqrt > 0: # no transmitted ray if negative
            transmit_origin = intersection + (-0.001 * surface_normal)

            transmit_direction = (ray_direction * Nr) + (surface_normal * (Nr * cos_theta - math.sqrt(_sqrt)))
            transmit_direction = normalize(transmit_direction)
            transmit_color = trace_path(scene, bvh, transmit_origin, transmit_direction, depth+1)

            color += transmit_color*(1 - reflection)*nearest_object.material.transmission*r_r

    return color


@numba.njit(parallel=True)
def render_scene(scene, bvh, number_of_samples=10):
    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    pix_count = 0
    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            color = np.zeros((3), dtype=np.float64)
            for _sample in range(number_of_samples):
                x = left_right[j]
                # screen is on origin
                pixel = np.array([x, y, scene.depth], dtype=np.float64)
                origin = scene.camera
                end = pixel
                # direction = normalize(end - origin)
                ray = Ray(origin, end)
                # for k in range(scene.max_depth):
                color += trace_path(scene, bvh, ray.origin, ray.direction, 0)
            color = color/number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)
        pix_count+=1
        print((pix_count/scene.height)*100)
    return scene.image