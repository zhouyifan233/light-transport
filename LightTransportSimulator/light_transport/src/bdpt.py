import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .ray_old import Ray
from .utils import hit_object, random_unit_vector_from_hemisphere, nearest_intersected_object
from .vectors import normalize


@numba.njit
def cast_shadow_ray(scene, bvh, intersected_object, intersection_point):
    light_contrib = 0
    for light in scene.lights:
        shadow_ray_direction = normalize(light.source - intersection_point)
        shadow_ray_magnitude = np.linalg.norm(light.source - intersection_point)
        shadow_ray = Ray(intersection_point, shadow_ray_direction)

        _objects = traverse_bvh(bvh, shadow_ray.origin, shadow_ray.direction)
        _, min_distance = nearest_intersected_object(_objects, shadow_ray.origin, shadow_ray.direction, t1=shadow_ray_magnitude)

        if min_distance is None:
            break

        is_shadowed = min_distance < shadow_ray_magnitude

        if is_shadowed:
            # no intensity
            continue
        else:
            cos_phi = np.dot(shadow_ray_direction, -light.normal)
            intensity = light.material.emission * light.total_area / np.square(shadow_ray_direction)
            light_contrib += light.material.color.diffuse * intensity * max(cos_phi, 0.0)

    total_light_contrib = light_contrib/len(scene.lights)
    albedo = intersected_object.material.color.diffuse/np.pi
    intersection_color = albedo * total_light_contrib
    return intersection_color



@numba.njit
def trace_path(scene, bvh, ray_origin, ray_direction, depth):
    # set the defaults
    color = np.zeros((3), dtype=np.float64)

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
        surface_normal = -surface_normal # normal facing opposite direction, hence flipped
        ray_inside_object = True

    # color += nearest_object.material.color.ambient # add ambient color

    if nearest_object.is_light:
        color += nearest_object.material.emission * nearest_object.material.color * r_r
        return color

    new_ray_origin = intersection + 1e-5 * surface_normal

    # direct lighting
    direct_light_contrib = cast_shadow_ray(scene, bvh, nearest_object, new_ray_origin)


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