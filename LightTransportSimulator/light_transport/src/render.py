import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi
from .ray_old import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize



@numba.njit
def cast_shadow_ray(scene, bvh, intersected_object, intersection_point, intersection_normal):
    light_contrib = np.zeros((3), dtype=np.float64)
    for light in scene.lights:
        shadow_ray_direction = normalize(light.source - intersection_point)
        shadow_ray_magnitude = np.linalg.norm(light.source - intersection_point)
        # shadow_ray = Ray(intersection_point, shadow_ray_direction)

        _objects = traverse_bvh(bvh, intersection_point, shadow_ray_direction)
        _, min_distance = nearest_intersected_object(_objects, intersection_point, shadow_ray_direction, t1=shadow_ray_magnitude)

        if min_distance is None:
            break

        visible = min_distance >= shadow_ray_magnitude
        if visible:
            brdf = (light.material.emission * light.material.color.diffuse) * (intersected_object.material.color.diffuse * inv_pi)
            cos_theta = np.dot(intersection_normal, shadow_ray_direction)
            cos_phi = np.dot(light.normal, -shadow_ray_direction)
            geometry_term = np.abs(cos_theta * cos_phi)/(shadow_ray_magnitude * shadow_ray_magnitude)
            light_contrib += brdf * geometry_term * light.total_area

    light_contrib = light_contrib/len(scene.lights)

    return light_contrib



@numba.njit
def trace_path(scene, bvh, ray_origin, ray_direction, depth, throughput):
    # set the defaults
    color = np.zeros((3), dtype=np.float64)
    # reflection = 1.0

    if depth>scene.max_depth:
        # reached max bounce
        return color

    # r_r = 1.0 # russian roulette factor
    # if depth >= 5:
    #     rr_stop = 0.1
    #     if np.random.rand() <= rr_stop:
    #         return color
    #     r_r = 1.0 / (1.0 - rr_stop)

    if depth > 3:
        r_r = max(0.5, 1-throughput[1]) # russian roulette factor
        if np.random.rand()<r_r:
            return color
        throughput /= 1-r_r

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

    if nearest_object.is_light and depth==0:
        color += nearest_object.material.emission * throughput
        return color

    new_ray_origin = intersection + 1e-5 * surface_normal

    if nearest_object.material.is_diffuse:
        # diffuse color
        # direct light
        direct_light = cast_shadow_ray(scene, bvh, nearest_object, new_ray_origin, surface_normal)

        # indirect light
        indirect_ray_direction, _pdf = cosine_weighted_hemisphere_sampling(surface_normal)
        indirect_ray_origin = intersection + 1e-5 * indirect_ray_direction

        # _prob = 1/(2*np.pi)
        cos_theta = np.dot(indirect_ray_direction, surface_normal)

        # print(cos_theta*inv_pi!=_pdf)

        brdf = nearest_object.material.color.diffuse * inv_pi

        throughput *= brdf * cos_theta / _pdf

        indirect_light = throughput * trace_path(scene, bvh, indirect_ray_origin, indirect_ray_direction, depth+1, throughput)

        color += (direct_light+indirect_light) #* r_r



    elif nearest_object.material.is_mirror:
        # specular color
        new_ray_direction = get_reflected_direction(ray_direction, surface_normal)
        cos_theta = np.dot(ray_direction, surface_normal)
        reflection = nearest_object.material.reflection
        color += trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1, throughput)*reflection #*r_r

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
        reflection_prob = R0 + (1 - R0) * (1 - np.cos(cos_theta))**5

        # reflection
        new_ray_direction = get_reflected_direction(ray_direction, surface_normal)
        color += trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1, throughput)*reflection_prob #*r_r

        Nr = nearest_object.material.ior
        if np.dot(ray_direction, surface_normal)>0:
            Nr = 1/Nr
        Nr = 1/Nr
        cos_theta = -(np.dot(ray_direction, surface_normal))
        _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)

        if _sqrt > 0: # no transmitted ray if negative
            transmit_origin = intersection + (-0.001 * surface_normal)

            transmit_direction = (ray_direction * Nr) + (surface_normal * (Nr * cos_theta - np.sqrt(_sqrt)))
            transmit_direction = normalize(transmit_direction)
            transmit_color = trace_path(scene, bvh, transmit_origin, transmit_direction, depth+1, throughput)

            color += transmit_color*(1 - reflection_prob)*nearest_object.material.transmission #*r_r

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
                throughput = np.ones((3), dtype=np.float64)
                x = left_right[j]
                # screen is on origin
                pixel = np.array([x, y, scene.depth], dtype=np.float64)
                origin = scene.camera
                end = pixel
                # direction = normalize(end - origin)
                ray = Ray(origin, end)
                # for k in range(scene.max_depth):
                color += trace_path(scene, bvh, ray.origin, ray.direction, 0, throughput)
            color = color/number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)
        pix_count+=1
        print((pix_count/scene.height)*100)
    return scene.image