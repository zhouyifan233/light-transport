import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi, inv_2_pi
from .ray_old import Ray
from .utils import cosine_weighted_hemisphere_sampling, uniform_hemisphere_sampling, create_orthonormal_system
from .utils import hit_object, nearest_intersected_object
from .vectors import normalize, get_direction



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
            brdf = light.material.emission * light.material.color.diffuse
            cos_theta = np.dot(intersection_normal, shadow_ray_direction)
            cos_phi = np.dot(light.normal, -shadow_ray_direction)
            geometry_term = np.abs(cos_theta * cos_phi)/(shadow_ray_magnitude * shadow_ray_magnitude)
            light_contrib += brdf * geometry_term * light.total_area

    light_contrib = light_contrib/len(scene.lights)

    return light_contrib



@numba.njit
def trace_path(scene, bvh, ray_origin, ray_direction, depth, weight=1):
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
        color += (nearest_object.material.emission * nearest_object.material.color.diffuse) * r_r * weight
        return color

    new_ray_origin = intersection + 1e-5 * surface_normal

    if nearest_object.material.is_diffuse:
        # direct lighting
        direct_light = cast_shadow_ray(scene, bvh, nearest_object, new_ray_origin, surface_normal)

        # indirect lighting
        indirect_light = np.zeros((3), dtype=np.float64)
        number_of_samples=10
        for n in range(number_of_samples):
            # random uniform samples
            r1 = np.random.rand()
            r2 = np.random.rand()

            theta = math.sqrt(max((0.0, 1.0-r1**2)))
            phi = 2 * np.pi * r2

            _point = [theta * np.cos(phi), theta * np.sin(phi), r1]

            sample_point = np.array(_point, dtype=np.float64)

            v2, v3 = create_orthonormal_system(surface_normal)

            indirect_ray_direction = np.array([sample_point[0] * v3[0] + sample_point[1] * surface_normal[0] + sample_point[2] * v2[0],
                                       sample_point[0] * v3[1] + sample_point[1] * surface_normal[1] + sample_point[2] * v2[1],
                                       sample_point[0] * v3[2] + sample_point[1] * surface_normal[2] + sample_point[2] * v2[2],
                                      0], dtype=np.float64)

            pdf = inv_2_pi

            indirect_ray_origin = intersection + 1e-5 * indirect_ray_direction

            cos_theta = np.dot(indirect_ray_direction, surface_normal)
            indirect_light += cos_theta*trace_path(scene, bvh, indirect_ray_origin, indirect_ray_direction, depth+1, weight)/pdf

        indirect_light = indirect_light/number_of_samples

        color +=  (direct_light*inv_pi + 2*indirect_light) * nearest_object.material.color.diffuse * r_r * weight

    elif nearest_object.material.is_mirror:
        # mirror reflection
        new_ray_direction = normalize(reflected_ray(ray_direction, surface_normal))
        color += trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1, weight) * r_r * weight

    elif nearest_object.material.transmission>0:
        # reflection and refraction
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
        new_ray_direction = normalize(reflected_ray(ray_direction, surface_normal))
        color += trace_path(scene, bvh, new_ray_origin, new_ray_direction, depth+1, weight)*reflection_prob*r_r*weight

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

            color += transmit_color*(1 - reflection_prob)*nearest_object.material.transmission*r_r*weight

    # weight *= nearest_object.material.reflection

    return color


@numba.njit(parallel=True)
def render_scene(scene, bvh):
    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    pix_count = 0
    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            color = np.zeros((3), dtype=np.float64)
            # for _sample in range(number_of_samples):
            x = left_right[j]
            # screen is on origin
            pixel = np.array([x, y, scene.depth], dtype=np.float64)
            origin = scene.camera
            end = pixel
            # direction = normalize(end - origin)
            ray = Ray(origin, end)
            # for k in range(scene.max_depth):
            color += trace_path(scene, bvh, ray.origin, ray.direction, 0)
            # color = color/number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)
        pix_count+=1
        print((pix_count/scene.height)*100)
    return scene.image