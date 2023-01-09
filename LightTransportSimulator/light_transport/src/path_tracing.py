import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi, EPSILON
from .light_samples import cast_one_shadow_ray
from .rays import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize


@numba.njit
def trace_path(scene, bvh, ray, bounce):
    throughput = np.ones((3), dtype=np.float64)
    light = np.zeros((3), dtype=np.float64)
    specular_bounce = False

    while True:
        # terminate path if max bounce is reached
        if bounce>=scene.max_depth:
            break

        _rand = np.random.rand()

        # intersect ray with scene
        nearest_object, min_distance, intersection, surface_normal = hit_object(bvh, ray.origin, ray.direction)

        # terminate path if no intersection is found
        if nearest_object is None:
            # no object was hit
            break

        # add emitted light at intersection
        if nearest_object.is_light and bounce==0:
            light += nearest_object.material.emission * throughput

        ray_inside_object = False
        if np.dot(surface_normal, ray.direction) > 0:
            surface_normal = -surface_normal # normal facing opposite direction, hence flipped
            ray_inside_object = True

        # check if diffuse surface
        if nearest_object.material.is_diffuse:
            shadow_ray_origin = intersection + EPSILON * surface_normal
            # direct light contribution
            direct_light = cast_one_shadow_ray(scene, bvh, nearest_object, shadow_ray_origin, surface_normal)

            # indirect light contribution
            indirect_ray_direction, pdf = cosine_weighted_hemisphere_sampling(surface_normal)

            if pdf==0:
                break

            indirect_ray_origin = intersection + EPSILON * indirect_ray_direction

            # change ray direction
            ray.origin = indirect_ray_origin
            ray.direction = indirect_ray_direction

            cos_theta = np.dot(indirect_ray_direction, surface_normal)

            brdf = nearest_object.material.color.diffuse * inv_pi

            throughput *= brdf * cos_theta / pdf

            indirect_light = throughput * trace_path(scene, bvh, ray, bounce+1)

            light += (direct_light+indirect_light)


        elif nearest_object.material.is_mirror:
            # mirror reflection
            ray.origin = intersection + EPSILON * surface_normal
            ray.direction = get_reflected_direction(ray.direction, surface_normal)

        elif nearest_object.material.transmission>0.0:
            # compute reflection
            # use Fresnel
            if ray_inside_object:
                n1 = nearest_object.material.ior
                n2 = 1
            else:
                n1 = 1
                n2 = nearest_object.material.ior

            R0 = ((n1 - n2)/(n1 + n2))**2
            theta = np.dot(ray.direction, surface_normal)

            reflection_prob = R0 + (1 - R0) * (1 - np.cos(theta))**5 # Schlick's approximation

            # ----

            Nr = nearest_object.material.ior
            if np.dot(ray.direction, surface_normal)>0:
                Nr = 1/Nr
            Nr = 1/Nr
            cos_theta = -(np.dot(ray.direction, surface_normal))
            _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)

            if _sqrt > 0 and _rand>reflection_prob:
                # refraction
                ray.origin = intersection + (-EPSILON * surface_normal)
                transmit_direction = (ray.direction * Nr) + (surface_normal * (Nr * cos_theta - np.sqrt(_sqrt)))
                ray.direction = normalize(transmit_direction)

            else:
                # reflection
                ray.origin = intersection + EPSILON * surface_normal
                ray.direction = get_reflected_direction(ray.direction, surface_normal)

        else:
            # error
            break

        # terminate path using russian roulette
        if bounce>3:
            r_r = max(0.05, 1-throughput[1]) # russian roulette factor
            if _rand<r_r:
                break
            throughput /= 1-r_r

        bounce += 1

    return light




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
                end = np.array([x, y, scene.f_distance, 1], dtype=np.float64) # pixel
                origin = np.array([scene.camera[0], scene.camera[1], scene.camera[2], 1], dtype=np.float64)
                direction = normalize(end - origin)
                ray = Ray(origin, direction)
                # for k in range(scene.max_depth):
                color += trace_path(scene, bvh, ray, 0)
            color = color/number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)
        pix_count+=1
        print((pix_count/scene.height)*100)
    return scene.image