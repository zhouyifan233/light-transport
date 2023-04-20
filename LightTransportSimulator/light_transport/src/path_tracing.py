import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi, EPSILON, MatType
from .control_variates import calculate_dlogpdu, estimate_alpha
from .light_samples import cast_one_shadow_ray
from .rays import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize


@numba.njit
def trace_path(scene, primitives, bvh, ray, bounce, rand_idx):
    throughput = np.ones((3), dtype=np.float64)
    light = np.zeros((3), dtype=np.float64)
    specular_bounce = False

    while True:
        # terminate path if max bounce is reached
        if bounce>=scene.max_depth:
            break

        rand_0 = scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), bounce]
        rand_1 = scene.rand_1[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), bounce]

        # intersect ray with scene
        nearest_object, min_distance, intersection, surface_normal = hit_object(primitives, bvh, ray)

        # terminate path if no intersection is found
        if nearest_object is None:
            # no object was hit
            for _b in range(bounce, scene.max_depth):
                scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), _b] = np.inf
            break

        # intersection = isect.intersected_point
        # surface_normal = isect.normal

        # add emitted light at intersection
        if bounce==0:
            light += nearest_object.material.emission * throughput

        ray_inside_object = False
        if np.dot(surface_normal, ray.direction) > 0:
            surface_normal = -surface_normal # normal facing opposite direction, hence flipped
            ray_inside_object = True

        # # check if diffuse surface
        # if nearest_object.material.is_diffuse:
        #     shadow_ray_origin = intersection + EPSILON * surface_normal
        #     # direct light contribution
        #     direct_light = cast_one_shadow_ray(scene, primitives, bvh, nearest_object, shadow_ray_origin, surface_normal)
        #
        #     # indirect light contribution
        #     indirect_ray_direction, pdf = cosine_weighted_hemisphere_sampling(surface_normal, ray.direction, [rand_0, rand_1])
        #
        #     if pdf==0:
        #         for _b in range(bounce+1, scene.max_depth):
        #             scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), _b] = np.inf
        #         break
        #
        #     indirect_ray_origin = intersection + EPSILON * indirect_ray_direction
        #
        #     # change ray direction
        #     ray.origin = indirect_ray_origin
        #     ray.direction = indirect_ray_direction
        #
        #     cos_theta = np.dot(indirect_ray_direction, surface_normal)
        #
        #     brdf = nearest_object.material.color.diffuse * inv_pi
        #
        #     throughput *= brdf * cos_theta / pdf
        #
        #     indirect_light = throughput * trace_path(scene, primitives, bvh, ray, bounce+1, rand_idx)
        #
        #     light += (direct_light+indirect_light)
        #
        #
        # elif nearest_object.material.is_mirror:
        #     # mirror reflection
        #     ray.origin = intersection + EPSILON * surface_normal
        #     ray.direction = get_reflected_direction(ray.direction, surface_normal)
        #
        # elif nearest_object.material.transmittance>0.0:
        #     # compute reflection
        #     # use Fresnel
        #     if ray_inside_object:
        #         n1 = nearest_object.material.ior
        #         n2 = 1
        #     else:
        #         n1 = 1
        #         n2 = nearest_object.material.ior
        #
        #     R0 = ((n1 - n2)/(n1 + n2))**2
        #     theta = np.dot(ray.direction, surface_normal)
        #
        #     reflection_prob = R0 + (1 - R0) * (1 - np.cos(theta))**5 # Schlick's approximation
        #
        #     # ----
        #
        #     Nr = nearest_object.material.ior
        #     if np.dot(ray.direction, surface_normal)>0:
        #         Nr = 1/Nr
        #     Nr = 1/Nr
        #     cos_theta = -(np.dot(ray.direction, surface_normal))
        #     _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)
        #
        #     if _sqrt > 0 and rand_0>reflection_prob:
        #         # refraction
        #         ray.origin = intersection + (-EPSILON * surface_normal)
        #         transmit_direction = (ray.direction * Nr) + (surface_normal * (Nr * cos_theta - np.sqrt(_sqrt)))
        #         ray.direction = normalize(transmit_direction)
        #
        #     else:
        #         # reflection
        #         ray.origin = intersection + EPSILON * surface_normal
        #         ray.direction = get_reflected_direction(ray.direction, surface_normal)
        #
        # else:
        #     # error
        #     break
        shadow_ray_origin = intersection + EPSILON * surface_normal
        direct_light = ZEROS #cast_one_shadow_ray(scene, primitives, bvh, nearest_object, shadow_ray_origin, surface_normal)
        if nearest_object.material.type==MatType.DIFFUSE.value:
            # diffuse surface
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object, surface_normal, ray, [rand_0, rand_1])
        elif nearest_object.material.type==MatType.MIRROR.value:
            # perfect mirror reflection
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object, surface_normal, ray, [rand_0, rand_1])
        elif nearest_object.material.type==MatType.SPECULAR.value:
            # specular reflection (only dielectric materials)
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object, surface_normal, ray, [rand_0, rand_1])
        else:
            # error in material metadata
            break

        if pdf_fwd==0:
            break

        throughput *= nearest_object.material.color.diffuse * brdf * np.abs(np.dot(new_ray_direction, surface_normal)) / pdf_fwd
        indirect_light = throughput * trace_path(scene, primitives, bvh, ray, bounce+1, rand_idx)
        light+=direct_light+indirect_light

        # change ray origin
        if intr_type == Medium.DIFFUSE.value or intr_type == Medium.MIRROR.value or intr_type == Medium.REFLECTION.value:
            ray.origin = intersection + EPSILON * new_ray_direction
        else:
            ray.origin = intersection + (-EPSILON * new_ray_direction)

        # change ray direction
        ray.direction = new_ray_direction

        # terminate path using russian roulette
        if np.max(throughput)<1 & bounce>3:
            r_r = max(0.05, 1-np.max(throughput))
            if rand_0<r_r:
                break
            throughput /= 1-r_r
        # if bounce>3:
        #     r_r = max(0.05, 1-throughput[1]) # russian roulette factor
        #     if rand_0<r_r:
        #         for _b in range(bounce+1, scene.max_depth):
        #             scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), _b] = np.inf
        #         break
        #     throughput /= 1-r_r

        bounce += 1

    return light




@numba.njit(parallel=True)
def render_scene(scene, primitives, bvh):
    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    pix_count = 0
    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            color = np.zeros((3), dtype=np.float64)
            for _sample in range(scene.number_of_samples):
                rand = [scene.rand_0[i, j, _sample], scene.rand_1[i, j, _sample]]
                rand_idx = [i, j, _sample]
                dlogpdu = calculate_dlogpdu(rand)
                x = left_right[j]
                # screen is on origin
                end = np.array([x, y, scene.f_distance, 1], dtype=np.float64) # pixel
                # anti-aliasing
                end[0] += rand[0][0]/scene.width
                end[1] += rand[0][0]/scene.height
                cam = scene.camera.position
                origin = np.array([cam[0], cam[1], cam[2], 1], dtype=np.float64)
                direction = normalize(end - origin)
                ray = Ray(origin, direction)
                # for k in range(scene.max_depth):
                color += trace_path(scene, primitives, bvh, ray, 0, rand_idx)
                alpha = estimate_alpha(color)
            color = color/scene.number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)
        pix_count+=1
        print((pix_count/scene.height)*100)
    return scene.image