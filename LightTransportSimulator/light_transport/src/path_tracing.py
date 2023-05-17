import math
import logging
import numba
import numpy as np
import scipy as sp
import copy

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi, EPSILON
from .control_variates import calculate_dlogpdu, estimate_alpha
from .light_samples import cast_one_shadow_ray
from .rays import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize


@numba.njit
def sigmoid(x):
    return 1 / (1 + np.exp(x))

@numba.njit
def trace_path(scene, primitives, bvh, ray, bounce, rand_0_set, rand_1_set, info=(0, 0, 0), calculate_grad=False):
    throughput = np.ones((3), dtype=np.float64)
    light = np.zeros((3), dtype=np.float64)
    record_log_pdf = np.zeros((scene.max_depth), dtype=np.float64)
    specular_bounce = False
    direct_light_list = np.zeros((3, scene.max_depth), dtype=np.float64)
    indirect_light_list = np.zeros((3, scene.max_depth), dtype=np.float64)

    while True:
        rand_0 = rand_0_set[bounce]
        rand_1 = rand_1_set[bounce]

        # terminate path if max bounce is reached
        if calculate_grad is True:
            if scene.bounce_record[int(info[0]), int(info[1]), int(info[2]), int(bounce)] == 0:
                break

        if bounce >= scene.max_depth:
            break

        # intersect ray with scene
        nearest_object, min_distance, intersection, surface_normal = hit_object(primitives, bvh, ray)

        # terminate path if no intersection is found
        if nearest_object is None:
            # no object was hit
            if calculate_grad is False:
                for _b in range(bounce, scene.max_depth):
                    scene.bounce_record[int(info[0]), int(info[1]), int(info[2]), int(_b)] = 0
            break

        # intersection = isect.intersected_point
        # surface_normal = isect.normal

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
            direct_light = cast_one_shadow_ray(scene, primitives, bvh, nearest_object, shadow_ray_origin, surface_normal)

            # indirect light contribution
            indirect_ray_direction, pdf = cosine_weighted_hemisphere_sampling(surface_normal, ray.direction, [rand_0, rand_1])

            if pdf == 0:
                if calculate_grad is False:
                    for _b in range(bounce+1, scene.max_depth):
                        scene.bounce_record[int(info[0]), int(info[1]), int(info[2]), int(_b)] = 0
                break

            indirect_ray_origin = intersection + EPSILON * indirect_ray_direction

            # change ray direction
            ray.origin = indirect_ray_origin
            ray.direction = indirect_ray_direction

            cos_theta = np.dot(indirect_ray_direction, surface_normal)

            brdf = nearest_object.material.color.diffuse * inv_pi

            throughput *= brdf * cos_theta / pdf
            tp, record_log_pdf, _ = trace_path(scene, primitives, bvh, ray, bounce+1, rand_0_set, rand_1_set, info=info, calculate_grad=calculate_grad)
            record_log_pdf[bounce] = np.log(pdf)
            if calculate_grad is False:
                scene.record_log_pdf[int(info[0]), int(info[1]), int(info[2]), int(bounce)] = np.log(pdf)

            indirect_light = throughput * tp
            direct_light_list[:, bounce] = direct_light
            indirect_light_list[:, bounce] = indirect_light
            light += (direct_light + indirect_light)

        elif nearest_object.material.is_mirror:
            # mirror reflection
            ray.origin = intersection + EPSILON * surface_normal
            ray.direction = get_reflected_direction(ray.direction, surface_normal)

        elif nearest_object.material.transmission > 0.0:
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

            if _sqrt > 0 and rand_0>reflection_prob:
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
        if bounce > 3:
            r_r = max(0.05, 1-throughput[1]) # russian roulette factor
            if rand_0 < r_r:
                if calculate_grad is False:
                    for _b in range(bounce+1, scene.max_depth):
                        scene.bounce_record[int(info[0]), int(info[1]), int(info[2]), int(_b)] = 0
                break
            throughput /= 1-r_r

        bounce += 1

    return light, record_log_pdf, (direct_light_list, indirect_light_list)


# @numba.njit
# def calculate_gradients(scene, primitives, bvh, origin, direction, rand_0_set, rand_1_set, info=(0, 0, 0), calculate_grad=True):
#     _s = np.zeros((scene.max_depth*2), dtype=np.float64)
#     _record_log_pdf_minus_set = np.zeros((scene.max_depth * 2, scene.max_depth), dtype=np.float64)
#     _record_log_pdf_plus_set = np.zeros((scene.max_depth * 2, scene.max_depth), dtype=np.float64)
#
#     for rand_i in numba.prange(scene.max_depth):
#         rand_0_set_s_minus = np.copy(rand_0_set)
#         # _s[rand_i] = rand_0_set[rand_i] / 100.0
#         _s[rand_i] = 0.001
#         rand_0_set_s_minus[rand_i] -= _s[rand_i]
#         rand_0_set_s_plus = np.copy(rand_0_set)
#         rand_0_set_s_plus[rand_i] += _s[rand_i]
#         ray = Ray(origin, direction)
#         tp_result, _record_log_pdf_minus = trace_path(scene, primitives, bvh, ray, 0, rand_0_set_s_minus,
#                                                       rand_1_set, info=info, calculate_grad=calculate_grad)
#         ray = Ray(origin, direction)
#         tp_result, _record_log_pdf_plus = trace_path(scene, primitives, bvh, ray, 0, rand_0_set_s_plus,
#                                                      rand_1_set, info=info, calculate_grad=calculate_grad)
#
#         _record_log_pdf_minus_set[rand_i, :] = _record_log_pdf_minus
#         _record_log_pdf_plus_set[rand_i, :] = _record_log_pdf_plus
#
#         rand_1_set_s_minus = np.copy(rand_1_set)
#         # _s[rand_i+scene.max_depth] = rand_1_set[rand_i] / 100.0
#         _s[rand_i + scene.max_depth] = 0.001
#         rand_1_set_s_minus[rand_i] -= _s[rand_i + scene.max_depth]
#         rand_1_set_s_plus = np.copy(rand_1_set)
#         rand_1_set_s_plus[rand_i] += _s[rand_i + scene.max_depth]
#         ray = Ray(origin, direction)
#         tp_result, _record_log_pdf_minus = trace_path(scene, primitives, bvh, ray, 0, rand_0_set,
#                                                       rand_1_set_s_minus, info=info, calculate_grad=calculate_grad)
#         ray = Ray(origin, direction)
#         tp_result, _record_log_pdf_plus = trace_path(scene, primitives, bvh, ray, 0, rand_0_set,
#                                                      rand_1_set_s_plus, info=info, calculate_grad=calculate_grad)
#         _record_log_pdf_minus_set[rand_i+scene.max_depth, :] = _record_log_pdf_minus
#         _record_log_pdf_plus_set[rand_i+scene.max_depth, :] = _record_log_pdf_plus
#
#     return _record_log_pdf_minus_set, _record_log_pdf_plus_set, _s


@numba.njit
def calculate_gradients(scene, primitives, bvh, origin, direction, rand_0_set_logit, rand_1_set_logit, info=(0, 0, 0), calculate_grad=True):
    _s = np.zeros((scene.max_depth*2), dtype=np.float64)
    _record_log_pdf_minus_set = np.zeros((scene.max_depth * 2, scene.max_depth), dtype=np.float64)
    _record_log_pdf_plus_set = np.zeros((scene.max_depth * 2, scene.max_depth), dtype=np.float64)

    for rand_i in numba.prange(scene.max_depth):
        _s[rand_i] = 0.01
        rand_0_set_s_minus_logit = np.copy(rand_0_set_logit)
        rand_0_set_s_minus_logit[rand_i] -= _s[rand_i]
        rand_0_set_s_plus_logit = np.copy(rand_0_set_logit)
        rand_0_set_s_plus_logit[rand_i] += _s[rand_i]

        rand_0_set_s_minus = sigmoid(rand_0_set_s_minus_logit)
        rand_0_set_s_plus = sigmoid(rand_0_set_s_plus_logit)
        rand_1_set = sigmoid(rand_1_set_logit)

        ray = Ray(origin, direction)
        tp_result, _record_log_pdf_minus, _ = trace_path(scene, primitives, bvh, ray, 0, rand_0_set_s_minus,
                                                      rand_1_set, info=info, calculate_grad=calculate_grad)
        ray = Ray(origin, direction)
        tp_result, _record_log_pdf_plus, _ = trace_path(scene, primitives, bvh, ray, 0, rand_0_set_s_plus,
                                                     rand_1_set, info=info, calculate_grad=calculate_grad)

        _record_log_pdf_minus_set[rand_i, :] = _record_log_pdf_minus
        _record_log_pdf_plus_set[rand_i, :] = _record_log_pdf_plus

        _s[rand_i + scene.max_depth] = 0.01
        rand_1_set_s_minus_logit = np.copy(rand_1_set_logit)
        rand_1_set_s_minus_logit[rand_i] -= _s[rand_i + scene.max_depth]
        rand_1_set_s_plus_logit = np.copy(rand_1_set_logit)
        rand_1_set_s_plus_logit[rand_i] += _s[rand_i + scene.max_depth]

        rand_0_set = sigmoid(rand_0_set_logit)
        rand_1_set_s_minus = sigmoid(rand_1_set_s_minus_logit)
        rand_1_set_s_plus = sigmoid(rand_1_set_s_plus_logit)

        ray = Ray(origin, direction)
        tp_result, _record_log_pdf_minus, _ = trace_path(scene, primitives, bvh, ray, 0, rand_0_set,
                                                      rand_1_set_s_minus, info=info, calculate_grad=calculate_grad)
        ray = Ray(origin, direction)
        tp_result, _record_log_pdf_plus, _ = trace_path(scene, primitives, bvh, ray, 0, rand_0_set,
                                                     rand_1_set_s_plus, info=info, calculate_grad=calculate_grad)
        _record_log_pdf_minus_set[rand_i+scene.max_depth, :] = _record_log_pdf_minus
        _record_log_pdf_plus_set[rand_i+scene.max_depth, :] = _record_log_pdf_plus

    return _record_log_pdf_minus_set, _record_log_pdf_plus_set, _s

@numba.njit
def render_scene_samples(scene, primitives, bvh):
    samples = np.zeros((scene.height, scene.width, scene.number_of_samples, 3), dtype=np.float64)
    record_log_pdf = np.zeros((scene.height, scene.width, scene.number_of_samples, scene.max_depth), dtype=np.float64)
    record_log_pdf_plus_set = np.zeros((scene.height, scene.width, scene.number_of_samples, scene.max_depth*2, scene.max_depth), dtype=np.float64)
    record_log_pdf_minus_set = np.zeros((scene.height, scene.width, scene.number_of_samples, scene.max_depth*2, scene.max_depth), dtype=np.float64)
    record_s_set = np.zeros((scene.height, scene.width, scene.number_of_samples, scene.max_depth*2), dtype=np.float64)
    direct_light_list = np.zeros((scene.height, scene.width, scene.number_of_samples, 3, scene.max_depth), dtype=np.float64)
    indirect_light_list = np.zeros((scene.height, scene.width, scene.number_of_samples, 3, scene.max_depth), dtype=np.float64)

    # gradient_pdf = np.zeros((scene.height, scene.width, scene.number_of_samples, 3), dtype=np.float64)

    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    pix_count = 0
    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            color = np.zeros((3), dtype=np.float64)
            for _sample in numba.prange(scene.number_of_samples):
                rand = [scene.rand_0[i, j, _sample], scene.rand_1[i, j, _sample]]
                # rand_idx = [i, j, _sample]

                rand_0_set = scene.rand_0[i, j, _sample]
                rand_1_set = scene.rand_1[i, j, _sample]

                dlogpdu = calculate_dlogpdu(rand)
                x = left_right[j]
                # screen is on origin
                end = np.array([x, y, scene.f_distance, 1], dtype=np.float64) # pixel
                # anti-aliasing
                end[0] += rand[0][0]/scene.width
                end[1] += rand[0][0]/scene.height
                origin = np.array([scene.camera[0], scene.camera[1], scene.camera[2], 1], dtype=np.float64)
                direction = normalize(end - origin)
                ray = Ray(origin, direction)
                # for k in range(scene.max_depth):
                tp_result, _record_log_pdf, tmp_light_list = trace_path(scene, primitives, bvh, ray, 0, rand_0_set, rand_1_set, info=(i, j, _sample))
                direct_light_list[i, j, _sample, :, :] = tmp_light_list[0]
                indirect_light_list[i, j, _sample, :, :] = tmp_light_list[1]

                rand_0_set_logit = scene.rand_0_logit[i, j, _sample]
                rand_1_set_logit = scene.rand_1_logit[i, j, _sample]
                _record_log_pdf_minus_set, _record_log_pdf_plus_set, _s = \
                    calculate_gradients(scene, primitives, bvh, origin, direction, rand_0_set_logit, rand_1_set_logit, info=(i, j, _sample))

                samples[i, j, _sample, :] = tp_result
                color += tp_result
                alpha = estimate_alpha(color)
                record_log_pdf[i, j, _sample, :] = _record_log_pdf
                record_log_pdf_minus_set[i, j, _sample, :, :] = _record_log_pdf_minus_set
                record_log_pdf_plus_set[i, j, _sample, :, :] = _record_log_pdf_plus_set
                record_s_set[i, j, _sample, :] = _s
            color = color/scene.number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)
        # pix_count += 1
        # print((pix_count/scene.height)*100)
        print('Height: ' + str(i) + ' finished...')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # choose some pixels and draw more samples
    more_samples_num = 500
    more_samples_h = [20, 60, 60, 65]
    more_samples_w = [60, 25, 80, 110]
    num_pixels = len(more_samples_h)
    samples_2 = np.zeros((num_pixels, more_samples_num, 3), dtype=np.float64)
    record_log_pdf_2 = np.zeros((num_pixels, more_samples_num, scene.max_depth), dtype=np.float64)
    record_log_pdf_plus_set_2 = np.zeros(
        (num_pixels, more_samples_num, scene.max_depth * 2, scene.max_depth), dtype=np.float64)
    record_log_pdf_minus_set_2 = np.zeros(
        (num_pixels, more_samples_num, scene.max_depth * 2, scene.max_depth), dtype=np.float64)
    record_s_set_2 = np.zeros((num_pixels, more_samples_num, scene.max_depth * 2), dtype=np.float64)

    for _i in numba.prange(num_pixels):
        i = more_samples_h[_i]
        j = more_samples_w[_i]
        y = top_bottom[i]
        rand_0_2 = np.random.rand(num_pixels, more_samples_num, scene.max_depth)
        rand_1_2 = np.random.rand(num_pixels, more_samples_num, scene.max_depth)
        for _sample in numba.prange(more_samples_num):
            rand = [rand_0_2[_i, _sample, :], rand_1_2[_i, _sample, :]]
            # rand_idx = [i, j, _sample]

            rand_0_set = rand_0_2[_i, _sample, :]
            rand_1_set = rand_1_2[_i, _sample, :]

            # dlogpdu = calculate_dlogpdu(rand)
            x = left_right[j]
            # screen is on origin
            end = np.array([x, y, scene.f_distance, 1], dtype=np.float64) # pixel
            # anti-aliasing
            end[0] += rand[0][0]/scene.width
            end[1] += rand[0][0]/scene.height
            origin = np.array([scene.camera[0], scene.camera[1], scene.camera[2], 1], dtype=np.float64)
            direction = normalize(end - origin)
            ray = Ray(origin, direction)
            # for k in range(scene.max_depth):
            tp_result, _record_log_pdf, _ = trace_path(scene, primitives, bvh, ray, 0, rand_0_set, rand_1_set, info=(i, j, _sample))

            _record_log_pdf_minus_set, _record_log_pdf_plus_set, _s = \
                calculate_gradients(scene, primitives, bvh, origin, direction, rand_0_set, rand_1_set, info=(i, j, _sample), calculate_grad=False)

            samples_2[_i, _sample, :] = tp_result
            # color += tp_result
            # alpha = estimate_alpha(color)
            record_log_pdf_2[_i, _sample, :] = _record_log_pdf
            record_log_pdf_minus_set_2[_i, _sample, :, :] = _record_log_pdf_minus_set
            record_log_pdf_plus_set_2[_i, _sample, :, :] = _record_log_pdf_plus_set
            record_s_set_2[_i, _sample, :] = _s
        # color = color/scene.number_of_samples
        # scene.image[i, j] = np.clip(color, 0, 1)
        # pix_count += 1
        # print((pix_count/scene.height)*100)
        print('Height (more samples): ' + str(i) + ' finished...')

    return scene.image, samples, record_log_pdf, record_log_pdf_minus_set, record_log_pdf_plus_set, record_s_set, \
        (samples_2, record_log_pdf_2, record_log_pdf_minus_set_2, record_log_pdf_plus_set_2, record_s_set_2), \
        (direct_light_list, indirect_light_list)


@numba.njit(parallel=True)
def render_scene(scene, primitives, bvh):
    image_ver0 = np.copy(scene.image)
    image_ver1 = np.copy(scene.image)
    image_ver2 = np.copy(scene.image)
    image, samples, record_log_pdf, record_log_pdf_minus_set, record_log_pdf_plus_set, record_s_set, data_more_samples, two_light_list = render_scene_samples(scene, primitives, bvh)

    for i in numba.prange(scene.height):
        for j in numba.prange(scene.width):
            color = np.zeros(3, dtype=np.float64)
            for _sample in numba.prange(scene.number_of_samples):
                color += samples[i, j, _sample]
            color = color / scene.number_of_samples
            image_ver1[i, j] = np.clip(color, 0, 1)

    return image_ver0, image_ver1, image_ver2, samples, record_log_pdf, \
        record_log_pdf_minus_set, record_log_pdf_plus_set, record_s_set, data_more_samples, two_light_list
