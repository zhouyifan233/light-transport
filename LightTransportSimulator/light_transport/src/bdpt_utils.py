import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.bvh_new import intersect_bvh
from LightTransportSimulator.light_transport.src.constants import Medium, inv_pi, ZEROS
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.utils import get_cosine_hemisphere_pdf, \
    cosine_weighted_hemisphere_sampling
from LightTransportSimulator.light_transport.src.vectors import normalize
from LightTransportSimulator.light_transport.src.vertex import Vertex, create_light_vertex, create_camera_vertex, \
    convert_density, is_on_surface


@numba.njit
def get_camera_pdf(screen_area, cos_theta):
    '''
    computes spatial and directional pdfs of a camera
    :param screen_area: area of the screen
    :param cos_theta: dot product of ray direction and camera normal
    :return: spatial and directional pdfs of a camera
    '''
    if cos_theta<=0:
        return 0, 0
    lens_area = 1
    pdf_pos = 1/lens_area
    pdf_dir = 1/(screen_area*(cos_theta**3))
    return pdf_pos, pdf_dir


@numba.njit
def get_bsdf_pdf(wp, wn):
    if wp is None:
        return get_cosine_hemisphere_pdf(np.abs(wn[2]))
    elif wp[2]*wn[2]>0:
        return get_cosine_hemisphere_pdf(np.abs(wn[2]))
    else:
        return 0


@numba.njit
def get_light_pdf(curr_v, next_v):
    w = normalize(next_v.point-curr_v.point)
    dist = np.linalg.norm(next_v.point-curr_v.point)
    inv_dist2 = 1/dist**2

    # only for area lights
    pdf_dir = get_cosine_hemisphere_pdf(np.dot(curr_v.g_norm, w))
    pdf = pdf_dir*inv_dist2

    if next_v.medium==Medium.SURFACE.value:
        pdf *= np.abs(np.dot(next_v.g_norm, w))

    return pdf


@numba.njit
def get_light_origin_pdf(scene, curr_v, next_v):
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]
    light_choice_pdf = 1 # 1/no_of_lights
    light_pdf_pos = 1/light.total_area
    light_pdf = light_choice_pdf * light_pdf_pos
    return light_pdf


@numba.njit
def get_pdf(scene, pre_v, curr_v, next_v):
    if curr_v.medium == Medium.LIGHT.value:
        return get_light_pdf(curr_v, next_v)

    # Compute directions to preceding and next vertex
    wn = normalize(next_v.point-curr_v.point)
    if pre_v is not None and pre_v.medium!=Medium.NONE.value:
        wp = normalize(pre_v.point-curr_v.point)

    # Compute directional density depending on the vertex type
    if curr_v.medium==Medium.CAMERA.value:
        cos_theta = np.dot(wn, scene.camera.normal)
        pdf_pos, pdf_dir = get_camera_pdf(scene.camera.screen_area, cos_theta)
        return pdf_dir
    elif curr_v.medium==Medium.SURFACE.value:
        pdf = get_bsdf_pdf(wp, wn)
        pdf = convert_density(pdf, next_v, curr_v)
        return pdf


@numba.njit
def sample_to_add_light(scene, primitives, bvh, vertices):
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    sampled_light = scene.lights[random_light_index]

    curr_v = vertices[-1]
    # check if the light is visible from the current vertex
    new_path_direction = normalize(sampled_light.source - curr_v.point)
    new_path = Ray(curr_v.point, new_path_direction)
    new_path_magnitude = np.linalg.norm(sampled_light.source - curr_v.point)

    next_v = create_light_vertex(sampled_light, new_path_direction, new_path_magnitude, 0, 0)

    pdf = get_light_pdf(curr_v, next_v)
    light_choice_pdf = 1

    if pdf>0:
        # set other attributes
        next_v.pdf_dir = pdf
        next_v.color = sampled_light.material.color.diffuse
        # emission from the light source
        next_v.throughput = (sampled_light.material.emission/(pdf*light_choice_pdf))
        next_v.pdf_pos = 1/sampled_light.total_area
        next_v.pdf_fwd = light_choice_pdf*next_v.pdf_pos



        _, min_distance = intersect_bvh(new_path, primitives, bvh)

        if min_distance is None or min_distance >= new_path_magnitude:
            # light is visible from current path
            next_v.throughput *=  next_v.throughput * curr_v.throughput # add the current path throughput

        if is_on_surface(curr_v):
            next_v.throughput *= np.abs(np.dot(new_path_direction, curr_v.g_norm))

    # append the light vertex to the current path
    vertices.append(next_v)

    return vertices