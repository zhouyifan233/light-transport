import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.constants import Medium, inv_pi, ZEROS
from LightTransportSimulator.light_transport.src.utils import get_cosine_hemisphere_pdf, \
    cosine_weighted_hemisphere_sampling
from LightTransportSimulator.light_transport.src.vectors import normalize


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
def convert_density(pdf, prev_v, next_v):
    '''
    converts pdf to solid angle density
    :param pdf: pdf
    :param prev_v: previous vertex
    :param next_v: next or current vertex
    :return: solid angle density
    '''
    path = normalize(next_v.point - prev_v.point)
    path_magnitude = np.linalg.norm(next_v.point - prev_v.point)
    if path_magnitude==0:
        return 0
    inv_dist_sqr = 1/(path_magnitude*path_magnitude)
    if next_v.medium==Medium.SURFACE.value:
        pdf *= np.abs(np.dot(next_v.g_norm, path*np.sqrt(inv_dist_sqr)))
    return pdf * inv_dist_sqr


@numba.njit
def get_bsdf_pdf(wp, wn):
    if wp[2]*wn[2]>0:
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
    if pre_v.medium!=Medium.NONE.value:
        wp = normalize(pre_v.point-curr_v.point)

    # Compute directional density depending on the vertex type
    if curr_v.medium==Medium.CAMERA.value:
        cos_theta = np.dot(wn, scene.camera.normal)
        return get_camera_pdf(scene.camera.screen_area, cos_theta)
    elif curr_v.medium==Medium.SURFACE.value:
        pdf = get_bsdf_pdf(wp, wn)
        pdf = convert_density(pdf, next_v, curr_v)
        return pdf


