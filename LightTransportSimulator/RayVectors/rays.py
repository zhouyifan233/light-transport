import numpy as np
from numba import jit

@jit(nopython=True)
def reflected_ray(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

@jit(nopython=True)
def get_ambience(amb_obj, amb_light):
    """
    illumination model for ambient BRDF
    :param amb_obj: ambient coefficient of an object
    :param amb_light: ambient intensity of a light source
    :return: illumination factor for ambience
    """
    return amb_obj * amb_light

@jit(nopython=True)
def get_diffuse(diff_obj, diff_light, light_vec, surf_norm, shiny_fact=1):
    """
    illumination model for diffuse BRDF
    :param diff_obj: diffuse coefficient of an abject
    :param diff_light: diffuse coefficient of a light source
    :param light_vec: a vector pointing towards light
    :param surf_norm: surface normal
    :param shiny_fact: shininess factor
    :return: illumination factor for diffuse
    """
    return diff_obj * diff_light * (np.dot(light_vec, surf_norm)**shiny_fact)

@jit(nopython=True)
def get_specular(spec_obj, spec_light, view_dir, surf_norm, shiny_fact):
    """
    illumination model for specular BRDF
    :param spec_obj: specular coefficient of an object
    :param spec_light: spec
    :param view_dir:
    :param surf_norm:
    :param shiny_fact:
    :return: illumination factor for specular
    """
    return spec_obj * spec_light * (np.dot(surf_norm, view_dir) ** (shiny_fact / 4))