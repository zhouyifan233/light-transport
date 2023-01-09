import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.bvh import traverse_bvh
from LightTransportSimulator.light_transport.src.constants import inv_pi, EPSILON
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.scene import Light
from LightTransportSimulator.light_transport.src.utils import nearest_intersected_object, uniform_hemisphere_sampling, cosine_weighted_hemisphere_sampling
from LightTransportSimulator.light_transport.src.vectors import normalize



def generate_area_light_samples(tri_1, tri_2, source_mat, number_of_samples, total_area):
    light_sources = numba.typed.List()

    light_samples = number_of_samples
    a = np.random.uniform(0,1,light_samples)
    b = np.random.uniform(1,0,light_samples)

    for x in range(light_samples):
        tp1 = tri_1.vertex_1 * (1-math.sqrt(a[x])) + tri_1.vertex_2 * (math.sqrt(a[x])*(1-b[x])) + tri_1.vertex_3 * (b[x]*math.sqrt(a[x]))
        l1 = Light(source=tp1, material=source_mat, normal=tri_1.normal, total_area=total_area)
        light_sources.append(l1)
        tp2 = tri_2.vertex_1 * (1-math.sqrt(a[x])) + tri_2.vertex_2 * (math.sqrt(a[x])*(1-b[x])) + tri_2.vertex_3 * (b[x]*math.sqrt(a[x]))
        l2 = Light(source=tp1, material=source_mat, normal=tri_2.normal, total_area=total_area)
        light_sources.append(l2)

    return light_sources


@numba.njit
def cast_one_shadow_ray(scene, bvh, intersected_object, intersection_point, intersection_normal):
    light_contrib = np.zeros((3), dtype=np.float64)
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]

    shadow_ray_direction = normalize(light.source - intersection_point)
    shadow_ray_magnitude = np.linalg.norm(light.source - intersection_point)

    _objects = traverse_bvh(bvh, intersection_point, shadow_ray_direction)
    _, min_distance = nearest_intersected_object(_objects, intersection_point, shadow_ray_direction, t1=shadow_ray_magnitude)

    if min_distance is None:
        return light_contrib # black background- unlikely

    visible = min_distance >= shadow_ray_magnitude
    if visible:
        brdf = (light.material.emission * light.material.color.diffuse) * (intersected_object.material.color.diffuse * inv_pi)
        cos_theta = np.dot(intersection_normal, shadow_ray_direction)
        cos_phi = np.dot(light.normal, -shadow_ray_direction)
        geometry_term = np.abs(cos_theta * cos_phi)/(shadow_ray_magnitude * shadow_ray_magnitude)
        light_contrib += brdf * geometry_term * light.total_area

    return light_contrib



@numba.njit
def sample_light(scene):
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]

    light_ray_direction, pdf_dir = cosine_weighted_hemisphere_sampling(light.normal)
    light_ray_origin = light.source + EPSILON * light_ray_direction
    light_ray = Ray(light_ray_origin, light_ray_direction)

    light_ray.pdf_pos = 1/light.total_area
    light_ray.pdf_dir = pdf_dir

    light_ray.fwd_pdf = light_ray.rev_pdf = pdf_dir # TODO: Fix

    light_ray.g_norm = light.normal

    light_pdf = 1 # TODO: Fix using Halton Index

    light_ray.throughput = (light.material.emission*abs(np.dot(light_ray.g_norm, light_ray.direction)))/(light_pdf*light_ray.pdf_pos*light_ray.pdf_dir)

    return light_ray


@numba.njit
def cast_all_shadow_rays(scene, bvh, intersected_object, intersection_point, intersection_normal):
    light_contrib = np.zeros((3), dtype=np.float64)
    for light in scene.lights:
        shadow_ray_direction = normalize(light.source - intersection_point)
        shadow_ray_magnitude = np.linalg.norm(light.source - intersection_point)

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


