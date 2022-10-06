import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .primitives import ShapeOptions
from .ray import Ray
from .utils import nearest_intersected_object
from .vectors import normalize


# from LightTransportSimulator.light_transport.src.ray import Ray
# from LightTransportSimulator.light_transport.src.utils import nearest_intersected_object
# from LightTransportSimulator.light_transport.src.brdf import *
# from LightTransportSimulator.light_transport.src.bvh import traverse_bvh


@numba.njit
def hit_object(bvh, ray_origin, ray_direction):
    # get hittable objects
    objects = traverse_bvh(bvh, ray_origin, ray_direction)
    # check for intersections
    nearest_object, min_distance = nearest_intersected_object(objects, ray_origin, ray_direction)

    if nearest_object is None:
        # no object was hit
        return None, None, None, None

    intersection = ray_origin + min_distance * ray_direction
    surface_normal = nearest_object.normal

    return nearest_object, min_distance, intersection, surface_normal



@numba.njit
def trace_ray(scene, bvh, ray_origin, ray_direction, depth):
    # set the defaults
    color = np.zeros((3), dtype=np.float64)
    reflection = 1.0

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

    # compute shadow
    shifted_point = intersection + 1e-5 * surface_normal

    shadow_color = np.zeros((3), dtype=np.float64)
    diffuse_color = shadow_color

    for light in scene.lights:
        # cast a shadow ray
        shadow_ray = Ray(shifted_point, light.source)
        intersection_to_light = shadow_ray.direction

        # get all the objects shadow ray could intersect
        _objects = traverse_bvh(bvh, shadow_ray.origin, shadow_ray.direction)
        _, min_distance = nearest_intersected_object(_objects, shadow_ray.origin, shadow_ray.direction, t1=shadow_ray.magnitude)
        # _, min_distance, _, _ = hit_object(bvh, shadow_ray, shadow_ray.magnitude)

        if min_distance is None:
            print('error!')
            break

        # check if there is an object between intersection and light source
        is_shadowed = min_distance < shadow_ray.magnitude

        illumination = np.zeros((3), dtype=np.float64)

        # ambient
        illumination += get_ambience(nearest_object.material.color.ambient, light.material.color.ambient)

        if is_shadowed:
            # only ambient color
            shadow_color += reflection * illumination
            continue

        # diffuse
        diffuse_color = get_diffuse(nearest_object.material.color.diffuse, light.material.color.diffuse, intersection_to_light, surface_normal)
        illumination += diffuse_color

        # specular
        intersection_to_camera = normalize(scene.camera - intersection)
        viewing_direction = normalize(intersection_to_light + intersection_to_camera)
        illumination += get_specular(nearest_object.material.color.specular, light.material.color.specular, viewing_direction, surface_normal, nearest_object.material.shininess)

        # reflection
        shadow_color += reflection * illumination


    # # color contribution from shadow
    color += shadow_color/len(scene.lights)
    diffuse_color = diffuse_color/len(scene.lights)

    # calculate reflectivity of the object
    if nearest_object.material.is_mirror:
        # use mirror reflection
        reflection *= nearest_object.material.reflection
    else:
        # use Fresnel
        if ray_inside_object:
            n1 = nearest_object.material.ior
            n2 = 1
        else:
            n1 = 1
            n2 = nearest_object.material.ior

        R0 = ((n1 - n2)/(n1 + n2))**2
        _angle = np.dot(ray_direction, surface_normal)
        reflection *= R0 + (1 - R0) * (1 - np.cos(_angle))**5

    if depth>0:
        # compute reflection
        reflect_origin = shifted_point
        reflect_direction = reflected_ray(ray_direction, surface_normal)
        reflect_direction = normalize(reflect_direction)
        # color contribution from reflection
        reflected_color = trace_ray(scene, bvh, reflect_origin, reflect_direction, depth-1)
        reflected_color *= reflection
        color+=reflected_color


    if depth > 0 and nearest_object.material.transmission > 0:
        # compute refraction
        Nr = nearest_object.material.ior
        norm_surface = surface_normal
        if np.dot(ray_direction, surface_normal)>0:
            norm_surface = -norm_surface
            Nr = 1/Nr
        Nr = 1/Nr
        cos_theta = -(np.dot(ray_direction, surface_normal))
        _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)

        if _sqrt > 0: # no transmitted ray if negative
            transmit_origin = intersection + (-0.001 * norm_surface)

            transmit_direction = (ray_direction * Nr) + (norm_surface * (Nr * cos_theta - math.sqrt(_sqrt)))
            transmit_direction = normalize(transmit_direction)
            transmit_color = trace_ray(scene, bvh, transmit_origin, transmit_direction, depth-1)

            color += transmit_color * (1 - reflection) * nearest_object.material.transmission # transmission factor for glass is 1

    if depth > 0:
        # print(surface_normal)
        z = surface_normal[:3]
        x = np.array([0, 0, 1], dtype=np.float64)

        xdotz = np.dot(x, z)
        if xdotz > 0.9 or xdotz < -0.9:
            x = np.array([0, 1, 0], dtype=np.float64)
        # Gram Schmit
        x = x - (np.dot(x, z) * z)
        x = normalize(x)
        y = np.cross(z, x)

        indirect_diffuse_color = np.zeros((3), dtype=np.float64)
        number_of_samples = 1

        for _i in range(number_of_samples):
            # random sample point on hemisphere
            r1 = np.random.rand() # uniformly sampled from 0 to 1
            r2 = np.random.rand()

            theta= math.sqrt(1.0-r1*r2)
            phi = 2 * np.pi * r2
            _point = [np.cos(phi)*theta, np.sin(phi)*theta, r1]
            random_point = np.array(_point, dtype=np.float64)

            transformation_matrix = np.array([[x[0], x[1], x[2]], [y[0], y[1], y[2]], [z[0], z[1], z[2]]], dtype=np.float64)
            global_ray_dir = transformation_matrix @ random_point
            global_ray_dir = np.array([global_ray_dir[0], global_ray_dir[1], global_ray_dir[2], 0], dtype=np.float64)

            raw_intensity = trace_ray(scene, bvh, intersection, global_ray_dir, depth-1)
            indirect_diffuse_color += raw_intensity * 0.1 * diffuse_color
        color += indirect_diffuse_color/number_of_samples


    # Finally
    return color



@numba.njit(parallel=True)
def render_scene(scene, bvh):
    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            x = left_right[j]
            # screen is on origin
            pixel = np.array([x, y, scene.depth], dtype=np.float64)
            origin = scene.camera
            end = pixel
            # direction = normalize(end - origin)
            ray = Ray(origin, end)
            # for k in range(scene.max_depth):
            color = trace_ray(scene, bvh, ray.origin, ray.direction, scene.max_depth)

            scene.image[i, j] = np.clip(color, 0, 1)
        print(i+1)
    return scene.image