import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .primitives import ShapeOptions
from .ray_old import Ray
from .utils import nearest_intersected_object
from .vectors import normalize


# from LightTransportSimulator.light_transport.src.ray import Ray
# from LightTransportSimulator.light_transport.src.utils import nearest_intersected_object
# from LightTransportSimulator.light_transport.src.brdf import *
# from LightTransportSimulator.light_transport.src.bvh import traverse_bvh

@numba.njit
def random_unit_vector_from_hemisphere(normal_at_intersection):

    # random sample point on hemisphere
    r1 = np.random.rand()
    r2 = np.random.rand()

    theta = math.sqrt(max((0.0, 1.0-r1**2)))
    phi = 2 * np.pi * r2

    _point = [theta * np.cos(phi), theta * np.sin(phi), r1]
    random_point = np.array(_point, dtype=np.float64)


    if abs(normal_at_intersection[0]) > abs(normal_at_intersection[1]):
        inv_len = 1.0 / math.sqrt(normal_at_intersection[0]**2 + normal_at_intersection[2]**2)
        v2 = np.array([-normal_at_intersection[2] * inv_len, 0.0, normal_at_intersection[0] * inv_len], dtype=np.float64)
    else:
        inv_len = 1.0 / math.sqrt(normal_at_intersection[1]**2 + normal_at_intersection[2]**2)
        v2 = np.array([0.0, normal_at_intersection[2] * inv_len, -normal_at_intersection[1] * inv_len], dtype=np.float64)

    v3 = np.cross(normal_at_intersection[:-1], v2)

    rot_x = np.dot(np.array([v2[0], v3[0], normal_at_intersection[0]], dtype=np.float64), random_point)
    rot_y = np.dot(np.array([v2[1], v3[1], normal_at_intersection[1]], dtype=np.float64), random_point)
    rot_z = np.dot(np.array([v2[2], v3[2], normal_at_intersection[2]], dtype=np.float64), random_point)

    global_ray_dir = np.array([rot_x, rot_y, rot_z, 0], dtype=np.float64)

    return global_ray_dir


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
        if np.dot(ray_direction, surface_normal)>0:
            # surface_normal = -surface_normal
            Nr = 1/Nr
        Nr = 1/Nr
        cos_theta = -(np.dot(ray_direction, surface_normal))
        _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)

        if _sqrt > 0: # no transmitted ray if negative
            transmit_origin = intersection + (-0.001 * surface_normal)

            transmit_direction = (ray_direction * Nr) + (surface_normal * (Nr * cos_theta - math.sqrt(_sqrt)))
            transmit_direction = normalize(transmit_direction)
            transmit_color = trace_ray(scene, bvh, transmit_origin, transmit_direction, depth-1)

            color += transmit_color * (1 - reflection) * nearest_object.material.transmission # transmission factor for glass is 1

    if depth > 0:
        indirect_diffuse_color = np.zeros((3), dtype=np.float64)
        number_of_samples = 10
        for _i in range(number_of_samples):
            global_ray_dir = random_unit_vector_from_hemisphere(surface_normal)
            cos_theta = np.dot(global_ray_dir, surface_normal)
            raw_intensity = trace_ray(scene, bvh, intersection, global_ray_dir, depth-1)
            indirect_diffuse_color += (nearest_object.material.color.diffuse*raw_intensity)*cos_theta*0.1
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