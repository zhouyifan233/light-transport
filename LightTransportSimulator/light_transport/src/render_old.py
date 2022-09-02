import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .primitives import ShapeOptions
from .ray import Ray
from .utils import nearest_intersected_object
from .vectors import normalize


@numba.njit
def render(scene, bvh):
    for i, y in enumerate(np.linspace(scene.top, scene.bottom, scene.height)):
        for j, x in enumerate(np.linspace(scene.left, scene.right, scene.width)):
            # screen is on origin
            pixel = np.array([x, y, scene.depth], dtype=np.float64)
            origin = scene.camera
            end = pixel
            # direction = normalize(end - origin)
            ray = Ray(origin, end)

            color = np.zeros((3), dtype=np.float64)
            reflection = 1.0

            for k in range(scene.max_depth):
                # check for intersections
                objects = traverse_bvh(bvh, ray.origin, ray.direction)
                nearest_object, min_distance = nearest_intersected_object(objects, ray.origin, ray.direction)

                if nearest_object is None:
                    break

                intersection = ray.origin + min_distance * ray.direction

                # if nearest_object.type == ShapeOptions.SPHERE.value:
                #     normal_to_surface = normalize(intersection - nearest_object.center)
                #     shifted_point = intersection + 1e-5 * normal_to_surface
                if nearest_object.type == ShapeOptions.TRIANGLEPC.value:
                    normal_to_surface = nearest_object.normal
                    shifted_point = intersection
                # elif nearest_object.type == ShapeOptions.PLANE.value:
                #     normal_to_surface = normalize(nearest_object.normal)
                #     shifted_point = intersection
                else:
                    break

                # if np.dot(normal_to_surface, direction) > 0:
                #     normal_to_surface = -normal_to_surface # normal facing opposite direction, hence flipped

                # shifted_point = intersection + 1e-5 * normal_to_surface

                # intersection_to_light = normalize(scene.lights[0].source - shifted_point)

                shadow_ray = Ray(shifted_point, scene.lights[0].source)
                intersection_to_light = shadow_ray.direction
                # print(nearest_intersected_object(objects, shifted_point, scene.lights[0].source))
                _, min_distance = nearest_intersected_object(objects, shadow_ray.origin, shadow_ray.direction)
                # print("shadow")
                # intersection_to_light_distance = np.linalg.norm(scene.lights[0].source - intersection)

                intersection_to_light_distance = shadow_ray.magnitude
                is_shadowed = min_distance < intersection_to_light_distance
                # print("is shadowed")

                illumination = np.zeros((3), dtype=np.float64)

                # ambient
                illumination += get_ambience(nearest_object.material.color.ambient, scene.lights[0].material.color.ambient)

                if is_shadowed:
                    # only ambient color
                    color += reflection * illumination
                    break

                # print("ambient")

                # diffuse
                illumination += get_diffuse(nearest_object.material.color.diffuse, scene.lights[0].material.color.diffuse, intersection_to_light, normal_to_surface)

                # print("diffuse")

                # specular
                intersection_to_camera = normalize(scene.camera - intersection)
                viewing_direction = normalize(intersection_to_light + intersection_to_camera)
                illumination += get_specular(nearest_object.material.color.specular, scene.lights[0].material.color.specular, viewing_direction, normal_to_surface, nearest_object.material.shininess)

                # print("specular")

                # reflection
                color += reflection * illumination

                reflection *= nearest_object.material.reflection

                ray.origin = shifted_point
                ray.direction = reflected_ray(ray.direction, normal_to_surface)

            scene.image[i, j] = np.clip(color, 0, 1)
        print(i+1)
    return scene.image