import numba
import numpy as np

from .brdf import *
from .primitives import ShapeOptions
from .utils import nearest_intersected_object
from .vectors import normalize


@numba.njit
def render_rt(scene, objects, dimension):
    for i, y in enumerate(np.linspace(scene.top, scene.bottom, scene.height)):
        for j, x in enumerate(np.linspace(scene.left, scene.right, scene.width)):
            # screen is on origin
            pixel = np.array([x, y, dimension], dtype=np.float64)
            origin = scene.camera
            end = pixel
            direction = normalize(end - origin)

            color = np.zeros((3), dtype=np.float64)
            reflection = 1.0

            for k in range(scene.max_depth):
                # check for intersections
                nearest_object, min_distance = nearest_intersected_object(objects, origin, end)

                if nearest_object is None:
                    break

                intersection = origin + min_distance * direction

                # if nearest_object.type == ShapeOptions.SPHERE.value:
                #     normal_to_surface = normalize(intersection - nearest_object.center)
                #     shifted_point = intersection + 1e-5 * normal_to_surface
                if nearest_object.type == ShapeOptions.TRIANGLE.value:
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

                intersection_to_light = normalize(scene.light.source - shifted_point)

                _, min_distance = nearest_intersected_object(objects, shifted_point, scene.light.source)
                intersection_to_light_distance = np.linalg.norm(scene.light.source - intersection)
                is_shadowed = min_distance < intersection_to_light_distance

                if is_shadowed:
                    break

                illumination = np.zeros((3), dtype=np.float64)

                # ambiant
                illumination += get_ambience(nearest_object.material.color.ambient, scene.light.material.color.ambient)

                # diffuse
                illumination += get_diffuse(nearest_object.material.color.diffuse, scene.light.material.color.diffuse, intersection_to_light, normal_to_surface)

                # specular
                intersection_to_camera = normalize(scene.camera - intersection)
                viewing_direction = normalize(intersection_to_light + intersection_to_camera)
                illumination += get_specular(nearest_object.material.color.specular, scene.light.material.color.specular, viewing_direction, normal_to_surface, nearest_object.material.shininess)

                # reflection
                color += reflection * illumination

                reflection *= nearest_object.material.reflection

                origin = shifted_point
                direction = reflected_ray(direction, normal_to_surface)

            scene.image[i, j] = np.clip(color, 0, 1)
        print(i+1)
    return scene.image