import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .primitives import ShapeOptions
from .ray import Ray
from .utils import nearest_intersected_object
from .vectors import normalize


@numba.njit
def render_scene(scene, bvh):
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
                #     surface_normal = normalize(intersection - nearest_object.center)
                #     shifted_point = intersection + 1e-5 * surface_normal
                if nearest_object.type == ShapeOptions.TRIANGLEPC.value:
                    surface_normal = nearest_object.normal
                    shifted_point = intersection + 1e-5 * surface_normal
                # elif nearest_object.type == ShapeOptions.PLANE.value:
                #     surface_normal = normalize(nearest_object.normal)
                #     shifted_point = intersection
                else:
                    break

                ray_inside_object = False
                if np.dot(surface_normal, ray.direction) > 0:
                    surface_normal = -surface_normal # normal facing opposite direction, hence flipped
                    ray_inside_object = True

                # shifted_point = intersection + 1e-5 * surface_normal

                # intersection_to_light = normalize(scene.lights[0].source - shifted_point)
                _color = np.zeros((3), dtype=np.float64)

                for light in scene.lights:
                    shadow_ray = Ray(shifted_point, light.source)
                    intersection_to_light = shadow_ray.direction
                    # print(nearest_intersected_object(objects, shifted_point, scene.lights[0].source))
                    _, min_distance = nearest_intersected_object(objects, shadow_ray.origin, shadow_ray.direction, t1=shadow_ray.magnitude)
                    # print("shadow")
                    # intersection_to_light_distance = np.linalg.norm(scene.lights[0].source - intersection)

                    intersection_to_light_distance = shadow_ray.magnitude
                    is_shadowed = min_distance < intersection_to_light_distance
                    # print("is shadowed")

                    illumination = np.zeros((3), dtype=np.float64)

                    # ambient
                    illumination += get_ambience(nearest_object.material.color.ambient, light.material.color.ambient)

                    if is_shadowed:
                        # only ambient color
                        _color += reflection * illumination
                        continue

                    # print("ambient")

                    # diffuse
                    illumination += get_diffuse(nearest_object.material.color.diffuse, light.material.color.diffuse, intersection_to_light, surface_normal)

                    # print("diffuse")

                    # specular
                    intersection_to_camera = normalize(scene.camera - intersection)
                    viewing_direction = normalize(intersection_to_light + intersection_to_camera)
                    illumination += get_specular(nearest_object.material.color.specular, light.material.color.specular, viewing_direction, surface_normal, nearest_object.material.shininess)

                    # print("specular")

                    # reflection
                    _color += reflection * illumination

                color += _color/len(scene.lights)

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
                    _angle = np.dot(ray.direction, surface_normal)
                    reflection *= R0 + (1 - R0) * (1 - np.cos(_angle))**5

                ray.origin = shifted_point
                ray.direction = reflected_ray(ray.direction, surface_normal)

            scene.image[i, j] = np.clip(color, 0, 1)
        print(i+1)
    return scene.image