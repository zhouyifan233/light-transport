import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi, EPSILON, Medium, ZEROS
from .light_samples import cast_one_shadow_ray, sample_light
from .rays import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize
from .vertex import Vertex


@numba.njit
def random_walk(scene, bvh, ray, max_depth, paths, throughput, pdf_init, type):

    if max_depth==0:
        return paths

    bounces = 0

    # Declare variables for forward and reverse probability densities
    pdf_fwd = pdf_init
    pdf_rev = 0

    while True:
        # attempt to find next vertex on the path

        _rand = np.random.rand()

        vx = paths[bounces] # current vertex
        prev = paths[bounces-1] # previous vertex

        # intersect ray with scene
        nearest_object, min_distance, intersection, surface_normal = hit_object(bvh, ray.origin, ray.direction)

        #TODO: Check for medium interaction: Volumetric Rendering (smoke, gas, fog, etc.)

        # terminate path if no intersection is found
        if nearest_object is None:
            # no object was hit
            break

        # check if the intersected object is a light
        if nearest_object.is_light:
            # create a light vertex
            # light += nearest_object.material.emission * throughput
            bounces += 1
            break # path ends here

        # compute scattering distribution
        #TODO: create a surface vertex
        bounces += 1

        if bounces>=max_depth:
            break


        ray_inside_object = False
        if np.dot(surface_normal, ray.direction) > 0:
            surface_normal = -surface_normal # normal facing opposite direction, hence flipped
            ray_inside_object = True

        # check if diffuse surface, i.e. light scatters equally in all direction
        if nearest_object.material.is_diffuse:
            # indirect light contribution
            indirect_ray_direction, pdf = cosine_weighted_hemisphere_sampling(surface_normal, ray.direction)

            if pdf==0:
                break

            indirect_ray_origin = intersection + EPSILON * indirect_ray_direction

            # change ray direction
            indirect_ray = Ray(indirect_ray_origin, indirect_ray_direction)

            cos_theta = np.dot(indirect_ray_direction, surface_normal)

            brdf = nearest_object.material.color.diffuse * inv_pi

            throughput *= brdf * cos_theta / pdf

            indirect_ray.fwd_pdf = indirect_ray.rev_pdf = pdf
            indirect_ray.color = brdf
            indirect_ray.medium = Medium.DIFFUSE.value

        # check if mirror surface, i.e. light reflects in a single outgoing direction
        elif nearest_object.material.is_mirror:
            # mirror reflection
            reflected_ray_origin = intersection + EPSILON * surface_normal
            reflected_ray_direction = get_reflected_direction(ray.direction, surface_normal)
            reflected_ray = Ray(reflected_ray_origin, reflected_ray_direction)

        # check if light gets transmitted in this medium
        elif nearest_object.material.transmission>0.0:
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

            if _sqrt > 0 and _rand>reflection_prob:
                # refraction
                transmit_origin = intersection + (-EPSILON * surface_normal)
                transmit_direction = (ray.direction * Nr) + (surface_normal * (Nr * cos_theta - np.sqrt(_sqrt)))
                transmit_direction = normalize(transmit_direction)
                transmit_ray = Ray(transmit_origin, transmit_direction)

            else:
                # reflection
                reflected_origin = intersection + EPSILON * surface_normal
                reflected_direction = get_reflected_direction(ray.direction, surface_normal)
                reflected_ray = Ray(reflected_origin, reflected_direction)

        else:
            # error
            break

        #TODO: break if throughput is black

        prev.pdf_rev = convert_density(pdf_rev, vx, prev)

        # # terminate path using russian roulette
        # if bounce>3:
        #     r_r = max(0.05, 1-ray.throughput[1]) # russian roulette factor
        #     if _rand<r_r:
        #         break
        #     ray.throughput /= 1-r_r

    return paths


@numba.njit
def setup_camera(scene, left_right, top_bottom):
    # computes the area of the visible screen and the camera normal
    screen_area = ((left_right[-1]-left_right[0])/scene.f_distance)*((top_bottom[-1]-top_bottom[0])/scene.f_distance)
    camera_plane_point_1 = camera_plane_point_2 = scene.camera
    camera_plane_point_1[1] = camera_plane_point_1[1]+1
    camera_plane_point_2[1] = camera_plane_point_2[1]-1
    camera_normal = normalize(np.cross(camera_plane_point_1-scene.camera, camera_plane_point_2-scene.camera))

    return screen_area, camera_normal


@numba.njit
def get_camera_importance(screen_area, cos_theta):
    # camera rays are always forward facing by design
    # camera rays are mapped to screen by default, hence 'importance' is always greater than 0
    # for point camera, the lens area is set to 1 and interpreted as a Dirac delta function
    if cos_theta<=0:
        cam_importance = 0
    else:
        cam_importance = 1/(screen_area*(cos_theta**4))
    return cam_importance


@numba.njit
def get_camera_pdf(screen_area, cos_theta):
    # returns spatial and directional pdfs of a camera
    pdf_pos = 1 # no lens, area is 1
    pdf_dir = 1/(screen_area*(cos_theta**3))
    return pdf_pos, pdf_dir


@numba.njit
def generate_camera_subpaths(scene, bvh, end, origin, screen_area, camera_normal):
    _rand = np.random.rand()
    end[0] += _rand/700
    end[1] += _rand/700
    direction = normalize(end - origin)
    ray = Ray(origin, direction)

    cos_theta = np.dot(ray.direction, camera_normal)

    pdf_pos, pdf_dir = get_camera_pdf(screen_area, cos_theta)

    importance = get_camera_importance(screen_area, cos_theta)

    throughput = 1 # 1 for simple camera model, otherwise a floating-point value that affects how much
    # the radiance arriving at the film plane along the generated ray will contribute to the final image.

    # camera is starting vertex for backward-path-tracing
    cam_vertex = Vertex(point=ray.origin,
                        g_norm=camera_normal,
                        pdf_pos=pdf_pos,
                        pdf_dir=pdf_dir,
                        importance=importance,
                        medium=Medium.CAMERA.value)


    camera_vertices = numba.typed.List() # will contain the vertices on the path starting from camera
    camera_vertices.append(cam_vertex)

    camera_vertices = random_walk(scene, bvh, ray, scene.max_depth-1, camera_vertices, throughput, pdf_dir, 1)

    return camera_vertices


@numba.njit
def sample_camera(scene, camera_normal, screen_area, intersection_point):
    # check if camera visible from the intersection point
    if not visible:
        return 0
    c_i_distance = np.linalg.norm(scene.camera - intersection_point)
    c_i_direction = normalize(scene.camera - intersection_point)
    pdf = (c_i_distance**2)/abs(np.dot(camera_normal, c_i_direction))
    cos_theta = np.dot(-c_i_direction, camera_normal)

    importance = get_camera_importance(screen_area, cos_theta)

    return pdf, importance


@numba.njit
def get_light_pdf(light, intersection_point):
    # call this function only if the light is visible from the intersection point
    # convert light sample weight to solid angle
    l_i_distance = np.linalg.norm(light.source - intersection_point)
    l_i_direction = normalize(light.source - intersection_point)
    pdf = (l_i_distance**2)/(abs(np.dot(light.normal, -l_i_direction))*light.area)
    return pdf




@numba.njit
def _sample_light(light, intersection_point):
    # check if light visible from the intersection point
    if not visible:
        return 0
    pdf = get_light_pdf(light, intersection_point)
    return pdf







@numba.njit
def generate_light_subpaths(scene, bvh):
    light_paths = numba.typed.List()
    # sample initial light ray
    light_ray = sample_light(scene)
    light_paths.append(light_ray)
    light_paths = random_walk(scene, bvh, scene.max_depth-1, light_paths, 2)

    return light_paths


@numba.njit
def convert_density(pdf, current_v, next_v):
    path = next_v.point - current_v.point
    path_magnitude = np.linalg.norm(path)
    inv_dist_sqr = 1/(path_magnitude*path_magnitude)
    if next_v.g_norm: # next vertex is on a surface if the geometric normal is non-zero
        pdf *= abs(np.dot(next_v.g_norm, path*math.sqrt(inv_dist_sqr)))
    return pdf * inv_dist_sqr


@numba.njit
def get_pdf(previous_v, current_v, next_v):
    if current_v.medium==Medium.LIGHT.value:
        pdf = current_v.pdf
    # compute prev and next direction
    next_d = normalize(next_v.origin-current_v.origin)
    prev_d = normalize(previous_v.origin-current_v.origin)
    # compute directional density
    if current_v.medium==Medium.CAMERA.value:
        pdf = current_v.pdf
    else:
        # vertex is on a surface
        pdf = get_bsdf_pdf(prev_d, next_d)

    return convert_density(pdf, current_v, next_v , next_normal)


@numba.njit
def connect_paths(scene, bvh, camera_paths, light_paths, t, s):
    light = np.zeros((3), dtype=np.float64)
    if t > 1 and s != 0 and camera_paths[t - 1].type == Medium.LIGHT.value:
        return light
    if s==0:
        # consider camera subpath as the entire path
        if camera_paths[t-1].type == Medium.LIGHT.value:
            L = camera_paths[t-1].color * camera_paths[t-1].throughput
    elif t==1:
        # connect camera to a light subpath
        light_vertex = light_paths[s-1].origin
        # sample a point on the camera
        cam = scene.camera
        # check if the camera is visible from the current vertex
        new_path = normalize(cam - light_vertex)
        new_path_magnitude = np.linalg.norm(cam - light_vertex)

        _objects = traverse_bvh(bvh, light_vertex, new_path)
        _, min_distance = nearest_intersected_object(_objects, light_vertex, new_path, t1=new_path_magnitude)

        if min_distance is None or min_distance >= new_path_magnitude:
            # no object in between, camera is visible
            if light_paths[s-1].medium.DIFFUSE:
                light += light_paths[s-1].throughput
            else:
                light += light_paths[s-1].throughput
    elif s==1:
        # connect the camera subpath with a light sample
        camera_vertex = camera_paths[t-1].origin
        # sample a point on the light
        random_light_index = np.random.choice(len(scene.lights), 1)[0]
        sampled_light = scene.lights[random_light_index]

        # check if the light is visible from the current vertex
        new_path = normalize(sampled_light.source - camera_vertex)
        new_path_magnitude = np.linalg.norm(sampled_light.source - camera_vertex)

        _objects = traverse_bvh(bvh, camera_vertex, new_path)
        _, min_distance = nearest_intersected_object(_objects, camera_vertex, new_path, t1=new_path_magnitude)

        if min_distance is None:
            # something went wrong
            return light
        if min_distance >= new_path_magnitude:
            # light is visible
            light += sampled_light.material.color * sampled_light.material.emission

    else:
        # follow rest of the strategies
        light += light_paths[s-1] * camera_paths[s-1]

    return light






@numba.njit(parallel=True)
def render_scene(scene, bvh, number_of_samples=10):

    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)

    screen_area, camera_normal = setup_camera(scene, left_right, top_bottom)

    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            light_contrib = np.zeros((3), dtype=np.float64)
            for _sample in range(number_of_samples):
                x = left_right[j]
                end = np.array([x, y, scene.depth, 1], dtype=np.float64) # pixel
                origin = np.array([scene.camera[0], scene.camera[1], scene.camera[2], 1], dtype=np.float64) # camera
                #TODO: implement better ray differentials, e.g. PBRT
                camera_vertices = generate_camera_subpaths(scene, bvh, end, origin, screen_area, camera_normal)
                light_vertices = generate_light_subpaths(scene, bvh)

                camera_n = len(camera_vertices)
                light_n = len(light_vertices)

                for t in range(1, camera_n+1):
                    for s in range(light_n+1):
                        depth = t+s-2
                        if (s == 1 and t == 1) or depth<0 or depth > scene.max_depth:
                            continue

                        light_contrib += connect_paths(scene, bvh, camera_paths, light_paths, t, s)

            light_contrib = light_contrib/number_of_samples
            scene.image[i, j] = np.clip(light_contrib, 0, 1)



        print(i)
    return scene.image