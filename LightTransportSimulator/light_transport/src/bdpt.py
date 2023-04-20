import math

import numba
import numpy as np
import logging

from LightTransportSimulator.light_transport.src.bdpt_utils import get_pdf, get_light_origin_pdf, \
    get_light_pdf, get_camera_pdf, get_bsdf_pdf, sample_to_add_light
from LightTransportSimulator.light_transport.src.brdf import get_reflected_direction, sample_diffuse, sample_mirror, \
    sample_specular, bxdf
from LightTransportSimulator.light_transport.src.bvh_new import intersect_bvh
from LightTransportSimulator.light_transport.src.constants import EPSILON, inv_pi, Medium, ZEROS, TransportMode, ONES
from LightTransportSimulator.light_transport.src.control_variates import calculate_dlogpdu
from LightTransportSimulator.light_transport.src.light_samples import sample_light, cast_one_shadow_ray
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.utils import hit_object, cosine_weighted_hemisphere_sampling
from LightTransportSimulator.light_transport.src.vectors import normalize
from LightTransportSimulator.light_transport.src.vertex import Vertex, create_camera_vertex, create_surface_vertex, \
    create_light_vertex, convert_density, is_on_surface, is_connectible, get_vertex_color


@numba.njit
def random_walk(scene, primitives, bvh, ray, vertices, throughput, pdf, max_depth, rand_idx, mode):

    if max_depth==0:
        return vertices

    bounce = 0

    pdf_fwd = pdf
    pdf_rev = 0

    while True:
        rand_0 = scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), bounce]
        rand_1 = scene.rand_1[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), bounce]

        # intersect ray with scene
        nearest_object, min_distance, intersection, surface_normal = hit_object(primitives, bvh, ray)

        # terminate path if no intersection is found
        if nearest_object is None:
            # no object was hit
            if mode==TransportMode.RADIANCE.value:
                vertices = sample_to_add_light(scene, primitives, bvh, vertices)
                bounce+= 1
            break

        # create new vertex with intersection info
        mat_color = nearest_object.material.color.diffuse
        current_vertex = create_surface_vertex(intersection,
                                               ray.direction,
                                               surface_normal,
                                               min_distance,
                                               throughput,
                                               mat_color,
                                               pdf_fwd,
                                               vertices[-1])

        bounce+= 1

        if bounce>=max_depth:
            break

        if nearest_object.material.is_diffuse:
            # diffuse surface
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object, surface_normal, ray, [rand_0, rand_1])
            current_vertex.intr_type = intr_type
        elif nearest_object.material.is_mirror:
            # perfect mirror reflection
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object, surface_normal, ray, [rand_0, rand_1])
            current_vertex.is_delta = True
            current_vertex.intr_type = intr_type
        elif nearest_object.material.transmittance>0.0:
            # specular reflection (only dielectric materials)
            current_vertex.is_delta = True
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object, surface_normal, ray, [rand_0, rand_1], mode)
            current_vertex.intr_type = intr_type
        else:
            # error in material metadata
            break

        if pdf_fwd==0:
            break

        cos_theta = np.abs(np.dot(new_ray_direction, surface_normal))

        throughput = throughput * brdf * cos_theta / pdf_fwd

        if current_vertex.is_delta:
            pdf_rev = pdf_fwd = 0
        else:
            pdf_rev = get_bsdf_pdf(new_ray_direction, ray.direction) # n.b. reverse order

        # change ray origin
        if current_vertex.intr_type==Medium.REFRACTION.value:
            ray.origin = intersection + (-EPSILON * new_ray_direction)
        else:
            ray.origin = intersection + EPSILON * new_ray_direction

        # change ray direction
        ray.direction = new_ray_direction

        # Compute reverse area density at preceding vertex
        vertices[-1].pdf_rev = convert_density(pdf_rev, vertices[-1], current_vertex)

        # add current vertex to the list of vertices
        vertices.append(current_vertex)

    return vertices


@numba.njit
def generate_light_subpaths(scene, bvh, primitives, max_depth, rand_idx):
    light_vertices = numba.typed.List() # will contain the vertices on the path starting from light

    rand_0 = scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), 0]
    rand_1 = scene.rand_1[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), 0]

    if max_depth==0:
        return light_vertices

    # sample initial light ray
    light_ray, light_vertex, throughput = sample_light(scene, [rand_0, rand_1])

    if light_vertex.pdf_dir==0 or light_vertex.pdf_pos==0:
        return light_vertices

    # add the first vertex: light source
    light_vertices.append(light_vertex)

    # start random walk
    light_vertices = random_walk(scene,
                                 primitives,
                                 bvh,
                                 light_ray,
                                 light_vertices,
                                 throughput,
                                 light_vertex.pdf_dir,
                                 max_depth-1,
                                 rand_idx,
                                 TransportMode.IMPORTANCE.value
                                 )

    return light_vertices


@numba.njit
def setup_camera(scene, left_right, top_bottom):
    # computes the area of the visible screen and the camera normal
    if scene.camera.screen_area is None:
        scene.camera.screen_area = np.abs((left_right[-1]-left_right[0])*(top_bottom[-1]-top_bottom[0]))

    if scene.camera.normal is None:
        scene.camera.normal = normalize(np.array([0,0,scene.f_distance, 1])-scene.camera.position)

    return scene


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
def generate_camera_subpaths(scene, bvh, primitives, origin, end, max_depth, rand_idx):
    camera_vertices = numba.typed.List() # will contain the vertices on the path starting from camera

    if max_depth==0:
        return camera_vertices

    _rand = scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), 0] #TODO: use different _rand value
    end[0] += _rand/700
    end[1] += _rand/700
    direction = normalize(end - origin)
    ray_magnitude = np.linalg.norm(end - origin)
    ray = Ray(origin, direction)

    cos_theta = np.dot(ray.direction, scene.camera.normal)

    pdf_pos, pdf_dir = get_camera_pdf(scene.camera.screen_area, cos_theta)

    # importance = get_camera_importance(screen_area, cos_theta)

    throughput = 1.0 # 1 for simple camera model, otherwise a floating-point value that affects how much
    # the radiance arriving at the film plane along the generated ray will contribute to the final image.

    # camera is starting vertex for backward-path-tracing
    cam_vertex = create_camera_vertex(ray.origin, ray.direction, scene.camera.normal, ray_magnitude, pdf_pos, pdf_dir, throughput)

    camera_vertices.append(cam_vertex)

    camera_vertices = random_walk(scene,
                                  primitives,
                                  bvh,
                                  ray,
                                  camera_vertices,
                                  throughput,
                                  cam_vertex.pdf_dir,
                                  max_depth-1,
                                  rand_idx,
                                  TransportMode.RADIANCE.value
                                  )

    return camera_vertices



# @numba.njit
# def get_mis_weight(scene, light_vertices, camera_vertices, sampled, s, t):
#     if s+t == 2:
#         return 1
#
#     sum_ri = 0
#
#     re_map = lambda f: f if f != 0 else 1 # to avoid divide by 0
#
#     # Temporarily update vertex properties for current strategy
#     # Look up connection vertices and their predecessors
#     qs = light_vertices[s - 1] if s > 0 else None
#     pt = camera_vertices[t - 1] if t > 0 else None
#     qs_minus = light_vertices[s - 2] if s > 1 else None
#     pt_minus = camera_vertices[t - 2] if t > 1 else None
#
#     # Update sampled vertex for s=1 or t=1 strategy
#     if s == 1:
#         qs = sampled
#     elif t == 1:
#         pt = sampled
#
#     # Update reverse density of vertex pt_{t-1}
#     if pt is not None and pt.medium!=Medium.NONE.value:
#         if s>0:
#             pt.pdf_rev = get_pdf(scene, qs_minus, qs, pt)
#         else:
#             pt.pdf_rev = get_light_origin_pdf(scene, pt, pt_minus)
#
#     # Update reverse density of vertex pt_{t-2}
#     if pt_minus is not None and pt_minus.medium!=Medium.NONE.value:
#         if s>0:
#             pt_minus.pdf_rev = get_pdf(scene, qs, pt, pt_minus)
#         else:
#             pt_minus.pdf_rev = get_light_pdf(pt, pt_minus)
#
#     # Update reverse density of vertices qs_{s-1} and qs_{s-2}
#     if qs is not None and qs.medium!=Medium.NONE.value:
#         qs.pdf_rev = get_pdf(scene, pt_minus, pt, qs)
#
#     if qs_minus is not None and qs_minus.medium!=Medium.NONE.value:
#         qs_minus.pdf_rev = get_pdf(scene, pt, qs, qs_minus)
#
#     # Consider hypothetical connection strategies along the camera subpath
#     ri = 1
#     for i in range(t - 1, 0, -1):
#         ri *= re_map(camera_vertices[i].pdf_rev) / re_map(camera_vertices[i].pdf_fwd)
#         if not camera_vertices[i].is_delta:
#             sum_ri += ri
#
#     # Consider hypothetical connection strategies along the light subpath
#     ri = 1
#     for i in range(s - 1, -1, -1):
#         ri *= re_map(light_vertices[i].pdf_rev) / re_map(light_vertices[i].pdf_fwd)
#         if not light_vertices[i].is_delta:
#             sum_ri += ri
#
#     return 1/(1+sum_ri)


@numba.njit
def get_mis_weight(scene, light_vertices, camera_vertices, sampled, s, t):
    if s+t == 2:
        return 1

    sum_ri = 0

    re_map = lambda f: f if f != 0 else 1 # to avoid divide by 0

    # Temporarily update vertex properties for current strategy
    # Look up connection vertices and their predecessors



    qs = light_vertices[s - 1] if s > 0 else None
    pt = camera_vertices[t - 1] if t > 0 else None
    qs_minus = light_vertices[s - 2] if s > 1 else None
    pt_minus = camera_vertices[t - 2] if t > 1 else None

    # Update sampled vertex for s=1 or t=1 strategy
    if s == 1:
        qs = sampled
    elif t == 1:
        pt = sampled

    if pt is not None and pt.medium!=Medium.NONE.value:
        pt.is_delta = False
    if qs is not None and qs.medium!=Medium.NONE.value:
        qs.is_delta = False

    # Update reverse density of vertex pt_{t-1}
    if pt is not None and pt.medium!=Medium.NONE.value:
        if s>0:
            pt.pdf_rev = get_pdf(scene, qs_minus, qs, pt)
        else:
            pt.pdf_rev = get_light_origin_pdf(scene, pt, pt_minus)

    # Update reverse density of vertex pt_{t-2}
    if pt_minus is not None and pt_minus.medium!=Medium.NONE.value:
        if s>0:
            pt_minus.pdf_rev = get_pdf(scene, qs, pt, pt_minus)
        else:
            pt_minus.pdf_rev = get_light_pdf(pt, pt_minus)

    # Update reverse density of vertices qs_{s-1} and qs_{s-2}
    if qs is not None and qs.medium!=Medium.NONE.value:
        qs.pdf_rev = get_pdf(scene, pt_minus, pt, qs)

    if qs_minus is not None and qs_minus.medium!=Medium.NONE.value:
        qs_minus.pdf_rev = get_pdf(scene, pt, qs, qs_minus)

    # Consider hypothetical connection strategies along the camera subpath
    ri = 1
    for i in range(t - 1, 0, -1):
        ri *= re_map(camera_vertices[i].pdf_rev) / re_map(camera_vertices[i].pdf_fwd)
        if not camera_vertices[i].is_delta:
            sum_ri += ri

    # Consider hypothetical connection strategies along the light subpath
    ri = 1
    for i in range(s - 1, -1, -1):
        ri *= re_map(light_vertices[i].pdf_rev) / re_map(light_vertices[i].pdf_fwd)
        if not light_vertices[i].is_delta:
            sum_ri += ri

    weight = 1/(1+sum_ri)

    # print('MIS_weight= ', weight)

    return weight



@numba.njit
def get_light(scene, curr_v, next_v):

    w = normalize(next_v.point - curr_v.point)

    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]

    if np.dot(light.normal, w)>0:
        throughput = light.material.emission*light.material.color.diffuse
    else:
        throughput = ZEROS

    return throughput


# @numba.njit
# def get_light_pdf(light, intersection_point):
#     # call this function only if the light is visible from the intersection point
#     # convert light sample weight to solid angle
#     l_i_distance = np.linalg.norm(light.source - intersection_point)
#     l_i_direction = normalize(light.source - intersection_point)
#     pdf = (l_i_distance**2)/(abs(np.dot(light.normal, -l_i_direction))*light.area)
#     return pdf



@numba.njit
def G(vertex_0, vertex_1, primitives, bvh):
    d = normalize(vertex_0.point-vertex_1.point)
    r = Ray(vertex_1.point, d)
    m = np.linalg.norm(vertex_0.point-vertex_1.point)
    g = 1/m**2
    if is_on_surface(vertex_0):
        g *= np.abs(np.dot(vertex_0.g_norm, d))
    if is_on_surface(vertex_1):
        g *= np.abs(np.dot(vertex_1.g_norm, d))

    _, min_distance = intersect_bvh(r, primitives, bvh)
    if min_distance is None or min_distance >= m:
        # vertex_0 and vertex_1 connectible
        vis = 1
    else:
        vis = 0
    return g*vis



@numba.njit
def connect_paths(scene, bvh, primitives, camera_vertices, light_vertices, s, t):
    light = ZEROS
    sampled = None

    # check for invalid connections
    if t > 1 and s != 0 and camera_vertices[t - 1].medium == Medium.LIGHT.value:
        return light


    if s==0:
        # consider camera subpath as the entire path
        if camera_vertices[t-1].medium == Medium.LIGHT.value:
            light = get_light(scene, camera_vertices[t-1], camera_vertices[t-2]) * (camera_vertices[t-1].throughput * get_vertex_color(camera_vertices[t-1]))


    elif t==1:
        # connect camera to a light subpath
        qs = light_vertices[s-1]

        if is_connectible(qs):
            # sample a point on the camera
            cam = camera_vertices[0]
            # check if the camera is visible from the current vertex
            new_path_direction = normalize(cam.point - qs.point)
            new_path = Ray(qs.point, new_path_direction)
            new_path_magnitude = np.linalg.norm(cam.point - qs.point)

            _, min_distance = intersect_bvh(new_path, primitives, bvh)

            if min_distance is None or min_distance >= new_path_magnitude:
                # no object in between, camera is visible
                cos_theta = np.dot(new_path_direction, scene.camera.normal)
                cos2_theta = cos_theta**2
                pdf_dir = (new_path_magnitude**2) / (np.abs(np.dot(scene.camera.normal, new_path_direction)) * 1) # lens_area = 1
                pdf_pos = 1

                throughput = 1 / (scene.camera.screen_area * 1 * cos2_theta * cos2_theta)

                if pdf_dir>0:
                    sampled = create_camera_vertex(cam.point, new_path_direction, scene.camera.normal, new_path_magnitude, pdf_pos, pdf_dir, throughput/pdf_dir)

                    light = (qs.throughput * get_vertex_color(qs)) * bxdf(qs, sampled) * sampled.throughput

                    if is_on_surface(qs):
                        light *= np.abs(np.dot(new_path_direction, qs.g_norm))


    elif s==1:
        # connect the camera subpath with a light sample
        pt = camera_vertices[t-1]

        if is_connectible(pt):
            # sample a point on the light
            random_light_index = np.random.choice(len(scene.lights), 1)[0]
            sampled_light = scene.lights[random_light_index]

            # check if the light is visible from the current vertex
            new_path_direction = -normalize(sampled_light.source - pt.point)
            new_path = Ray(pt.point, new_path_direction) # n.b. -ve direction
            new_path_magnitude = np.linalg.norm(sampled_light.source - pt.point)

            _, min_distance = intersect_bvh(new_path, primitives, bvh)

            if min_distance is None or min_distance >= new_path_magnitude:
                # light is visible, create a vertex
                sampled = create_light_vertex(sampled_light, new_path_direction, new_path_magnitude, 0, 0)
                pdf = get_light_pdf(sampled, pt)

                if pdf==0 or new_path_magnitude==0:
                    pdf = 0
                    light_throughput = 0
                else:
                    if np.dot(sampled_light.normal, new_path_direction)>0:
                        light_throughput = sampled_light.material.emission
                    else:
                        light_throughput = 0

                light_pdf = 1

                if pdf>0 and not np.array_equal(light_throughput, ZEROS):
                    # update light vertex
                    sampled.pdf_dir = pdf
                    sampled.throughput = light_throughput/(pdf*light_pdf)
                    sampled.color = sampled_light.material.color.diffuse
                    sampled.pdf_pos = 1/sampled_light.total_area
                    sampled.pdf_fwd = get_light_origin_pdf(scene, sampled, pt)
                    sampled.color = sampled_light.material.color.diffuse

                    light = (pt.throughput * get_vertex_color(pt)) * bxdf(pt, sampled) * (sampled.throughput * sampled.color)

                    if is_on_surface(pt):
                        light *= np.abs(np.dot(new_path_direction, pt.g_norm))


    else:
        # follow rest of the strategies
        qs = light_vertices[s-1]
        pt = camera_vertices[t-1]

        if is_connectible(qs) and is_connectible(pt):

            light = (qs.throughput*get_vertex_color(qs)) * bxdf(qs, pt) * bxdf(pt, qs) * (pt.throughput*get_vertex_color(pt))
            if not np.array_equal(light, ZEROS):
                light *= G(qs, pt, primitives, bvh)


    # compute MIS-weights for the above connection strategies
    if np.array_equal(light, ZEROS):
        mis_weight = 0.0
    else:
        mis_weight = get_mis_weight(scene, light_vertices, camera_vertices, sampled, s, t)

    # print('mis_weight: ', mis_weight, ', light: ', light)

    light = light * mis_weight

    return light


@numba.njit(parallel=True)
def render_scene(scene, primitives, bvh):

    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    pix_count = 0

    scene = setup_camera(scene, left_right, top_bottom)

    for i in numba.prange(scene.height):
        y = top_bottom[i]
        for j in numba.prange(scene.width):
            color = np.zeros((3), dtype=np.float64)
            for _sample in range(scene.number_of_samples):
                rand = [scene.rand_0[i, j, _sample], scene.rand_1[i, j, _sample]]
                rand_idx = [i, j, _sample]
                dlogpdu = calculate_dlogpdu(rand)
                x = left_right[j]
                # screen is on origin
                end = np.array([x, y, scene.f_distance, 1], dtype=np.float64) # pixel
                # anti-aliasing
                end[0] += rand[0][0]/scene.width
                end[1] += rand[0][0]/scene.height
                cam = scene.camera.position
                origin = np.array([cam[0], cam[1], cam[2], 1], dtype=np.float64)
                #TODO: implement better ray differentials, e.g. PBRT
                camera_vertices = generate_camera_subpaths(scene, bvh, primitives, origin, end, scene.max_depth+2, rand_idx)
                light_vertices = generate_light_subpaths(scene, bvh, primitives, scene.max_depth+1, rand_idx)

                camera_n = len(camera_vertices)
                light_n = len(light_vertices)

                for t in range(1, camera_n+1):
                    for s in range(light_n+1):
                        depth = t+s-2
                        if (s == 1 and t == 1) or depth<0 or depth > scene.max_depth:
                            continue

                        color += connect_paths(scene, bvh, primitives, camera_vertices, light_vertices, s, t)
                        # print('color: ', color)

            color = color/scene.number_of_samples
            # print('color: ', color)
            scene.image[i, j] = np.clip(color, 0, 1)
        pix_count += 1
        print('Progress:-', (pix_count/scene.height)*100)
    return scene.image