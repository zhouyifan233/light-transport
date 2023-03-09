import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.bdpt_utils import convert_density, get_pdf, get_light_origin_pdf, \
    get_light_pdf, get_camera_pdf, get_bsdf_pdf
from LightTransportSimulator.light_transport.src.brdf import get_reflected_direction, sample_diffuse, sample_mirror, \
    sample_specular
from LightTransportSimulator.light_transport.src.bvh_new import intersect_bvh
from LightTransportSimulator.light_transport.src.constants import EPSILON, inv_pi, Medium, ZEROS
from LightTransportSimulator.light_transport.src.control_variates import calculate_dlogpdu
from LightTransportSimulator.light_transport.src.light_samples import sample_light, cast_one_shadow_ray
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.utils import hit_object, cosine_weighted_hemisphere_sampling
from LightTransportSimulator.light_transport.src.vectors import normalize
from LightTransportSimulator.light_transport.src.vertex import Vertex, create_camera_vertex, create_surface_vertex, \
    create_light_vertex


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
            break

        # create new vertex with intersection info
        current_vertex = create_surface_vertex(intersection, throughput, pdf_fwd, vertices[-1])

        bounce+= 1

        if bounce>=max_depth:
            break

        if nearest_object.material.is_diffuse:
            # diffuse surface
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object, surface_normal, ray, [rand_0, rand_1])
        elif nearest_object.material.is_mirror:
            # perfect mirror reflection
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object, surface_normal, ray, [rand_0, rand_1])
            current_vertex.is_delta = True
        elif nearest_object.material.transmittance>0.0:
            # specular reflection (only dielectric materials)
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object, surface_normal, ray, [rand_0, rand_1])
        else:
            # error in material metadata
            break

        if pdf_fwd==0:
            break

        throughput *= brdf * np.abs(np.dot(new_ray_direction, surface_normal)) / pdf_fwd

        pdf_rev = get_bsdf_pdf(new_ray_direction, ray.direction) # n.b. reverse order

        # change ray origin
        if intr_type == Medium.DIFFUSE.value or intr_type == Medium.MIRROR.value or intr_type == Medium.REFLECTION.value:
            ray.origin = intersection + EPSILON * new_ray_direction
        else:
            ray.origin = intersection + (-EPSILON * new_ray_direction)

        # change ray direction
        ray.direction = new_ray_direction

        # Compute reverse area density at preceding vertex
        vertices[-1].pdf_rev = convert_density(pdf_rev, vertices[-1].point, current_vertex.point)

        # add current vertex to the list of vertices
        vertices.append(current_vertex)

    return vertices


@numba.njit
def generate_light_subpaths(scene, bvh, primitives, rand_idx):
    #TODO: Implement all the conditions that would return None

    light_vertices = numba.typed.List() # will contain the vertices on the path starting from light

    # sample initial light ray
    light_ray, light_vertex, throughput = sample_light(scene)

    # add the first vertex: light source
    light_vertices.append(light_vertex)

    # start random walk
    light_vertices = random_walk(scene, primitives, bvh, light_ray, light_vertices, throughput, light_vertex.pdf_dir, scene.max_depth-1, rand_idx, 1)

    return light_vertices


@numba.njit
def setup_camera(scene, left_right, top_bottom):
    # computes the area of the visible screen and the camera normal
    if scene.camera.screen_area is not None:
        screen_area = scene.camera.screen_area
    else:
        screen_area = (left_right[-1]-left_right[0])*(top_bottom[-1]-top_bottom[0])

    if scene.camera.normal is not None:
        camera_normal = scene.camera.camera_normal
    else:
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
def generate_camera_subpaths(scene, bvh, primitives, end, origin, screen_area, camera_normal, max_depth, rand_idx):
    camera_vertices = numba.typed.List() # will contain the vertices on the path starting from camera

    if max_depth==0:
        return camera_vertices

    _rand = scene.rand_0[int(rand_idx[0]), int(rand_idx[1]), int(rand_idx[2]), 0] #TODO: use different _rand value
    end[0] += _rand/700
    end[1] += _rand/700
    direction = normalize(end - origin)
    ray = Ray(origin, direction)

    cos_theta = np.dot(ray.direction, camera_normal)

    pdf_pos, pdf_dir = get_camera_pdf(screen_area, cos_theta)

    # importance = get_camera_importance(screen_area, cos_theta)

    throughput = 1 # 1 for simple camera model, otherwise a floating-point value that affects how much
    # the radiance arriving at the film plane along the generated ray will contribute to the final image.

    # camera is starting vertex for backward-path-tracing
    cam_vertex = create_camera_vertex(ray.origin, camera_normal, pdf_pos, pdf_dir, throughput)

    camera_vertices.append(cam_vertex)

    camera_vertices = random_walk(scene, primitives, bvh, ray, camera_vertices, throughput, cam_vertex.pdf_dir, max_depth-1, rand_idx, 1)

    return camera_vertices



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

    return 1/(1+sum_ri)


def get_light(vx):
    if vx.medium != Medium.LIGHT.value:
        return np.zeros((3), dtype=np.float64)

    return vx.throughput


# @numba.njit
# def get_light_pdf(light, intersection_point):
#     # call this function only if the light is visible from the intersection point
#     # convert light sample weight to solid angle
#     l_i_distance = np.linalg.norm(light.source - intersection_point)
#     l_i_direction = normalize(light.source - intersection_point)
#     pdf = (l_i_distance**2)/(abs(np.dot(light.normal, -l_i_direction))*light.area)
#     return pdf


@numba.njit
def is_on_surface(vertex):
    if (vertex.medium == Medium.DIFFUSE.value) or \
            (vertex.medium == Medium.MIRROR.value) or \
            (vertex.medium == Medium.SPECULAR.value):
        return True
    return False


@numba.njit
def G(vertex_0, vertex_1, primitives, bvh):
    d = normalize(vertex_1.point-vertex_0.point)
    m = np.linalg.norm(vertex_1.point-vertex_0.point)
    g = 1/m**2
    if is_on_surface(vertex_0):
        g *= np.abs(np.dot(vertex_0.g_norm, d))
    if is_on_surface(vertex_1):
        g *= np.abs(np.dot(vertex_1.g_norm, d))

    _, min_distance = intersect_bvh(d, primitives, bvh)
    if min_distance is None or min_distance >= m:
        # vertex_0 and vertex_1 connectible
        vis = 1
    else:
        vis = 0
    return g*vis








@numba.njit
def connect_paths(scene, bvh, primitives, camera_vertices, light_vertices, t, s):
    light = np.zeros((3), dtype=np.float64)
    sampled = Vertex(np.array([np.inf, np.inf, np.inf]))

    # check for invalid connections
    if t > 1 and s != 0 and camera_vertices[t - 1].medium == Medium.LIGHT.value:
        return light


    if s==0:
        # consider camera subpath as the entire path
        if camera_vertices[t-1].medium == Medium.LIGHT.value:
            light += get_light(camera_vertices[t-2]) * camera_vertices[t-1].throughput


    elif t==1:
        # connect camera to a light subpath
        light_p = light_vertices[s-1].point
        # sample a point on the camera
        sampled = camera_vertices[0]
        # check if the camera is visible from the current vertex
        new_path = normalize(sampled.point - light_p)
        new_path_magnitude = np.linalg.norm(sampled.point - light_p)

        _, min_distance = intersect_bvh(new_path, primitives, bvh)

        if min_distance is None or min_distance >= new_path_magnitude:
            # no object in between, camera is visible
            light += light_vertices[s-1].throughput*light_vertices[s-1].bsdf*cam.throughput


    elif s==1:
        # connect the camera subpath with a light sample
        camera_vertex = camera_vertices[t-1].point
        # sample a point on the light
        random_light_index = np.random.choice(len(scene.lights), 1)[0]
        sampled_light = scene.lights[random_light_index]

        # check if the light is visible from the current vertex
        new_path = normalize(sampled_light.source - camera_vertex)
        new_path_magnitude = np.linalg.norm(sampled_light.source - camera_vertex)
        pdf = get_light_pdf(sampled_light, camera_vertex)
        light_choice_pdf = 1
        if pdf>0:
            # create a light vertex
            sampled = create_light_vertex(sampled_light, pdf, 0)
            sampled.throughput = sampled_light.material.emission/(pdf*light_choice_pdf)
            sampled.pdf_pos = 1/sampled_light.total_area
            sampled.pdf_fwd = light_choice_pdf*sampled.pdf_pos
            sampled.color = sampled_light.material.color

            total_light = sampled.color * sampled.throughput

            _, min_distance = intersect_bvh(new_path, primitives, bvh)

            if min_distance is None or min_distance >= new_path_magnitude:
                # light is visible
                light +=  total_light * camera_vertices[t-1].throughput

            if is_on_surface(camera_vertices[t - 1]):
                light *= np.abs(np.dot(new_path, camera_vertices[t - 1].g_norm))


    else:
        # follow rest of the strategies
        light += light_vertices[s-1].throughput * camera_vertices[t-1].throughput
        light *= G(light_vertices[s-1], camera_vertices[t-1], primitives, bvh)

    # compute MIS-weights for the above connection strategies
    if np.array_equal(light, ZEROS):
        mis_weight = 0.0
    else:
        mis_weight = get_mis_weight(scene, light_vertices, camera_vertices, sampled, s, t)

    light *= mis_weight


    return light


@numba.njit(parallel=True)
def render_scene(scene, primitives, bvh):

    top_bottom = np.linspace(scene.top, scene.bottom, scene.height)
    left_right = np.linspace(scene.left, scene.right, scene.width)
    pix_count = 0

    screen_area, camera_normal = setup_camera(scene, left_right, top_bottom)

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
                origin = np.array([scene.camera[0], scene.camera[1], scene.camera[2], 1], dtype=np.float64)
                #TODO: implement better ray differentials, e.g. PBRT
                camera_vertices = generate_camera_subpaths(scene, bvh, primitives, end, origin, screen_area, camera_normal, scene.max_depth+2, rand_idx)
                light_vertices = generate_light_subpaths(scene, bvh, primitives, rand_idx)

                camera_n = len(camera_vertices)
                light_n = len(light_vertices)

                for t in range(1, camera_n+1):
                    for s in range(light_n+1):
                        depth = t+s-2
                        if (s == 1 and t == 1) or depth<0 or depth > scene.max_depth:
                            continue

                        color += connect_paths(scene, bvh, primitives, camera_vertices, light_vertices, t, s)

            color = color/scene.number_of_samples
            scene.image[i, j] = np.clip(color, 0, 1)

        print(i)
    return scene.image