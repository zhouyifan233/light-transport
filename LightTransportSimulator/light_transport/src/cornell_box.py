import numpy as np
import numba

from .constants import EPSILON
from .material import Material, Color
from .primitives import Triangle, PreComputedTriangle, Sphere, Primitive, ShapeOptions
import pyvista as pv

xx = Primitive(ShapeOptions.SHAPE.value)

def get_cornell_box(scene, dim, surface_mat, left_wall_mat, right_wall_mat):
    box_triangles = [] #numba.typed.List()

    a = np.array([-dim, -dim, -dim], dtype=np.float64)
    b = np.array([-dim, -dim, dim], dtype=np.float64)
    c = np.array([dim, -dim, dim], dtype=np.float64)
    d = np.array([dim, -dim, -dim], dtype=np.float64)
    e = np.array([-dim, dim, -dim], dtype=np.float64)
    f = np.array([-dim, dim, dim], dtype=np.float64)
    g = np.array([dim, dim, dim], dtype=np.float64)
    h = np.array([dim, dim, -dim], dtype=np.float64)

    # right wall
    rectangle = pv.Rectangle([d, c, g, h])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=right_wall_mat)
        box_triangles.append(triangle)

    # left wall
    rectangle = pv.Rectangle([f, b, a, e])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=left_wall_mat)
        box_triangles.append(triangle)

    # back wall
    rectangle = pv.Rectangle([e, a, d, h])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # bottom wall
    rectangle = pv.Rectangle([a, b, c, d])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    i = np.array([-1, dim, -dim], dtype=np.float64)
    j = np.array([-1, dim, -1], dtype=np.float64)
    k = np.array([-1, dim, 1], dtype=np.float64)
    l = np.array([-1, dim, dim], dtype=np.float64)
    m = np.array([1, dim, dim], dtype=np.float64)
    n = np.array([1, dim, 1], dtype=np.float64)
    o = np.array([1, dim, -1], dtype=np.float64)
    p = np.array([1, dim, -dim], dtype=np.float64)

    # top wall - 1
    rectangle = pv.Rectangle([h, g, m, p]) # h, g, f, e -> entire top wall
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # top wall - 2
    rectangle = pv.Rectangle([n, m, l, k])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # top wall - 3
    rectangle = pv.Rectangle([p, o, j, i])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # top wall - 4
    rectangle = pv.Rectangle([i, l, f, e])
    rectangle.transform(scene.t_matrix)
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)


    return box_triangles


def get_floor(x_dim, y_dim, z_dim, surface_mat):
    box_triangles = numba.typed.List()

    a = np.array([-x_dim, -y_dim, -z_dim], dtype=np.float64)
    b = np.array([-x_dim, -y_dim, z_dim], dtype=np.float64)
    c = np.array([x_dim, -y_dim, z_dim], dtype=np.float64)
    d = np.array([x_dim, -y_dim, -z_dim], dtype=np.float64)

    rectangle = pv.Rectangle([a, b, c ,d])
    tri_rect = rectangle.triangulate()

    # if sub_divide is not None and sub_divide>0:
    #     tri_rect = tri_rect.subdivide_adaptive(max_n_passes=sub_divide)

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    return box_triangles


def get_cornell_sphere_box(scene, surface_mat, left_wall_mat, right_wall_mat, front_wall_mat):
    box_spheres = numba.typed.List()

    left_sphere_position = np.array([1e5+1, 40.8, 81.6], dtype=np.float64)
    right_sphere_position = np.array([-1e5+99, 40.8, 81.6], dtype=np.float64)
    bottom_sphere_position = np.array([50, 1e5, 81.6], dtype=np.float64)
    top_sphere_position = np.array([50, -1e5+81.6, 81.6], dtype=np.float64)
    back_sphere_position = np.array([50, 40.8, 1e5], dtype=np.float64)
    front_sphere_position = np.array([50, 40.8, -1e5+170], dtype=np.float64)

    left = Sphere(center=left_sphere_position, radius=1e5, material=left_wall_mat)
    box_spheres.append(left)
    right = Sphere(center=right_sphere_position, radius=1e5, material=right_wall_mat)
    box_spheres.append(right)
    back = Sphere(center=back_sphere_position, radius=1e5, material=surface_mat)
    box_spheres.append(back)
    bottom = Sphere(center=bottom_sphere_position, radius=1e5, material=surface_mat)
    box_spheres.append(bottom)
    top = Sphere(center=top_sphere_position, radius=1e5, material=surface_mat)
    box_spheres.append(top)
    front = Sphere(center=front_sphere_position, radius=1e5, material=front_wall_mat)
    box_spheres.append(front)

    return box_spheres

