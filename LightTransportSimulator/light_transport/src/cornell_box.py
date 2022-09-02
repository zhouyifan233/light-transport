import numpy as np
import numba
from .material import Material, Color
from .primitives import Triangle, PreComputedTriangle
import pyvista as pv



def get_cornell_box(dim, surface_mat, left_wall_mat, right_wall_mat, start_id):
    box_triangles = numba.typed.List()

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
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=right_wall_mat)
        box_triangles.append(triangle)

    # left wall
    rectangle = pv.Rectangle([f, b, a, e])
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=left_wall_mat)
        box_triangles.append(triangle)

    # back wall
    rectangle = pv.Rectangle([e, a, d, h])
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # bottom wall
    rectangle = pv.Rectangle([a, b, c, d])
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
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
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # top wall - 2
    rectangle = pv.Rectangle([n, m, l, k])
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # top wall - 3
    rectangle = pv.Rectangle([p, o, j, i])
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    # top wall - 4
    rectangle = pv.Rectangle([i, l, f, e])
    tri_rect = rectangle.triangulate()

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        x, y, z = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(x, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(y, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(z, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)


    return box_triangles





# def get_floor(x_dim, y_dim, z_dim, surface_mat, start_id):
#     box_triangles = numba.typed.List()
#
#     start_id+=1
#     bottom_wall_1 = PreComputedTriangle(id=start_id,
#                              vertex_1=np.array([-x_dim, -y_dim, -z_dim, 1], dtype=np.float64),
#                              vertex_2=np.array([-x_dim, -y_dim, z_dim, 1], dtype=np.float64),
#                              vertex_3=np.array([x_dim, -y_dim, z_dim, 1], dtype=np.float64),
#                              material=surface_mat)
#     box_triangles.append(bottom_wall_1)
#
#     start_id+=1
#     bottom_wall_2 = PreComputedTriangle(id=start_id,
#                              vertex_1=np.array([-x_dim, -y_dim, -z_dim, 1], dtype=np.float64),
#                              vertex_2=np.array([x_dim, -y_dim, z_dim, 1], dtype=np.float64),
#                              vertex_3=np.array([x_dim, -y_dim, -z_dim, 1], dtype=np.float64),
#                              material=surface_mat)
#     box_triangles.append(bottom_wall_2)
#
#     return box_triangles


def get_floor(x_dim, y_dim, z_dim, surface_mat, start_id):
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
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=surface_mat)
        box_triangles.append(triangle)

    return box_triangles