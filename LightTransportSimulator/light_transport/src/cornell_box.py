import numpy as np
import numba
from .material import Material, Color
from .primitives import Triangle


def get_cornell_box(dim, surface_mat, left_wall_mat, right_wall_mat):
    box_triangles = []

    back_wall_1 = Triangle(vertex_1=np.array([-dim, dim, -dim], dtype=np.float64),
                           vertex_2=np.array([-dim, -dim, -dim], dtype=np.float64),
                           vertex_3=np.array([dim, -dim, -dim], dtype=np.float64),
                           material=surface_mat)

    box_triangles.append(back_wall_1)

    back_wall_2 = Triangle(vertex_1=np.array([-dim, dim, -dim], dtype=np.float64),
                           vertex_2=np.array([dim, -dim, -dim], dtype=np.float64),
                           vertex_3=np.array([dim, dim, -dim], dtype=np.float64),
                           material=surface_mat)

    box_triangles.append(back_wall_2)

    top_wall_1 = Triangle(vertex_1=np.array([-dim, dim, -dim], dtype=np.float64),
                          vertex_2=np.array([dim, dim, dim], dtype=np.float64),
                          vertex_3=np.array([-dim, dim, dim], dtype=np.float64),
                          material=surface_mat)

    box_triangles.append(top_wall_1)

    top_wall_2 = Triangle(vertex_1=np.array([dim, dim, dim], dtype=np.float64),
                          vertex_2=np.array([-dim, dim, -dim], dtype=np.float64),
                          vertex_3=np.array([dim, dim, -dim], dtype=np.float64),
                          material=surface_mat)

    box_triangles.append(top_wall_2)

    bottom_wall_1 = Triangle(vertex_1=np.array([-dim, -dim, -dim], dtype=np.float64),
                             vertex_2=np.array([-dim, -dim, dim], dtype=np.float64),
                             vertex_3=np.array([dim, -dim, dim], dtype=np.float64),
                             material=surface_mat)

    box_triangles.append(bottom_wall_1)

    bottom_wall_2 = Triangle(vertex_1=np.array([-dim, -dim, -dim], dtype=np.float64),
                             vertex_2=np.array([dim, -dim, dim], dtype=np.float64),
                             vertex_3=np.array([dim, -dim, -dim], dtype=np.float64),
                             material=surface_mat)

    box_triangles.append(bottom_wall_2)

    # front_wall_1 = Triangle(vertex_1=np.array([-dim, -dim, dim], dtype=np.float64),
    #                          vertex_2=np.array([-dim, dim, dim], dtype=np.float64),
    #                          vertex_3=np.array([dim, -dim, dim], dtype=np.float64),
    #                          material=surface_mat)
    #
    # box_triangles.append(front_wall_1)
    #
    # front_wall_2 = Triangle(vertex_1=np.array([dim, -dim, dim], dtype=np.float64),
    #                          vertex_2=np.array([-dim, dim, dim], dtype=np.float64),
    #                          vertex_3=np.array([dim, dim, dim], dtype=np.float64),
    #                          material=surface_mat)
    #
    # box_triangles.append(front_wall_2)

    left_wall_1 = Triangle(vertex_1=np.array([-dim, dim, -dim], dtype=np.float64),
                           vertex_2=np.array([-dim, dim, dim], dtype=np.float64),
                           vertex_3=np.array([-dim, -dim, -dim], dtype=np.float64),
                           material=left_wall_mat)

    box_triangles.append(left_wall_1)

    left_wall_2 = Triangle(vertex_1=np.array([-dim, -dim, -dim], dtype=np.float64),
                           vertex_2=np.array([-dim, dim, dim], dtype=np.float64),
                           vertex_3=np.array([-dim, -dim, dim], dtype=np.float64),
                           material=left_wall_mat)

    box_triangles.append(left_wall_2)

    right_wall_1 = Triangle(vertex_1=np.array([dim, -dim, -dim], dtype=np.float64),
                            vertex_2=np.array([dim, -dim, dim], dtype=np.float64),
                            vertex_3=np.array([dim, dim, dim], dtype=np.float64),
                            material=right_wall_mat)

    box_triangles.append(right_wall_1)

    right_wall_2 = Triangle(vertex_1=np.array([dim, dim, -dim], dtype=np.float64),
                            vertex_2=np.array([dim, -dim, -dim], dtype=np.float64),
                            vertex_3=np.array([dim, dim, dim], dtype=np.float64),
                            material=right_wall_mat)

    box_triangles.append(right_wall_2)

    return box_triangles