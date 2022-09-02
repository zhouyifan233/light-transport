import numba
import numpy as np
import numba

from .primitives import Triangle
from .vectors import normalize
from typing import Optional


@numba.njit
def sphere_intersect(ray_origin, ray_direction, sphere):
    """
    returns the distance from the origin of the ray to the nearest intersection point
    if the ray actually intersects the sphere, otherwise None.
    :param center: center of the sphere
    :param radius: radius of the sphere
    :param ray_origin: center of the camera
    :param ray_end: pixel on the sphere
    :return: distance from origin to the intersection point, or None
    """
    center = sphere.center
    radius = sphere.radius

    # ray_direction = normalize(ray_end - ray_origin)
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c # discriminant

    if delta > 0:
        # ray goes through the sphere
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            # return the first point of intersection
            return min(t1, t2)

    return None


@numba.njit
def triangle_intersect(ray_origin, ray_direction, triangle):
    """
     Möller–Trumbore ray-triangle intersection algorithm
    returns the distance from the origin of the ray to the nearest intersection point
    :param triangle: triangle primitive
    :param ray_origin: origin of the ray
    :param ray_end: pixel on the triangle
    :return: distance from origin to the intersection point, or None
    """

    eps = 0.0000001

    vertex_a = triangle.vertex_1
    vertex_b = triangle.vertex_2
    vertex_c = triangle.vertex_3

    plane_normal = triangle.normal

    ab = vertex_b - vertex_a
    ac = vertex_c - vertex_a

    # ray_direction = normalize(ray_end - ray_origin)

    ray_dot_plane = np.dot(ray_direction, plane_normal)

    if abs(ray_dot_plane)<=eps:
        return None

    pvec = np.cross(ray_direction[:-1], ac[:-1])
    pvec = np.append(pvec, 0)

    det = np.dot(ab, pvec)

    if -eps < det < eps:
        return None

    inv_det = 1.0 / det

    tvec = ray_origin - vertex_a

    u = np.dot(tvec, pvec) * inv_det

    if u < 0 or u > 1:
        return None

    qvec = np.cross(tvec[:-1], ab[:-1])
    qvec = np.append(qvec, 0)

    v = np.dot(ray_direction, qvec) * inv_det

    if v < 0 or u+v > 1:
        return None

    t = np.dot(ac, qvec) * inv_det

    if t > eps:
        return t
    else:
        return None


@numba.njit
def __triangle_intersect(ray_origin, ray_end, triangle):
    """
    Based on ray–tetrahedron intersection
    returns the distance from the origin of the ray to the nearest intersection point
    :param ray_origin: origin of the ray
    :param ray_end: pixel on the triangle
    :param vertex_a: first vertex of the triangle
    :param vertex_b: second vertex of the triangle
    :param vertex_c: third vertex of the triangle
    :return: distance from origin to the intersection point, or None
    """
    def signed_tetra_volume(a,b,c,d):
        return np.sign(np.dot(np.cross(b-a,c-a),d-a)/6.0)

    vertex_a = triangle.vertex_1
    vertex_b = triangle.vertex_2
    vertex_c = triangle.vertex_3

    s1 = signed_tetra_volume(ray_origin,vertex_a,vertex_b,vertex_c)
    s2 = signed_tetra_volume(ray_end,vertex_a,vertex_b,vertex_c)

    if s1 != s2:
        s3 = signed_tetra_volume(ray_origin,ray_end,vertex_a,vertex_b)
        s4 = signed_tetra_volume(ray_origin,ray_end,vertex_b,vertex_c)
        s5 = signed_tetra_volume(ray_origin,ray_end,vertex_c,vertex_a)
        if s3 == s4 and s4 == s5:
            n = np.cross(vertex_b-vertex_a,vertex_c-vertex_a)
            t = -np.dot(ray_origin,n-vertex_a) / np.dot(ray_origin,ray_end-ray_origin)
            return t

    return None


@numba.njit
def plane_intersect(ray_origin, ray_direction, plane):
    """
    returns the distance from the origin of the ray to the nearest intersection point
    :param ray_origin: origin of the ray
    :param ray_end: pixel on the plane
    :param plane_point: a point on the plane
    :param plane_normal: a vector normal to the plane
    :return: distance from origin to the intersection point, or None
    """
    plane_point = plane.point
    plane_normal = plane.normal

    # ray_direction = normalize(ray_end - ray_origin)
    ray_dot_plane = np.dot(ray_direction, plane_normal)

    if abs(ray_dot_plane)>1e-6:
        t = np.dot((plane_point - ray_origin), plane_normal)/ray_dot_plane
        if t>0:
            return t

    return None


@numba.njit
def aabb_intersect(ray_origin, ray_direction, box):
    t_min = 0.0
    t_max = np.inf
    ray_inv_dir = 1/ray_direction
    for i in range(3):
        t1 = (box.min_point[i] - ray_origin[i]) * ray_inv_dir[i]
        t2 = (box.max_point[i] - ray_origin[i]) * ray_inv_dir[i]
        t_min = min(max(t1, t_min), max(t2, t_min))
        t_max = max(min(t1, t_max), min(t2, t_max))
    return t_min<=t_max


@numba.njit
def pc_triangle_intersect(ray_origin, ray_direction, triangle):
    trans_s = triangle.transformation[8] * ray_origin[0]+\
              triangle.transformation[9] * ray_origin[1]+\
              triangle.transformation[10] * ray_origin[2]+\
              triangle.transformation[11]
    trans_d = triangle.transformation[8] * ray_direction[0]\
              +triangle.transformation[9] * ray_direction[1]\
              +triangle.transformation[10] * ray_direction[2]

    # t = (-(trans_s) / trans_d).item()
    # print(-(trans_s) / trans_d)
    t = (-(trans_s) / trans_d).item()

    # print("-----")
    # print(t)
    # print("-----")

    if t <= 0.0000001:
        return None

    gc = ray_origin + t * ray_direction

    bary_x = triangle.transformation[0] * gc[0]\
             + triangle.transformation[1] * gc[1]\
             + triangle.transformation[2] * gc[2]\
             + triangle.transformation[3]
    bary_y = triangle.transformation[4] * gc[0]\
             + triangle.transformation[5] * gc[1]\
             + triangle.transformation[6] * gc[2]\
             + triangle.transformation[7]

    if bary_x >= 0 and bary_y >= 0 and bary_x+bary_y < 1:
        return t

    return None