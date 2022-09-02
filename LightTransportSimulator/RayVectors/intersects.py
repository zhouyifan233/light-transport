import numpy as np
from numba import jit

from LightTransportSimulator.RayVectors.vectors import normalize


def sphere_intersect(center, radius, ray_origin, ray_end):
    """
    returns the distance from the origin of the ray to the nearest intersection point
    if the ray actually intersects the sphere, otherwise None.
    :param center: center of the sphere
    :param radius: radius of the sphere
    :param ray_origin: center of the camera
    :param ray_end: pixel on the sphere
    :return: distance from origin to the intersection point, or None
    """
    ray_direction = normalize(ray_end - ray_origin)
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


def triangle_intersect(ray_origin, ray_end, vertex_a, vertex_b, vertex_c):
    """
    Möller-Trumbore algorithm
    returns the distance from the origin of the ray to the nearest intersection point
    :param ray_origin: origin of the ray
    :param ray_end: pixel on the triangle
    :param vertex_a: first vertex of the triangle
    :param vertex_b: second vertex of the triangle
    :param vertex_c: third vertex of the triangle
    :return: distance from origin to the intersection point, or None
    """

    eps = 0.0000001

    ab = vertex_b - vertex_a
    ac = vertex_c - vertex_a

    ray_direction = normalize(ray_end - ray_origin)

    plane_normal = normalize(np.cross(ab, ac))
    ray_dot_plane = np.dot(ray_direction, plane_normal)

    if abs(ray_dot_plane)<=eps:
        return None

    pvec = np.cross(ray_direction, ac)

    det = np.dot(ab, pvec)

    if -eps < det < eps:
        return None

    inv_det = 1.0 / det

    tvec = ray_origin - vertex_a

    u = np.dot(tvec, pvec) * inv_det

    if u < 0 or u > 1:
        return None

    qvec = np.cross(tvec, ab)

    v = np.dot(ray_direction, qvec) * inv_det

    if v < 0 or u+v > 1:
        return None

    t = np.dot(ac, qvec) * inv_det

    if t > eps:
        return t
    else:
        return None


def __triangle_intersect(ray_origin, ray_end, vertex_a, vertex_b, vertex_c):
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


def plane_intersect(ray_origin, ray_end, plane_point, plane_normal):
    """
    returns the distance from the origin of the ray to the nearest intersection point
    :param ray_origin: origin of the ray
    :param ray_end: pixel on the plane
    :param plane_point: a point on the plane
    :param plane_normal: a vector normal to the plane
    :return: distance from origin to the intersection point, or None
    """
    ray_direction = normalize(ray_end - ray_origin)
    ray_dot_plane = np.dot(ray_direction, plane_normal)
    if abs(ray_dot_plane)>1e-6:
        t = np.dot((plane_point - ray_origin), plane_normal)/ray_dot_plane
        if t>0:
            return t
    return None