import numpy as np

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


# def triangle_intersect(ray_origin, ray_direction, vertex_a, vertex_b, vertex_c):
#     """
#     returns the distance from the origin of the ray to the nearest intersection point
#     :param ray_origin: origin of the ray
#     :param ray_direction: direction from the camera to the triangle
#     :param vertex_a: first vertex of the triangle
#     :param vertex_b: second vertex of the triangle
#     :param vertex_c: third vertex of the triangle
#     :return: distance from origin to the intersection point, or None
#     """
#     ab = vertex_b - vertex_a
#     ac = vertex_c - vertex_a
#
#     normal = np.cross(ab, ac)
#
#     det = -np.dot(ray_direction, normal)
#
#     ao = ray_origin - vertex_a
#
#     dao = np.cross(ao, ray_direction)
#
#     u = np.dot(ac, dao) / det
#     v = -(np.dot(ab, dao) / det)
#     t = np.dot(ao, normal) / det
#
#     if det >= 1e-6 and t >= 0 and u >= 0 and v >= 0 and (u+v) <= 1:
#         print(t)
#         return t
#     else:
#         return None


def triangle_intersect(ray_origin, ray_end, vertex_a, vertex_b, vertex_c):
    """
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
