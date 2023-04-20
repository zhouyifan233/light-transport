import numba
import numpy as np
import numba

from .constants import EPSILON
from .primitives import Triangle, Sphere, PreComputedTriangle
from .vectors import normalize
from typing import Optional


# @numba.njit
# def sphere_intersect(ray_origin, ray_direction, sphere):
#     """
#     returns the distance from the origin of the ray to the nearest intersection point
#     if the ray actually intersects the sphere, otherwise None.
#     :param center: center of the sphere
#     :param radius: radius of the sphere
#     :param ray_origin: center of the camera
#     :param ray_end: pixel on the sphere
#     :return: distance from origin to the intersection point, or None
#     """
#     center = sphere.center
#     radius = sphere.radius
#
#     # ray_direction = normalize(ray_end - ray_origin)
#     b = 2 * np.dot(ray_direction, ray_origin - center)
#     c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
#     delta = b ** 2 - 4 * c # discriminant
#
#     if delta > 0:
#         # ray goes through the sphere
#         t1 = (-b + np.sqrt(delta)) / 2
#         t2 = (-b - np.sqrt(delta)) / 2
#         if t1 > 0 and t2 > 0:
#             # return the first point of intersection
#             # return min(t1, t2)
#             if t1>t2:
#                 return t2
#             else:
#                 return t1
#
#     return None



@numba.njit
def sphere_intersect(ray_origin, ray_direction, sphere):

    op = sphere.center - ray_origin
    eps = 1e-4
    b = np.dot(ray_direction, op)
    det = b*b - np.dot(op, op) + sphere.radius*sphere.radius
    if det<0:
        return None

    det = np.sqrt(det)
    t1 = b-det
    t2 = b+det

    if t1>eps:
        return t1
    else:
        if t2>eps:
            return t2
        else:
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
def intersect_bounds(bounds, ray, inv_dir, hit0=None, hit1=None):
    t0 = 0
    t1 = ray.tmax
    for i in range(3):
        t_near = (bounds.min_point[i]-ray.origin[i])*inv_dir[i]
        t_far = (bounds.max_point[i]-ray.origin[i])*inv_dir[i]
        if t_near>t_far:
            t_near, t_far = t_far, t_near
        t_far *= 1+2*gamma(3)
        t0 = t_near if t_near>t0 else t0
        t1 = t_far if t_far<t1 else t1
        if t0>t1:
            return False
    if hit0 is not None:
        hit0=t0
    if hit1 is not None:
        hit1=t1
    return True




@numba.njit
def create_orthonormal_system(normal):
    if abs(normal[0]) > abs(normal[1]):
        v2 = np.array([-normal[2], 0.0, normal[0]], dtype=np.float64) / np.sqrt(np.array([normal[0] * normal[0] + normal[2] * normal[2]], dtype=np.float64))
    else:
        v2 = np.array([0.0, normal[2], -normal[1]], dtype=np.float64) / np.sqrt(np.array([normal[1] * normal[1] + normal[2] * normal[2]], dtype=np.float64))

    v3 = np.cross(normal, v2)

    return v2, v3


@numba.njit
def max_dimension(v):
    return 0 if v[0] > v[1] and v[0] > v[2] else 1 if v[1] > v[2] else 2


@numba.njit
def permute(point, x, y, z):
    return np.array([point[x], point[y], point[z]])


@numba.njit
def max_component(v):
    return max(v[0], max(v[1], v[2]))


@numba.njit
def get_machine_epsilon():
    return np.finfo(np.float32).eps*0.5

@numba.njit
def gamma(n):
    eps = get_machine_epsilon()
    return (n * eps) / (1 - n * eps)


@numba.experimental.jitclass([
    ('intersected_point', numba.float64[:]),
    ('incoming_direction', numba.float64[:]),
    ('dpdu', numba.float64[:]),
    # ('dpdv', numba.float64[:]),
    ('triangle', PreComputedTriangle.class_type.instance_type),
    ('normal', numba.float64[:]),
    ('shading_normal', numba.float64[:])
])
class SurfaceInteraction:
    def __init__(self, intersected_point, incoming_direction, dpdu, normal, shading_normal):
        _intersected_point = intersected_point
        self.intersected_point = np.append(intersected_point,1)
        _incoming_direction = incoming_direction
        self.incoming_direction = np.append(incoming_direction,0)
        self.dpdu = dpdu
        # self.dpdv = dpdv
        # self.triangle = triangle
        _normal = normal
        self.normal = np.append(_normal,0)
        _shading_normal = shading_normal
        self.shading_normal = np.append(_shading_normal,0)


@numba.njit
def get_UVs():
    return np.array([[0,0], [1,0], [1,1]])


@numba.njit
def pc_triangle_intersect(ray_origin, ray_direction, triangle):

    # print("--1--")

    ray_origin = np.array([ray_origin[0], ray_origin[1], ray_origin[2]], dtype=np.float64)
    ray_direction = np.array([ray_direction[0], ray_direction[1], ray_direction[2]], dtype=np.float64)

    p0 = np.array([triangle.vertex_1[0], triangle.vertex_1[1], triangle.vertex_1[2]], dtype=np.float64)
    p1 = np.array([triangle.vertex_2[0], triangle.vertex_2[1], triangle.vertex_2[2]], dtype=np.float64)
    p2 = np.array([triangle.vertex_3[0], triangle.vertex_3[1], triangle.vertex_3[2]], dtype=np.float64)

    # Transform triangle vertices to ray coordinate space
    # Translate vertices based on ray origin
    p0t = p0 - ray_origin
    p1t = p1 - ray_origin
    p2t = p2 - ray_origin

    # print("--2--")

    # Permute components of triangle vertices and ray direction
    kz = max_dimension(np.abs(ray_direction))
    kx = kz + 1
    if kx == 3:
        kx = 0
    ky = kx + 1
    if ky == 3:
        ky = 0
    d = permute(ray_direction, kx, ky, kz)
    p0t = permute(p0t, kx, ky, kz)
    p1t = permute(p1t, kx, ky, kz)
    p2t = permute(p2t, kx, ky, kz)

    # print("--3--")

    # Apply shear transformation to translated vertex positions
    Sx = -d[0] / d[2]
    Sy = -d[1] / d[2]
    Sz = 1.0 / d[2]
    p0t[0] += Sx * p0t[2]
    p0t[1] += Sy * p0t[2]
    p1t[0] += Sx * p1t[2]
    p1t[1] += Sy * p1t[2]
    p2t[0] += Sx * p2t[2]
    p2t[1] += Sy * p2t[2]

    # print("--4--")

    # Compute edge function coefficients e0, e1, and e2
    e0 = p1t[0] * p2t[1] - p1t[1] * p2t[0]
    e1 = p2t[0] * p0t[1] - p2t[1] * p0t[0]
    e2 = p0t[0] * p1t[1] - p0t[1] * p1t[0]

    #TODO: Fall back to double precision test at triangle edges

    # print("--5--")

    # Perform triangle edge and determinant tests
    if (e0 < 0 or e1 < 0 or e2 < 0) and (e0 > 0 or e1 > 0 or e2 > 0):
        return None, None
    det = e0 + e1 + e2
    if det == 0:
        return None, None

    # print("--6--")

    # Compute scaled hit distance to triangle and test against ray $t$ range
    p0t[2] *= Sz
    p1t[2] *= Sz
    p2t[2] *= Sz

    tScaled = e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]

    # print("--7--")

    # ray.tMax=np.inf
    if det < 0 and (tScaled >= 0 or tScaled < np.inf * det):
        return None, None
    elif det > 0 and (tScaled <= 0 or tScaled > np.inf * det):
        return None, None

    # Compute barycentric coordinates and t value for triangle intersection
    invDet = 1 / det
    b0 = e0 * invDet
    b1 = e1 * invDet
    b2 = e2 * invDet
    t = tScaled * invDet

    # print("--8--")

    # Ensure that computed triangle t is conservatively greater than zero

    # Compute delta_z term for triangle t error bounds
    maxZt = max_component(np.abs(np.array([p0t[2], p1t[2], p2t[2]])))
    deltaZ = gamma(3) * maxZt

    # Compute delta_x and delta_y terms for triangle t error bounds
    maxXt = max_component(np.abs(np.array([p0t[0], p1t[0], p2t[0]])))
    maxYt = max_component(np.abs(np.array([p0t[1], p1t[1], p2t[1]])))
    deltaX = gamma(5) * (maxXt + maxZt)
    deltaY = gamma(5) * (maxYt + maxZt)

    # print("--9--")

    # Compute delta_e term for triangle t error bounds
    deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt)

    # Compute delta_t term for triangle t error bounds and check t
    maxE = max_component(np.abs(np.array([e0, e1, e2])))

    # print(maxE)

    deltaT = 3 * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * np.abs(invDet)

    if t <= deltaT:
        return None, None

    # print("--10--")

    # Compute triangle partial derivatives
    uv = get_UVs()

    # Compute deltas for triangle partial derivatives
    duv02 = uv[0] - uv[2]
    duv12 = uv[1] - uv[2]
    dp02 = p0 - p2
    dp12 = p1 - p2

    determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0]

    # print("--11--")

    if determinant<EPSILON:
        # Handle zero determinant for triangle partial derivative matrix
        dpdu, dpdv = create_orthonormal_system(normalize(np.cross(p2 - p0, p1 - p0)))
    else:
        invdet = 1 / determinant
        dpdu = ( duv12[1] * dp02 - duv02[1] * dp12) * invdet
        dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invdet

    # print("--12--")

    ## Compute error bounds for triangle intersection
    # xAbsSum = (np.abs(b0 * p0[0]) + np.abs(b1 * p1[0]) + np.abs(b2 * p2[0]))
    # yAbsSum = (np.abs(b0 * p0[1]) + np.abs(b1 * p1[1]) + np.abs(b2 * p2[1]))
    # zAbsSum = (np.abs(b0 * p0[2]) + np.abs(b1 * p1[2]) + np.abs(b2 * p2[2]))
    # pError = gamma(7) * np.array([xAbsSum, yAbsSum, zAbsSum])

    # Interpolate (u,v) parametric coordinates and hit point
    pHit = b0 * p0 + b1 * p1 + b2 * p2
    # print(pHit)
    uvHit = b0 * uv[0] + b1 * uv[1] + b2 * uv[2]

    # Override surface normal in _isect_ for triangle
    normal = shading_normal = normalize(np.cross(dp02, dp12))

    # Fill in _SurfaceInteraction_ from triangle hit
    # print(pHit.shape)
    # print("Intersected")
    # print(ray_direction)
    # print(dpdu)
    # print(dpdv)
    # print(triangle)
    # print(normal)
    # print(shading_normal)

    # isect = None

    # isect = SurfaceInteraction(intersected_point=pHit,
    #                            incoming_direction=-ray_direction,
    #                            dpdu=dpdu,
    #                            dpdv=dpdv,
    #                            triangle=triangle,
    #                            normal=normal,
    #                            shading_normal=shading_normal)

    isect = SurfaceInteraction(pHit, ray_direction, dpdu, normal, shading_normal)

    return t, isect
