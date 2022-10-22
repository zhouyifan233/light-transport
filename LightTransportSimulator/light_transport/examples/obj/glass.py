import pyvista as pv
import numpy as np
import numba

from LightTransportSimulator.light_transport.src.material import Material
from LightTransportSimulator.light_transport.src.constants import *
from LightTransportSimulator.light_transport.src.primitives import PreComputedTriangle


def design_glass(start_id):
    glass_triangles = []
    points = []

    body = pv.CylinderStructured(radius=np.linspace(6, 7, 5), height=12, center=[0,6.5,0], direction=[0, 1, 0])
    base = pv.Cylinder(radius=7, height=1, center=[0,0,0], direction=[0, 1, 0])
    liquid = pv.Cylinder(radius=6, height=8, center=[0,4.5,0], direction=[0, 1, 0])
    ice = pv.Cube(center=[0,3,0], x_length=5, y_length=5, z_length=5)

    # hard_glass_mat = Material(color=WHITE, shininess=1, reflection=0.9, ior=1.5)
    # soft_glass_mat = Material(color=WHITE, shininess=1, reflection=0.9, ior=1.5)
    water_mat = Material(color=WHITE, shininess=1, reflection=0.9, ior=1.333, transmission=1.0)
    whisky_mat = Material(color=YELLOW, shininess=1, reflection=0.9, ior=1.356, transmission=0.75)
    ice_mat = Material(color=WHITE, shininess=1, reflection=0.9, ior=1.310, transmission=1.0)

    # glass - body
    body = body.extract_surface(nonlinear_subdivision=1)
    tri_glass = body.triangulate()

    glass_points = np.ascontiguousarray(tri_glass.points)
    for p in glass_points:
        points.append(p)
    glass_faces = tri_glass.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(glass_points[glass_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=GLASS_MAT)
        glass_triangles.append(triangle)

    # glass - base
    tri_glass = base.triangulate()

    glass_points = np.ascontiguousarray(tri_glass.points)
    for p in glass_points:
        points.append(p)
    glass_faces = tri_glass.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(glass_points[glass_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=GLASS_MAT)
        glass_triangles.append(triangle)

    # glass - liquid
    tri_glass = liquid.triangulate()

    glass_points = np.ascontiguousarray(tri_glass.points)
    for p in glass_points:
        points.append(p)
    glass_faces = tri_glass.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(glass_points[glass_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=whisky_mat)
        glass_triangles.append(triangle)

    # glass - ice
    tri_glass = ice.triangulate()

    glass_points = np.ascontiguousarray(tri_glass.points)
    for p in glass_points:
        points.append(p)
    glass_faces = tri_glass.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(glass_points[glass_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = np.append(vx[0], 1),np.append(vx[1], 1),np.append(vx[2], 1)
        start_id+=1
        triangle = PreComputedTriangle(id=start_id,
                                       vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                                       vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                                       vertex_3=np.ascontiguousarray(r, dtype=np.float64),
                                       material=ice_mat)
        glass_triangles.append(triangle)

    return glass_triangles, points