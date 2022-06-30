from pathlib import Path

import pywavefront
import numpy as np

from .colors import RED
from .material import Material
from .primitives import Triangle


def load_obj(file_path):
    #load object
    path = Path(file_path).resolve()
    object = pywavefront.Wavefront(
        path,
        create_materials=True,
        collect_faces=True
    )

    obj_faces = object.mesh_list[0].faces
    obj_vertices = object.vertices

    #set scene dimensions
    vx = np.ascontiguousarray(obj_vertices)
    xmax, ymax, zmax = vx.max(axis=0)

    dimension = abs(max(xmax, ymax, zmax))

    objects = []

    for face in obj_faces:
        a,b,c = obj_vertices[face[0]],obj_vertices[face[1]],obj_vertices[face[2]]
        obj_mat = Material(color=RED, shininess=100, reflection=0.5)
        triangle = Triangle(vertex_1=np.asarray(a, dtype=np.float64),
                            vertex_2=np.asarray(b, dtype=np.float64),
                            vertex_3=np.asarray(c, dtype=np.float64),
                            material=obj_mat)
        objects.append(triangle)

    return objects, dimension

