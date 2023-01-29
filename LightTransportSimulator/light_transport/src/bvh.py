import numba
import numpy as np

from .intersects import aabb_intersect
from .primitives import AABB, Triangle, PreComputedTriangle


node_type = numba.deferred_type()

@numba.experimental.jitclass([
    ('root', numba.types.ListType(PreComputedTriangle.class_type.instance_type)),
    ('axis', numba.intp),
    ('box', AABB.class_type.instance_type),
    ('left', numba.optional(node_type)),
    ('right', numba.optional(node_type))
])
class Volume:
    def __init__(self, primitives, axis, box):
        self.root = primitives
        self.axis = axis
        self.box = box
        self.left = None
        self.right = None

    # def print(self):
    #     print(self.root)


node_type.define(Volume.class_type.instance_type)


@numba.njit
def traverse_bvh(bvh, ray_origin, ray_direction):
    shapes = numba.typed.List()

    if aabb_intersect(ray_origin, ray_direction, bvh.box):
        if bvh.left is None and bvh.right is None:
            for _s in bvh.root:
                # if _s not in shapes:
                shapes.append(_s)
        if bvh.left is not None:
            for _s in traverse_bvh(bvh.left, ray_origin, ray_direction):
                # if _s not in shapes:
                shapes.append(_s)
        if bvh.right is not None:
            for _s in traverse_bvh(bvh.right, ray_origin, ray_direction):
                # if _s not in shapes:
                shapes.append(_s)

    return shapes


def get_volumes(bvh):
    shapes = []

    if bvh.left is None and bvh.right is None:
        for _s in bvh.root:
            # if _s not in shapes:
            shapes.append(_s)
    if bvh.left is not None:
        for _s in get_volumes(bvh.left):
            # if _s not in shapes:
            shapes.append(_s)
    if bvh.right is not None:
        for _s in get_volumes(bvh.right):
            # if _s not in shapes:
            shapes.append(_s)

    return shapes



class BVH:
    def __init__(self):
        self.top = None

    def create_hierarchy(self, vol, prm, box, min_leaf_size, from_left=None, last_vol=None):
        if vol is None:
            ax = (box.max_point-box.min_point).argmax() # longest axis
            # create volume
            vol = Volume(prm, ax, box)

            if len(prm)>min_leaf_size:

                _left = numba.typed.List.empty_list(numba.typeof(prm[0]))
                left_min_coords = []
                left_max_coords = []
                _right = numba.typed.List.empty_list(numba.typeof(prm[0]))
                right_min_coords = []
                right_max_coords = []

                for _p in prm:
                    max_coords = np.maximum.reduce([_p.vertex_1, _p.vertex_2, _p.vertex_3])
                    min_coords = np.minimum.reduce([_p.vertex_1, _p.vertex_2, _p.vertex_3])
                    if min_coords[ax]<=box.centroid[ax]:
                        _left.append(_p)
                        left_min_coords.append(min_coords)
                        left_max_coords.append(max_coords)

                    if max_coords[ax]>=box.centroid[ax]:
                        _right.append(_p)
                        right_min_coords.append(min_coords)
                        right_max_coords.append(max_coords)

                box_left_min = np.minimum.reduce(left_min_coords)
                box_left_max = np.maximum.reduce(left_max_coords)
                box_right_min = np.minimum.reduce(right_min_coords)
                box_right_max = np.maximum.reduce(right_max_coords)

                if np.array_equal(box_left_min, box_right_min) or np.array_equal(box_left_max, box_right_max):
                    # identical boxes on both sides
                    return vol

                if from_left is not None:
                    if from_left:
                        if last_vol==len(_left):
                            # last volume is a leaf node
                            return vol
                    else:
                        if last_vol==len(_right):
                            # last volume is a leaf node
                            return vol

                if len(_left)>0:
                    # create left sub-hierarchy
                    left_box = AABB(min_point=box_left_min, max_point=box_left_max)
                    vol.left = self.create_hierarchy(vol.left, _left, left_box, min_leaf_size, True, len(_left))


                if len(_left) == len(_right) == len(prm):
                    # right sub-hierarchy not required
                    return vol


                if len(_right)>0:
                    # create right sub-hierarchy
                    right_box = AABB(min_point=box_right_min, max_point=box_right_max)
                    vol.right = self.create_hierarchy(vol.right, _right, right_box, min_leaf_size, False, len(_right))

            return vol

    def insert(self, shapes, bounding_box):
        min_leaf_size = len(shapes)*0.1
        min_leaf_size = 2 if min_leaf_size < 2 else round(min_leaf_size)
        self.top = self.create_hierarchy(self.top, shapes, bounding_box, min_leaf_size)




