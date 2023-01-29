import numba
import numpy as np

from .intersects import aabb_intersect, triangle_intersect, intersect_bounds
from .primitives import AABB, Triangle, PreComputedTriangle
from .stl4py import nth_element, partition


class BoundedBox:
    def __init__(self, prim):
        self.prim = prim
        self.bounds = get_bounds(prim)


node_type = numba.deferred_type()

@numba.experimental.jitclass([
    ('bounds', numba.optional(AABB.class_type.instance_type)),
    ('child_0', numba.optional(numba.intp)),
    ('child_1', numba.optional(numba.intp)),
    ('split_axis', numba.optional(numba.intp)),
    ('n_primitives', numba.optional(numba.intp))
])
class BVHNode:
    def __init__(self):
        self.bounds = None
        self.child_0 = None
        self.child_1 = None
        self.split_axis = None
        self.n_primitives = None


node_type.define(BVHNode.class_type.instance_type)


@numba.experimental.jitclass([
    ('bounds', numba.optional(AABB.class_type.instance_type)),
    ('primitives_offset', numba.optional(numba.intp)),
    ('second_child_offset', numba.optional(numba.intp)),
    ('n_primitives', numba.optional(numba.intp)),
    ('axis', numba.optional(numba.intp))
])
class LinearBVHNode:
    def __init__(self):
        self.bounds = None
        self.primitives_offset = None
        self.second_child_offset = None
        self.n_primitives = None
        self.axis = None
        # self.pad = [0]


class BucketInfo:
    def __init__(self):
        self.count = 0
        self.bounds = None


def get_bounds(prim):
    min_p = np.minimum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    max_p = np.maximum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    bounded_box = AABB(min_p, max_p)

    return bounded_box


@numba.njit
def enclose_volumes(box_1, box_2):
    if box_1 is not None:
        if box_2 is None:
            bounded_box = box_1
        else:
            min_p = np.minimum(box_1.min_point, box_2.min_point)
            max_p = np.maximum(box_1.max_point, box_2.max_point)
            bounded_box = AABB(min_p, max_p)
    else:
        bounded_box = box_2
    return bounded_box


def enclose_centroids(box, cent):
    if box is not None:
        min_p = np.minimum(box.min_point, cent)
        max_p = np.maximum(box.max_point, cent)
        bounded_box = AABB(min_p, max_p)
    else:
        bounded_box = AABB(cent, cent)

    return bounded_box


def get_largest_dim(box):
    dx = abs(box.max_point[0] - box.min_point[0])
    dy = abs(box.max_point[1] - box.min_point[1])
    dz = abs(box.max_point[2] - box.min_point[2])
    if dx > dy and dx > dz:
        return 0
    elif dy > dz:
        return 1
    else:
        return 2


def get_surface_area(box):
    diagonal = box.max_point - box.min_point
    surface_area = 2 * (diagonal[0] * diagonal[1] + diagonal[0] * diagonal[2] + diagonal[1] * diagonal[2])
    return surface_area


def offset_bounds(bounds, point):
    o = point - bounds.min_point
    if bounds.max_point[0] > bounds.min_point[0]:
        o[0] /= bounds.max_point[0] - bounds.min_point[0]

    if bounds.max_point[1] > bounds.min_point[1]:
        o[1] /= bounds.max_point[1] - bounds.min_point[1]

    if bounds.max_point[2] > bounds.min_point[2]:
        o[2] /= bounds.max_point[2] - bounds.min_point[2]

    return o


def partition_pred(x, n_buckets, centroid_bounds, dim, min_cost_split_bucket):
    b = n_buckets*offset_bounds(centroid_bounds, x.bounds.centroid)[dim]
    if b==n_buckets:
        b = n_buckets-1
    return b <= min_cost_split_bucket


def build_bvh(bvh_nodes, bounded_boxes, start, end, max_prims=1):
    # append a node
    bvh_nodes.append(BVHNode())
    node = bvh_nodes[-1]
    n_primitives = start - end
    if n_primitives<=max_prims:
        node.n_primitives = n_primitives
        for i in range(start, end):
            node.bounds = enclose_volumes(node.bounds, bounded_boxes[i].bounds)
        node.child_0 = start - 0
    else:
        node.n_primitives = 0
        centroid_bounds = None
        for i in range(start, end):
            centroid_bounds = enclose_centroids(centroid_bounds, bounded_boxes[i].bounds.centroid)
        node.split_axis = get_largest_dim(centroid_bounds)
        mid = start + (end-start)//2
        nth_element(bounded_boxes, mid, first=start, last=end, key=lambda x: x.bounds.centroid[node.split_axis])

        child_0_idx = len(bvh_nodes)
        bvh_nodes, bounded_boxes = build_bvh(bvh_nodes, bounded_boxes, start, mid)
        node.child_1 = len(bvh_nodes)
        bvh_nodes, bounded_boxes = build_bvh(bvh_nodes, bounded_boxes, mid, end)
        node.bounds = enclose_volumes(bvh_nodes[child_0_idx].bounds, bvh_nodes[node.child_1].bounds)

    return bvh_nodes, bounded_boxes



@numba.njit
def flatten_bvh(linear_nodes, node, offset):
    """performs a depth-first traversal and
    stores the nodes in memory in linear order"""
    linear_nodes[offset].bounds = node.bounds
    _offset = offset
    if node.n_primitives>0:
        # leaf node
        linear_nodes[offset].primitives_offset = node.first_prim_offset
        linear_nodes[offset].n_primitives = node.n_primitives
    else:
        # create interior flattened bvh node
        linear_nodes[offset].axis = node.split_axis
        linear_nodes[offset].n_primitives = 0
        #TODO: fix this
        linear_nodes, offset = flatten_bvh(linear_nodes, node.child_0, offset+1)
        linear_nodes, linear_nodes[_offset].second_child_offset = flatten_bvh(linear_nodes, node.child_1, offset+1)
        offset = linear_nodes[_offset].second_child_offset

    return linear_nodes, offset


@numba.njit
def intersect_bvh(ray_origin, ray_direction, linear_bvh, primitives):
    triangles = []
    hit = False

    inv_dir = 1/ray_direction
    dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
    to_visit_offset = 0
    current_node_index = 0
    nodes_to_visit = [0 for _ in range(64)] #

    while True:
        node = linear_bvh[int(current_node_index)]
        if intersect_bounds(node.bounds, ray_origin, ray_direction, inv_dir, dir_is_neg):
        # if aabb_intersect(ray_origin, ray_direction, node.bounds):
            if node.n_primitives > 0:
                for i in range(node.n_primitives):
                    if triangle_intersect(ray_origin, ray_direction, primitives[node.primitives_offset + i]) is not None:
                        # print("intersected!!!")
                        hit = True
                        triangles.append(primitives[node.primitives_offset + i])
                if to_visit_offset == 0:
                    break
                current_node_index = nodes_to_visit[to_visit_offset-1]
                to_visit_offset -= 1
            else:
                if dir_is_neg[node.axis]:
                    nodes_to_visit[to_visit_offset] = current_node_index + 1
                    to_visit_offset += 1
                    current_node_index = node.second_child_offset
                else:
                    nodes_to_visit[to_visit_offset] = node.second_child_offset
                    to_visit_offset += 1
                    current_node_index += 1
        else:
            if to_visit_offset == 0:
                break

            current_node_index = nodes_to_visit[to_visit_offset-1]
            to_visit_offset -= 1

    return triangles