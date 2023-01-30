import numba
import numpy as np

from .constants import EPSILON
from .intersects import aabb_intersect, triangle_intersect, intersect_bounds
from .primitives import AABB, Triangle, PreComputedTriangle
from .stl4py import nth_element, partition



class BoundedBox:
    def __init__(self, prim, n):
        self.prim = prim
        self.prim_num = n
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


def build_bvh(bvh_nodes, bounded_boxes, start, end):
    max_prims = 1
    # append a node
    bvh_nodes.append(BVHNode())
    node = bvh_nodes[-1]

    n_primitives = end - start
    if n_primitives <= max_prims:
        # make node a leaf
        node.n_primitives = n_primitives
        for i in range(start, end):
            node.bounds = enclose_volumes(node.bounds, bounded_boxes[i].bounds)
        node.child_0 = start
    else:
        # make node an interior node
        node.n_primitives = 0

        centroid_bound = None
        for i in range(start, end):
            centroid_bound = enclose_centroids(centroid_bound, bounded_boxes[i].bounds.centroid)
        node.split_axis = get_largest_dim(centroid_bound)
        mid = start + (end - start) // 2
        bounded_boxes[start:end] = sorted(bounded_boxes[start:end], key=lambda x: x.bounds.centroid[node.split_axis], reverse=False)

        # recursively call for left and right child and note the index of the second child in-between
        child_0_idx = len(bvh_nodes)
        bvh_nodes, bounded_boxes = build_bvh(bvh_nodes, bounded_boxes, start, mid)
        node.child_1 = len(bvh_nodes)
        bvh_nodes, bounded_boxes = build_bvh(bvh_nodes, bounded_boxes, mid, end)

        # the world bound of this node encloses the ones of both children
        # print(len(bvh_nodes), child_0_idx, node.child_1)
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
def intersect_bvh(ray_origin, ray_direction, bvh_nodes, primitives):
    triangles = []
    current_t = np.inf
    current_idx = 0
    inv_dir = 1/ray_direction
    dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
    visited_nodes = [False for _ in range(len(bvh_nodes))]
    while True:
        if not visited_nodes[current_idx]:
            visited_nodes[current_idx] = True
            node = bvh_nodes[int(current_idx)]
            # print("checking node: "+ str(current_idx))
            if intersect_bounds(node.bounds, ray_origin, ray_direction, inv_dir, dir_is_neg):
                # check if it's a leaf node
                if node.n_primitives>0:
                    # print("intersected node: "+ str(current_idx))
                    for i in range(node.child_0, node.child_0+node.n_primitives):
                        t = triangle_intersect(ray_origin, ray_direction, primitives[i])
                        if t is None:
                            continue
                        if EPSILON < t < current_t:
                            current_t = t
                            triangles.append(primitives[i])
                else:
                    if dir_is_neg[node.split_axis]:
                        current_idx = node.child_1
                    else:
                        for i in range(len(visited_nodes)):
                            if not visited_nodes[i]:
                                current_idx = i
                                break
            else:
                for i in range(len(visited_nodes)):
                    if not visited_nodes[i]:
                        current_idx = i
                        break
        else:
            for i in range(len(visited_nodes)):
                if not visited_nodes[i]:
                    current_idx = i
                    break
        all_visited = True
        for i in range(len(visited_nodes)):
            if not visited_nodes[i]:
                all_visited=False
                break
        if all_visited:
            break

    return triangles



# @numba.njit
# def intersect_bvh(ray_origin, ray_direction, bvh_nodes, primitives):
#     triangles = []
#     current_t = np.inf
#
#     inv_dir = 1/ray_direction
#     dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
#     to_visit_offset = 0
#     current_node_index = 0
#     nodes_to_visit = [1000 for _ in range(64)] #
#
#     while True:
#         node = bvh_nodes[int(current_node_index)]
#         if intersect_bounds(node.bounds, ray_origin, ray_direction, inv_dir, dir_is_neg):
#         # if aabb_intersect(ray_origin, ray_direction, node.bounds):
#             if node.n_primitives > 0:
#                 for i in range(node.child_0, node.child_0+node.n_primitives):
#                     t = triangle_intersect(ray_origin, ray_direction, primitives[i])
#                     if t is None:
#                         continue
#                     if EPSILON < t < current_t:
#                         current_t = t
#                         triangles.append(primitives[i])
#                 if to_visit_offset == 0:
#                     break
#                 to_visit_offset -= 1
#                 current_node_index = nodes_to_visit[to_visit_offset]
#             else:
#                 if dir_is_neg[node.split_axis]:
#                     nodes_to_visit[to_visit_offset] = current_node_index + 1
#                     to_visit_offset += 1
#                     current_node_index = node.child_1
#                 else:
#                     nodes_to_visit[to_visit_offset] = node.child_1
#                     to_visit_offset += 1
#                     current_node_index += 1
#         else:
#             if to_visit_offset == 0:
#                 break
#
#             to_visit_offset -= 1
#             current_node_index = nodes_to_visit[to_visit_offset]
#
#     return triangles