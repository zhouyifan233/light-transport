import numba
import numpy as np
from dataclasses import dataclass

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


def get_bounds(prim):
    min_p = np.minimum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    max_p = np.maximum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    bounded_box = AABB(min_p, max_p)

    return bounded_box


def bound_centroid(box, cent):
    min_p = np.minimum(box.min_point, cent)
    max_p = np.maximum(box.max_point, cent)

    bounded_box = AABB(min_p, max_p)

    return bounded_box


@numba.experimental.jitclass([
    ('prim_idx', numba.intp),
    ('morton_code', numba.uint32)
])
class MortonPrimitive:
    def __init__(self, prim_idx, morton_code):
        self.prim_idx = prim_idx
        self.morton_code = morton_code


def offset_bounds(bounds, point):
    o = point - bounds.min_point
    if bounds.max_point[0] > bounds.min_point[0]:
        o[0] /= bounds.max_point[0] - bounds.min_point[0]

    if bounds.max_point[1] > bounds.min_point[1]:
        o[1] /= bounds.max_point[1] - bounds.min_point[1]

    if bounds.max_point[2] > bounds.min_point[2]:
        o[2] /= bounds.max_point[2] - bounds.min_point[2]

    return o


def left_shift(bits):
    if bits == 1 << 10:
        bits = bits-1
    bits = (bits | (bits << 16)) & 0b00000011000000000000000011111111
    bits = (bits | (bits <<  8)) & 0b00000011000000001111000000001111
    bits = (bits | (bits <<  4)) & 0b00000011000011000011000011000011
    bits = (bits | (bits <<  2)) & 0b00001001001001001001001001001001

    return bits


def encode_morton(point):
    a = left_shift(point[2])
    b = left_shift(point[1])
    c = left_shift(point[0])
    return (a << 2) | (b << 1) | c


def radix_sort(morton_primitives):
    tmp_vec = []
    bits_per_pass = 6
    n_bits = 30
    n_passes = n_bits/bits_per_pass

    for p in n_passes:
        # one pass radix sort
        low_bit = p * bits_per_pass
        _in = tmp_vec if p & 1 else morton_primitives
        _out = morton_primitives if p & 1 else tmp_vec
        # Count number of zero bits in array for current radix sort bit>
        n_buckets = 1 << bits_per_pass
        bucket_count = [0 for i in range(n_buckets)]
        bit_mask = (1 << bits_per_pass) - 1

        for mp in _in:
            bucket = (mp.morton_code >> low_bit) & bit_mask
            bucket_count[bucket] += 1

        # Compute starting index in output array for each bucket
        out_index = []
        out_index[0] = 0
        for i in range(1, n_buckets):
            out_index[i] = out_index[i - 1] + bucket_count[i - 1]

        for mp in _in:
            bucket = (mp.morton_code >> low_bit) & bit_mask
            _out[out_index[bucket]] = mp
            out_index[bucket] += 1


    if n_passes & 1:
        return tmp_vec
    else:
        return morton_primitives


class BucketInfo:
    def __init__(self, count, box):
        self.count = count
        self.bounds = box


def tree_partition(param, param1, node, dim):
    centroid = node.bounds.min_point[dim]


def build_upper_sah(treelet_roots, start, end):
    n_nodes = end - start
    if n_nodes == 1:
        return treelet_roots[start]

    bounds = get_bounds(treelet_roots[start])

    for i in range(start, end):
        bounds = bound_centroid(bounds, treelet_roots[i].bounds)

    centroid_bounds = get_bounds(treelet_roots[start])

    for i in range(start, end):
        centroid = 0.5 * (treelet_roots.bounds.min_point + treelet_roots.bounds.max_point)
        centroid_bounds = bound_centroid(centroid_bounds, centroid)

    dim = centroid_bounds.maximun_extent()

    n_buckets = 12
    buckets = BucketInfo(n_buckets, bounds)

    for i in range(start, end):
        centroid = 0.5 * (treelet_roots[i].bounds.min_point+treelet_roots[i].bounds.max_point)
        b = n_buckets * ((centroid-centroid_bounds.min_point[dim])/(centroid_bounds.max_point[dim]-centroid_bounds.min_point[dim]))
        if b==n_buckets:
            b = n_buckets-1

        buckets[b].count += 1 # TODO: Check if required
        buckets[b].bounds = bound_centroid(buckets[b].bounds, treelet_roots[i].bounds)


    # compute cost for splitting each bucket
    costs = []
    for i in range(n_buckets):
        b0 = b1 = get_bounds(buckets[0].bounds)
        count_0 = 0
        count_1 = 0
        for j in range(i+1):
            b0 = bound_centroid(b0, buckets[j].bounds)
            count_0 += buckets[j].count
        for j in (i+1, n_buckets):
            b1 = bound_centroid(b1, buckets[j].bounds)
            count_1 += buckets[j].count

        costs[i] = .125 * (count_0*b0.surface_area()+count_1*b1.surface_area())/bounds.surface_area()


    # find bucket to split at which minimizes SAH metric
    min_cost = costs[0]
    min_cost_split_bucket = 0
    for i in range(1, n_buckets-1):
        if costs[i]<min_cost:
            min_cost = costs[i]
            min_cost_split_bucket = i

    # split nodes and create interior HLBVH SAH node
    # pmid = tree_partition(treelet_roots[start], treelet_roots[end-1]+1)
    centroid = 0.5 * (node.bounds.min_point[dim] + node.bounds.max_point[dim])
    b = n_buckets + ((centroid-centroid_bounds.min_point[dim])/(centroid_bounds.max_point[dim]-centroid_bounds.min_point[dim]))
    if b == n_buckets:
        b = n_buckets-1

    pmid = b<=min_cost_split_bucket

    mid = pmid - treelet_roots[0]

    node.init_interior(dim, build_upper_sah(treelet_roots, start, mid), build_upper_sah(treelet_roots, mid, end))

    return node


def build_bvh(primitives):
    bounds = get_bounds(primitives[0]) # get AABB of the first primitive

    # compute AABB for all primitives
    for prim in primitives:
        bounds = bound_centroid(bounds, prim.centroid)

    # compute morton indices of all primitives
    morton_primitives = []
    #TODO: make it parallel
    for i in range(len(primitives)):
        morton_bits = 10
        morton_scale = 1 << morton_bits
        centroid_offset = offset_bounds(bounds, primitives[i].centroid)
        encoded_bits = encode_morton(centroid_offset*morton_scale)
        m_prim = MortonPrimitive(prim_idx=i, morton_code=encoded_bits)
        morton_primitives.append(m_prim)

    # Radix sort morton indices
    sorted_morton_prims = radix_sort(morton_primitives)

    # Create LBVH treelets at bottom of BVH
    lbvh_treelets = []
    # Find intervals of primitives for each treelet
    start = 0
    for end in range(1, len(sorted_morton_prims)):
        mask = 0b00111111111111000000000000000000
        if (end == len(sorted_morton_prims) | ((sorted_morton_prims[start].morton_code & mask) != (sorted_morton_prims[end].morton_code & mask))):
            n_primitives = end - start
            max_bvh_nodes = 2 * n_primitives - 1
            nodes = max_bvh_nodes # TODO: Update this function
            treelet = LBVHTreelet(start, n_primitives, nodes)
            lbvh_treelets.append(treelet)
            start = end


    # Create LBVHs for treelets in parallel
    for i in range(len(lbvh_treelets)):
        first_bit_index = 29 - 12
        lbvh_treelets[i].build_nodes = emit_lbvh()

    final_treelets = []
    for _tree in lbvh_treelets:
        final_treelets.append(_tree.build_nodes)
    return build_upper_sah()








