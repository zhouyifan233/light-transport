import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.bvh import traverse_bvh
from LightTransportSimulator.light_transport.src.constants import inv_pi, EPSILON, inv_4_pi
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.scene import Light
from LightTransportSimulator.light_transport.src.utils import nearest_intersected_object, uniform_hemisphere_sampling, cosine_weighted_hemisphere_sampling
from LightTransportSimulator.light_transport.src.vectors import normalize


def henyey_greenstein(cosTheta, g):
    denom = 1 + g**2 + 2 * g * cosTheta
    return inv_4_pi * (1 - g**2) / (denom * np.sqrt(denom))