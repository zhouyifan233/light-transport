import numba
import numpy as np
import matplotlib.pyplot as plt
import time

from .io import load_obj
from .render import render_rt

from .scene import Light, Scene
from .material import Color, Material
from .cornell_box import get_cornell_box
from .colors import *


if __name__ == '__main__':

    objects, dimension = load_obj("examples/obj/cube.obj")

    #setup cornell box
    surface_color = SILVER
    left_wall_color = MAROON
    right_wall_color = GREEN

    surface_mat = Material(color=surface_color, shininess=89.6, reflection=0.5)
    left_wall_mat = Material(color=left_wall_color, shininess=100, reflection=0.5)
    right_wall_mat = Material(color=right_wall_color, shininess=100, reflection=0.5)

    c_box = get_cornell_box(dimension, surface_mat, left_wall_mat, right_wall_mat)

    for t in c_box:
        objects.append(t)

    # setup scene

    #light
    source_mat = Material(color=WHITE, shininess=1, reflection=1)
    light_source = Light(source=np.array([0, 0, dimension+2], dtype=np.float64), material=source_mat)

    #camera
    camera = np.array([0, 0, dimension+1], dtype=np.float64)

    #screen
    width=400
    height=400
    max_depth=3

    #scene
    scene = Scene(camera=camera, light=light_source, width=width, height=height, max_depth=max_depth)

    start = time.time()
    image = render_rt(scene, objects)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    plt.imshow(image)

