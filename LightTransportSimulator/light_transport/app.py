import math
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import numba
import pyvista as pv

from src.bvh import BVH
from src.cornell_box import get_floor, get_cornell_box
from src.primitives import PreComputedTriangle, AABB
from src.scene import Light, Scene
from src.constants import *
from src.render import render_scene



st.set_page_config(page_title='Light Transport Simulator', page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-hxt7ib {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title("Light Transport Simulator")

object = st.sidebar.selectbox('Select object', ['Cube', 'Cone', 'Teapot', 'Sphere', 'Cow', 'Upload File ...'])
mesh = pv.Cube() # default object

# initialize objects
objects = numba.typed.List()

# background
background = st.sidebar.selectbox('Select background', ['None', 'Platform', 'Cornell Box'])

# light sources
light_sources = numba.typed.List()
l_source = st.sidebar.selectbox('Select Light Source', ['Point', 'Square', 'Sphere'])

lcol_1, lcol_2, lcol_3 = st.sidebar.columns(3)

setup_expanded = True

with st.expander("Setup Scene", expanded=setup_expanded):
    col1, col2, col3 = st.columns(3)
    with col1:
        # object specs
        st.subheader("Object Specs")
        if object=='Cube':
            x_length = st.number_input('Enter X length', value=1)
            y_length = st.number_input('Enter Y length', value=1)
            z_length = st.number_input('Enter Z length', value=1)
            mesh = pv.Cube(x_length=x_length, y_length=y_length, z_length=z_length)

        elif object=='Cone':
            o_r = st.number_input('Radius', value=2)
            oc_x = st.number_input('Center X', value=0)
            oc_y = st.number_input('Center Y', value=0)
            oc_z = st.number_input('Center Z', value=0)
            o_h = st.number_input('height', value=5)
            mesh = pv.Cone(radius=o_r, center=[oc_x, oc_y, oc_z], height=o_h)

        elif object=='Sphere':
            o_r = st.number_input('Radius', value=2)
            oc_x = st.number_input('Center X', value=0)
            oc_y = st.number_input('Center Y', value=0)
            oc_z = st.number_input('Center Z', value=0)
            mesh = pv.Sphere(radius=o_r, center=[oc_x, oc_y, oc_z])

        elif object=='Upload File ...':
            uploaded_file = st.file_uploader('Choose a .obj file', type=["obj"], accept_multiple_files=False)
            if uploaded_file is not None:
                os.mkdir("tempDir")
                file_path = os.path.join("tempDir", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                path = Path(file_path).resolve()
                mesh = pv.read(path)
                shutil.rmtree(Path("tempDir").resolve())
                st.info("Uploaded "+uploaded_file.name)


        else:
            st.error("Method not implemented")

    tri = mesh.triangulate()
    points = np.ascontiguousarray(tri.points)
    faces = tri.faces.reshape((-1,4))[:, 1:4]
    vertices = np.ascontiguousarray(points[faces], dtype=np.float64)
    xmax, ymax, zmax = points.max(axis=0)
    xmin, ymin, zmin = points.min(axis=0)


    with col2:
        # background specs
        st.subheader("Background")
        padding = st.number_input('Add padding', value=5.0)

        depth = abs(max(xmax, ymax, zmax)) + padding
        x_depth = abs(xmax) + 10
        y_depth = abs(ymax) + 1
        z_depth = abs(zmax) + 10

        if background=='Platform':
            s_color = st.selectbox('Select Surface Color', ['Green', 'Yellow', 'White'])
            if s_color=='Green':
                surface_color=GREEN
            surface_shininess = st.number_input('Shininess', value=90.0)
            surface_reflection = st.number_input('Reflection', value=0.1)
            surface_mat = Material(color=surface_color, shininess=surface_shininess, reflection=surface_reflection, ior=1.460)
            start_id = len(objects)
            # sub_divide = st.number_input('Sub divide faces by', value=0)
            objects = get_floor(x_depth, y_depth, z_depth, surface_mat, start_id)

        elif background=='Cornell Box':
            start_id = len(objects)
            surface_color = WHITE_2
            left_wall_color = GREEN
            right_wall_color = RED
            surface_mat = Material(color=surface_color, shininess=30, reflection=0.1, ior=1.5210, transmission=1) # calcium sulphate
            left_wall_mat = Material(color=left_wall_color, shininess=30, reflection=0.1, ior=1.5210, transmission=1)
            right_wall_mat = Material(color=right_wall_color, shininess=30, reflection=0.1, ior=1.5210, transmission=1)
            # objects = get_cornell_box(depth, surface_mat, left_wall_mat, right_wall_mat, start_id)
            objects = get_cornell_box(depth, surface_mat, left_wall_mat, right_wall_mat, start_id)


    with col3:
        # lights & camera
        st.subheader("Light Specs")
        l_color = st.selectbox('Light Color', ['White'])
        if l_color=='White':
            light_color=WHITE
        light_shininess = st.number_input('Shininess', value=1)
        light_reflection = st.number_input('Reflection', value=0.9)
        source_mat = Material(color=light_color, shininess=light_shininess, reflection=light_reflection, ior=1.5, emission=10000)
        if l_source == 'Point':
            l_x = lcol_1.number_input('Light X', value=3.0)
            l_y = lcol_2.number_input('Light Y', value=5.0)
            l_z = lcol_3.number_input('Light Z', value=3.0)
            l_pos = np.array([l_x, l_y, l_z, 1], dtype=np.float64)
            l1 = Light(source=l_pos, material=source_mat)
            light_sources.append(l1)
        elif l_source == 'Square':
            l_x = lcol_1.number_input('Light X', value=1)
            l_y = lcol_2.number_input('Light Y', value=depth)
            l_z = lcol_3.number_input('Light Z', value=1)

            id = len(objects)+1

            light_1 = PreComputedTriangle(id=id,
                                          vertex_1=np.array([-1, depth, -1, 1], dtype=np.float64),
                                          vertex_2=np.array([1, depth, 1, 1], dtype=np.float64),
                                          vertex_3=np.array([-1, depth, 1, 1], dtype=np.float64),
                                          material=source_mat,
                                          is_light=True)
            id += 1

            light_2 = PreComputedTriangle(id=id,
                                          vertex_1=np.array([-1, depth, -1, 1], dtype=np.float64),
                                          vertex_2=np.array([1, depth, -1, 1], dtype=np.float64),
                                          vertex_3=np.array([1, depth, 1, 1], dtype=np.float64),
                                          material=source_mat,
                                          is_light=True)


            light_samples = 10
            a = np.random.uniform(0,1,light_samples)
            b = np.random.uniform(1,0,light_samples)

            for x in range(light_samples):
                tp1 = light_1.vertex_1 * (1-math.sqrt(a[x])) + light_1.vertex_2 * (math.sqrt(a[x])*(1-b[x])) + light_1.vertex_3 * (b[x]*math.sqrt(a[x]))
                l1 = Light(source=tp1, material=source_mat)
                light_sources.append(l1)
                tp2 = light_2.vertex_1 * (1-math.sqrt(a[x])) + light_2.vertex_2 * (math.sqrt(a[x])*(1-b[x])) + light_2.vertex_3 * (b[x]*math.sqrt(a[x]))
                l2 = Light(source=tp1, material=source_mat)
                light_sources.append(l2)

            objects.append(light_1)
            objects.append(light_2)


# camera position
cam_1, cam_2, cam_3 = st.sidebar.columns(3)

cam_x = cam_1.number_input('Camera X', value=0.0)
cam_y = cam_2.number_input('Camera Y', value=0.0)
cam_z = cam_3.number_input('Camera Z', value=depth+0.5)
camera = np.array([cam_x, cam_y, cam_z, 1], dtype=np.float64)


# screen
s_1, s_2, s_3 = st.sidebar.columns(3)

width=s_1.number_input('Screen Width', value=400, step=100)
height=s_2.number_input('Screen Height', value=400, step=100)
max_depth=s_3.number_input('Scene Depth', value=3)


render = st.sidebar.button('Render')


#setup loaded objects
id = len(objects)+1
for v in vertices:
    a,b,c = np.append(v[0], 1),np.append(v[1], 1),np.append(v[2], 1)
    id+=1
    triangle = PreComputedTriangle(id=id,
                                   vertex_1=np.ascontiguousarray(a, dtype=np.float64),
                                   vertex_2=np.ascontiguousarray(b, dtype=np.float64),
                                   vertex_3=np.ascontiguousarray(c, dtype=np.float64),
                                   material=GLASS_MAT)
    objects.append(triangle)


#scene
scene = Scene(camera=camera, lights=light_sources, width=width, height=height, max_depth=max_depth, depth=depth)

if render:
    # render scene
    setup_expanded=False
    # enclose the scene in a box
    min_point=np.array([-depth, -depth, depth], dtype=np.float64)
    max_point=np.array([depth, depth, -depth], dtype=np.float64)
    box = AABB(min_point=min_point, max_point=max_point)
    # create a BVH tree
    bvh = BVH()
    bvh.insert(objects, box)
    with st.expander("Scene", expanded=True):
        with st.spinner('Loading...'):
            start = time.time()
            image = render_scene(scene, bvh.top, number_of_samples=10)
            end = time.time()

        logs, plots = st.columns(2)

        if image is not None:
            with logs.container():
                logs.success("Elapsed time = %s seconds" % (end - start))
                logs.info("Depth of the box: "+ str(2*depth))
                logs.info("Number of triangles: "+ str(len(objects)))
            fig = plt.figure()
            plt.imshow(image)
            with plots.container():
                st.pyplot(fig)






