import numpy as np

from .material import Color, Material

WHITE = Color(ambient=np.array([1, 1, 1], dtype=np.float64),
              diffuse=np.array([1, 1, 1], dtype=np.float64),
              specular=np.array([1, 1, 1], dtype=np.float64))

WHITE_2 = Color(ambient=np.array([0, 0, 0], dtype=np.float64),
              diffuse=np.array([0.55, 0.55, 0.55], dtype=np.float64),
              specular=np.array([0.7, 0.7, 0.7], dtype=np.float64))

RED = Color(ambient=np.array([0.1, 0, 0], dtype=np.float64),
            diffuse=np.array([0.7, 0, 0], dtype=np.float64),
            specular=np.array([1, 1, 1], dtype=np.float64))

PURPLE = Color(ambient=np.array([0.1, 0, 0.1], dtype=np.float64),
            diffuse=np.array([0.7, 0, 0.7], dtype=np.float64),
            specular=np.array([1, 1, 1], dtype=np.float64))

YELLOW = Color(ambient=np.array([0.05, 0.05, 0.0], dtype=np.float64),
               diffuse=np.array([0.5, 0.5, 0.4], dtype=np.float64),
               specular=np.array([0.7, 0.7, 0.04], dtype=np.float64))


SILVER = Color(ambient=np.array([0.23125, 0.23125, 0.23125], dtype=np.float64),
               diffuse=np.array([0.2775, 0.2775, 0.2775], dtype=np.float64),
               specular=np.array([0.773911, 0.773911, 0.773911], dtype=np.float64))

GREEN = Color(ambient=np.array([0, 0.1, 0], dtype=np.float64),
              diffuse=np.array([0, 0.6, 0], dtype=np.float64),
              specular=np.array([1, 1, 1], dtype=np.float64))

GREY = Color(ambient=np.array([0.1, 0.1, 0.1], dtype=np.float64),
              diffuse=np.array([0.6, 0.6, 0.6], dtype=np.float64),
              specular=np.array([1, 1, 1], dtype=np.float64))

TURQUOISE = Color(ambient=np.array([0.1, 0.18725, 0.1745], dtype=np.float64),
                  diffuse=np.array([0.396, 0.74151, 0.69102], dtype=np.float64),
                  specular=np.array([0.297254, 0.30829, 0.306678], dtype=np.float64))
TURQUOISE_MAT = Material(color=TURQUOISE, shininess=0.1, reflection=2, ior=1.65)

BRONZE = Color(ambient=np.array([0.2125, 0.1275, 0.054], dtype=np.float64),
                  diffuse=np.array([0.714, 0.4284, 0.18144], dtype=np.float64),
                  specular=np.array([0.393548, 0.271906, 0.166721], dtype=np.float64))
BRONZE_MAT = Material(color=PURPLE, shininess=10, reflection=0.2, ior=1.180)

GLASS = Color(ambient=np.array([0.0, 0.0, 0.0], dtype=np.float64),
              diffuse=np.array([0.588235, 0.670588, 0.729412], dtype=np.float64),
              specular=np.array([0.9, 0.9, 0.9], dtype=np.float64))
GLASS_MAT = Material(color=GLASS, shininess=96, reflection=0.2, ior=1.5, transmission=1.0)