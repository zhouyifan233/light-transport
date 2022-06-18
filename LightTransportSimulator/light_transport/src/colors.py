import numpy as np

from LightTransportSimulator.light_transport.src.material import Color


WHITE = Color(ambient=np.array([1, 1, 1], dtype=np.float64),
              diffuse=np.array([1, 1, 1], dtype=np.float64),
              specular=np.array([1, 1, 1], dtype=np.float64))

RED = Color(ambient=np.array([0.1, 0, 0], dtype=np.float64),
            diffuse=np.array([0.7, 0, 0], dtype=np.float64),
            specular=np.array([1, 1, 1], dtype=np.float64))

SILVER = Color(ambient=np.array([0.23125, 0.23125, 0.23125], dtype=np.float64),
               diffuse=np.array([0.2775, 0.2775, 0.2775], dtype=np.float64),
               specular=np.array([0.773911, 0.773911, 0.773911], dtype=np.float64))

MAROON = Color(ambient=np.array([0.1, 0, 0], dtype=np.float64),
               diffuse=np.array([0.7, 0, 0], dtype=np.float64),
               specular=np.array([1, 1, 1], dtype=np.float64))

GREEN = Color(ambient=np.array([0, 0.1, 0], dtype=np.float64),
              diffuse=np.array([0, 0.6, 0], dtype=np.float64),
              specular=np.array([1, 1, 1], dtype=np.float64))