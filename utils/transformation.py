import numpy as np
import random
from abc import ABC, abstractmethod

SEED = 0x5EED

class Transformation(ABC):
    @abstractmethod
    def transform(points):
        pass

class Normalization(Transformation):
    @staticmethod
    def transform(points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
        points /= furthest_distance
        return points

class Rotation(Transformation):
    def __init__(self):
        self.rng = random.Random(SEED)

    def transform(self, points):
        # variables
        roll = np.deg2rad(self.rng.uniform(-180, 180))
        pitch = np.deg2rad(self.rng.uniform(-90, 90))    # googlear gimbal lock
        yaw = np.deg2rad(self.rng.uniform(-180, 180))
        centroid = np.mean(points, axis=0)

        # matrices de rotación
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx

        # rotación
        shifted = points - centroid
        rotated = shifted @ R.T
        return rotated + centroid

"""
Aplica una translación aleatoria en cada eje
max_units: unidades a transladar (positivo o negativo) por cada eje
normalize: normalizar despues de la translación
"""
class Translation(Transformation):
    def __init__(self, max_units=10, normalize=False):
        self.max_units=max_units
        self.normalize=normalize
        self.rng = random.Random(SEED)

    def transform(self, points):
        # variables
        delta_x = self.rng.uniform(-self.max_units, self.max_units)
        delta_y = self.rng.uniform(-self.max_units, self.max_units)
        delta_z = self.rng.uniform(-self.max_units, self.max_units)

        # translación
        points[:, 0] += delta_x
        points[:, 1] += delta_y
        points[:, 2] += delta_z

        # normalización (sin centroide)
        if self.normalize:
            furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
            points /= furthest_distance
        return points

class Reflection(Transformation):
    def __init__(self):
        self.rng = random.Random(SEED)

    def transform(self, points):
        # variables
        on_x = random.choice(False, True)
        on_y = random.choice(False, True)
        on_z = random.choice(False, True)

        # reflexión
        if on_x: points[:, 0] *= -1
        if on_y: points[:, 1] *= -1
        if on_z: points[:, 2] *= -1
        return points


"""
Multiplica por una constante aleatoria distinta cada eje
max_ratio: la constante esta en el intervalo [1/max_ratio, max_ratio]
keep_norm: mantiene la "distancia máxima" antes y despues de la transformación
uniform: todos los ejes se multiplican por la misma constante
"""
class Scale(Transformation):
    def __init__(self, max_ratio=10, keep_norm=True, uniform=False):
        self.max_ratio=max_ratio
        self.keep_norm=keep_norm
        self.uniform=uniform
        self.rng = random.Random(SEED)

    def transform(self, points):
        # guardar norma
        if self.keep_norm:
            size = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))

        # variables
        if self.uniform:
            scale = self.rng.uniform(1/self.max_ratio, self.max_ratio)
        else:
            scale_x = self.rng.uniform(1/self.max_ratio, self.max_ratio)
            scale_y = self.rng.uniform(1/self.max_ratio, self.max_ratio)
            scale_z = self.rng.uniform(1/self.max_ratio, self.max_ratio)

        # escalamiento
        if self.uniform:
            points *= scale
        else:
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points[:, 2] *= scale_z

        # mantener norma
        if self.keep_norm:
            new_size = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
            points *= (size/new_size)

        return points
