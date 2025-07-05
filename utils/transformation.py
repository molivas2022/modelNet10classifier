import numpy as np
import random
from typing import List
from abc import ABC, abstractmethod

SEED = 0x5EED

"""
Interfaz de una transformación
"""
class Transformation(ABC):
    @abstractmethod
    def transform(self, points):
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
max_units: unidades a trasladar (positivo o negativo) por cada eje
           1 unidad = "distancia máxima" antes de la transformación
keep_norm: mantiene la "distancia máxima" antes y despues de la transformación
"""
class Translation(Transformation):
    def __init__(self, max_units=10, keep_norm=False):
        self.max_units=max_units
        self.keep_norm=keep_norm
        self.rng = random.Random(SEED)

    def transform(self, points):
        # guardar norma
        size = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))

        # variables
        delta_x = self.rng.uniform(-self.max_units, self.max_units) * size
        delta_y = self.rng.uniform(-self.max_units, self.max_units) * size
        delta_z = self.rng.uniform(-self.max_units, self.max_units) * size

        # translación
        points[:, 0] += delta_x
        points[:, 1] += delta_y
        points[:, 2] += delta_z

        # mantener norma
        if self.keep_norm:
            new_size = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
            points *= (size/new_size)

        return points

class Reflection(Transformation):
    def __init__(self):
        self.rng = random.Random(SEED)

    def transform(self, points):
        # variables
        on_x = random.choice([False, True])
        on_y = random.choice([False, True])
        on_z = random.choice([False, True])

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

"""
Interfaz de una transformación que bota puntos de la point cloud
    - loss_ratio : que porcentaje de puntos se pierden
    - padding: de que forma se rellenan los valores perdidos
        - "duplicate": duplicando puntos cualquiera
        - "zero": rellenando con ceros
        - "noise": rellena con valores aleatorios (dentro de la bounding box)
Las implementaciones de esta interfaz especifican como se decide que puntos se botan
"""
class Drop(Transformation):
    def __init__(self, loss_ratio=0.2, padding="duplicate"):
        self.loss_ratio=loss_ratio
        self.padding=padding
        self.rng = random.Random(SEED)

    def transform(self, points):
        survivors = self.drop(points)
        loss_size = points.shape[0] - len(survivors)
        if self.padding == "duplicate":
            out = survivors
            for _ in range(loss_size):
                duplicate = self.rng.choice(survivors)
                out.append(duplicate)
        elif self.padding == "zero":
            out = survivors
            for _ in range(loss_size):
                out.append([0, 0, 0])
        elif self.padding == "noise":
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            out = survivors
            for _ in range(loss_size):
                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                z = self.rng.uniform(z_min, z_max)
                out.append([x, y, z])
        else:
            raise Exception("Drop augmentation padding undefined")
        return np.array(out)
    
    @abstractmethod
    def drop(self, points) -> List:
        pass

"""
Bota puntos de la point cloud de forma completamente aleatoria
"""
class DropRandom(Drop):
    def drop(self, points):
        survivor_size = int(np.ceil( (1 - self.loss_ratio) * points.shape[0] ))
        return self.rng.sample(points.tolist(), survivor_size)
    
"""
Bota puntos pertenecientes a una esfera definida aleatoriamente
"""
class DropSphere(Drop):
    def drop(self, points):
        N = points.shape[0]
        drop_size = int(np.ceil(self.loss_ratio * N))

        # centro de la esfera
        center_idx = self.rng.randint(0, N - 1)
        center = points[center_idx]

        # distancias
        distances = np.linalg.norm(points - center, axis=1)

        # ordenar distancias para obtener indices de los 'drop_size' puntos más cercanos
        drop_indices = np.argsort(distances)[:drop_size]

        # mascara para mantener el resto
        mask = np.ones(N, dtype=bool)
        mask[drop_indices] = False
        survivors = points[mask]

        return survivors.tolist()

"""
Desplaza cada punto de la cloud point de forma aleatoria
max_units: unidades a trasladar (positivo o negativo) por cada eje
           1 unidad = "distancia máxima" antes de la transformación
keep_norm: mantiene la "distancia máxima" antes y despues de la transformación
"""
class Jittering(Transformation):
    def __init__(self, max_units=0.002, keep_norm=False):
        self.max_units=max_units
        self.keep_norm=keep_norm
        self.rng = random.Random(SEED)

    def transform(self, points):
        # guardar norma
        size = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))

        # variables
        for i in range(points.shape[0]):
            delta_x = self.rng.uniform(-self.max_units, self.max_units) * size
            delta_y = self.rng.uniform(-self.max_units, self.max_units) * size
            delta_z = self.rng.uniform(-self.max_units, self.max_units) * size

            # translación
            points[i, 0] += delta_x
            points[i, 1] += delta_y
            points[i, 2] += delta_z

        # mantener norma
        if self.keep_norm:
            new_size = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
            points *= (size/new_size)

        return points

"""
Añade ruido: Reemplaza puntos por puntos aleatorios (dentro de la bounding box)
    - noise_ratio : que porcentaje de puntos se reemplazan por ruido
"""
class Noise(DropRandom):
    def __init__(self, noise_ratio=0.1):
        super().__init__(loss_ratio=noise_ratio, padding="noise")