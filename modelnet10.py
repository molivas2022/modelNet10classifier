import os
from enum import Enum, auto
from random import Random
from typing import List
import torch
from torch.utils.data import Dataset
from utils.transformation import Transformation, Normalization
import open3d as o3d
import numpy as np


VALIDATION_SEED = 0x5EED    # Un número arbitario que determina la partición del set de validación
VALIDATION_RATIO = 0.2

TRANSFORMATION_SEED = 0x5EED # Un número arbitrario que determina qué transformaciones aplicar.

class ModelNetClass(Enum):
    def __init__(self, label, train_size, test_size):
        self._label = label
        self._train_size = train_size   # train + validation
        self._test_size = test_size


    
    BATHTUB = ('bathtub', 106, 50)
    BED = ('bed', 515, 100)
    CHAIR = ('chair', 889, 100)
    DESK = ('desk', 200, 86)
    DRESSER = ('dresser', 200, 86)
    MONITOR = ('monitor', 465, 100)
    NIGHT_STAND = ('night_stand', 200, 86)
    SOFA = ('sofa', 680, 100)
    TABLE = ('table', 392, 100)
    TOILET = ('toilet', 344, 100)

    # no sé para qué hago esto jaja
    @property
    def label(self):
        return self._label
    @property
    def train_size(self):
        return self._train_size
    @property
    def effective_train_size(self):
        return self.train_size - self.validation_size
    @property
    def validation_size(self):
        train_size = self.train_size
        return int(train_size * VALIDATION_RATIO)
    @property
    def test_size(self):
        return self._test_size

    def train_path(self, root_dir=None):
        if not root_dir:
            root_dir = os.getcwd()
        dataset_dir = os.path.join(root_dir, "ModelNet10", "pcd")
        return os.path.join(dataset_dir, self.label, "train")

    def test_path(self, root_dir=None):
        if not root_dir:
            root_dir = os.getcwd()
        dataset_dir = os.path.join(root_dir, "ModelNet10", "pcd")
        return os.path.join(dataset_dir, self.label, "test")

    def train_files(self, root_dir=None):
        path = self.train_path(root_dir)
        files = list()
        for file in os.scandir(path):  
            if ".pcd" in str(file):
                files.append(file)
        return sorted(files, key=lambda f: f.name)

    def effective_train_files(self, root_dir=None):
        return [file for file in self.train_files(root_dir) if file not in self.validation_files(root_dir)]

    def validation_files(self, root_dir=None):
        train_files = self.train_files(root_dir)
        rng = Random(VALIDATION_SEED)
        return rng.sample(train_files, self.validation_size)

    def test_files(self, root_dir=None):
        path = self.test_path(root_dir)
        files = list()
        for file in os.scandir(path):  
            if ".pcd" in str(file):
                files.append(file)
        return sorted(files, key=lambda f: f.name)

class DatasetType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

class ModelNet(Dataset):
    def __init__(self,
                 classes: list[ModelNetClass],
                 type: DatasetType,
                 repetitions: int =0,
                 transformations: List[Transformation] =[],
                 normalize: bool=True,
                 preserve_original: bool=True,
                 root_dir=None
                 ):

        self.root_dir = root_dir    
        X = list()
        y = list()
        rng = Random(TRANSFORMATION_SEED)
        for i in range(len(classes)):
            match type:
                case DatasetType.TRAIN:
                    files = classes[i].effective_train_files(root_dir)
                case DatasetType.VALIDATION:
                    files = classes[i].validation_files(root_dir)
                case DatasetType.TEST:
                    files = classes[i].test_files(root_dir)

            for file in files:
                pcd = o3d.io.read_point_cloud(file.path)
                points = np.asarray(pcd.points, dtype=float)
                for _ in range(repetitions):
                    _points = np.copy(points)
                    N = len(transformations)
                    M = rng.randint(0, N)
                    indices = sorted(rng.sample(range(N), M))
                    for idx in indices:
                        _points = transformations[idx].transform(_points)
                    if normalize:
                        _points = Normalization.transform(_points)
                    X.append(_points)
                    y.append(i)
                if preserve_original:
                    if normalize:
                        points = Normalization.transform(points)
                    X.append(points)
                    y.append(i)
        
        X = np.transpose(X, (0, 2, 1))
        
        self._X = torch.tensor(X, dtype=torch.float32)
        self._y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self._X)
    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]
    
    @property
    def length(self):
        return len(self._X)
