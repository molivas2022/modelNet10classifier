import os
from enum import Enum
import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np

ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, "ModelNet10", "pcd")

class ModelNetClass(Enum):
    def __init__(self, label, train_size, test_size):
        self._label = label
        self._train_size = train_size
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
    def test_size(self):
        return self._test_size
    @property
    def train_path(self):
        return os.path.join(DATASET_DIR, self.label, "train")
    @property
    def test_path(self):
        return os.path.join(DATASET_DIR, self.label, "test")
    @property
    def train_files(self):
        path = self.train_path
        files = list()
        for file in os.scandir(path):  
            if ".pcd" in str(file):
                files.append(file)
        return files
    @property
    def test_files(self):
        path = self.test_path
        files = list()
        for file in os.scandir(path):  
            if ".pcd" in str(file):
                files.append(file)
        return files


class ModelNet(Dataset):
    def __init__(self, classes: list[ModelNetClass], train: bool, test: bool):
        if not (train or test):
            raise Exception("Error creating instance of ModelNet dataset:" +
                            "'train' and 'test' cannot both be false at the same time")
        
        X = list()
        y = list()
        for i in range(len(classes)):
            if train:
                for file in classes[i].train_files:
                    pcd = o3d.io.read_point_cloud(file.path)
                    points = np.asarray(pcd.points, dtype=float)
                    X.append(points)
                    y.append(i)
            if test:
                for file in classes[i].test_files:
                    pcd = o3d.io.read_point_cloud(file.path)
                    points = np.asarray(pcd.points, dtype=float)
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