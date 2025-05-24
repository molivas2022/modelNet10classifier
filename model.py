import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## T-net
"""
T-net es una 'mini-red' que aprende una matriz de transformación de tamaño
dimxdim que transforma puntos a una representación 'canónica', la cuál
es invariante a transformaciones rigidas (rotación, translación, reflexión).
Args:
    dim: dimensión de los puntos
    num_points: número de puntos
"""
class Tnet(nn.Module):
    def __init__(self, dim, num_points):
        super(Tnet, self).__init__()

        self.dim = dim

        # Función de activación
        self.act = F.relu

        # Conv1d es una implementación sencilla de una 'MLP compartida'
        self.shared_mlp1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.shared_mlp2 = nn.Conv1d(64, 128, kernel_size=1)
        self.shared_mlp3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

        # MLPs no compartidas
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        bs = x.shape[0]

        # Paso a través de las MLPs compartidas
        x = self.bn1(self.act(self.shared_mlp1(x)))
        x = self.bn2(self.act(self.shared_mlp2(x)))
        x = self.bn3(self.act(self.shared_mlp3(x)))

        # Max pool
        x = self.max_pool(x).view(bs, -1)
        
        # Paso a través de las MLPs no compartidas
        x = self.bn4(self.act(self.linear1(x)))
        x = self.bn5(self.act(self.linear2(x)))
        x = self.linear3(x)
        
        # Reshape de 'T-Net(x)' a una matriz
        x = x.view(-1, self.dim, self.dim)
        # Le sumamos la matriz identidad para mayor estabilidad
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x += iden

        return x


## Point-net classifier
"""
Arquitectura de PointNet para clasificación
Args:
    dim: dimensión de los puntos de la entrada
    num_points: número de puntos en la entrada
    num_global_feats: número de features globales que determinará la red
    num_classes: número de categorias de clasificación
"""
class PointnetClassifier(nn.Module):
    def __init__(self, dim, num_points, num_global_feats, num_classes):
        super(PointnetClassifier, self).__init__()

        # Función de activación
        self.act = F.relu

        # T-Net en los puntos de la entrada
        self.input_transform = Tnet(dim, num_points)

        # Primera MLP compartida, transforma los puntos de la entrada en features
        self.shared_mlp1 = nn.Conv1d(3, 64, kernel_size=1)
        self.shared_mlp2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # T-Net en las features
        self.feature_transform = Tnet(64, num_points)

        # Segunda MLP compartida, determina las features globales
        self.shared_mlp3 = nn.Conv1d(64, 64, kernel_size=1)
        self.shared_mlp4 = nn.Conv1d(64, 128, kernel_size=1)
        self.shared_mlp5 = nn.Conv1d(128, num_global_feats, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(num_global_feats)
        # Max pool para extraer las features globales
        # Devolver los indices nos permite ver los indices críticos que determinan las features globales
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

        # MLP para clasificación
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.bn_linear1 = nn.BatchNorm1d(512)
        self.bn_linear2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

        # Output layer
        self.linear3 = nn.Linear(256, num_classes)
    
    def forward(self, x):

        # Tamaño del batch, es decir cuantos ejemplos hay en el batch
        bs = x.shape[0]

        # Transformación del input
        input_matrix = self.input_transform(x)
        # x = torch.bmm(x.tranpose(2, 1), input_matrix).tranpose(2, 1)
        x = torch.transpose(torch.bmm(torch.transpose(x, 2, 1), input_matrix), 2, 1)

        # Paso a través de las primeras MLPs compartidas
        x = self.bn1(self.act(self.shared_mlp1(x)))
        x = self.bn2(self.act(self.shared_mlp2(x)))

        # Transformación de features
        feature_matrix = self.feature_transform(x)
        # x = torch.bmm(x.tranpose(2, 1), feature_matrix).tranpose(2, 1)
        x = torch.transpose(torch.bmm(torch.transpose(x, 2, 1), feature_matrix), 2, 1)

        # Paso a través de las segundas MLPs compartidas
        x = self.bn3(self.act(self.shared_mlp3(x)))
        x = self.bn4(self.act(self.shared_mlp4(x)))
        x = self.bn5(self.act(self.shared_mlp5(x)))

        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        # Clasificación
        x = self.bn_linear1(self.act(self.linear1(global_features)))
        x = self.bn_linear2(self.act(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        # Devolver logits
        return x, critical_indexes, feature_matrix


# Clase para el cómputo de la pérdida
"""
Args:
    alpha            El peso de las clases para pérdida CrossEntropy.
    reg_weight       Peso de regularización.
    size_average     Booleano que define si es que la pérdida final se computa como promedio o no.
"""
class PointNetLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=0, reg_weight=0, size_average=True):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average

        # Convertimos en tensores
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.Tensor(alpha)

        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)

    def forward(self, predictions, targets, A):
        # tamaño de batch
        batch_size = predictions.size(0)

        # computamos pérdida CE
        ce_loss = self.cross_entropy_loss(predictions, targets)
        
        # Probabilidades predichas
        pn = F.softmax(predictions, dim=1)
        pn.gather(1, targets.view(-1, 1)).view(-1)

        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1)
            if A.is_cuda:
                I = I.cuda()
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))
            reg = self.reg_weight * reg / batch_size

        focal_loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average:
            return focal_loss.mean() + reg
        else:
            return focal_loss.sum() + reg