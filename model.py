import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(norm_type, num_channels, dim='1d', num_groups=8):
    if norm_type == 'batchnorm':
        return nn.BatchNorm1d(num_channels) if dim == '1d' else nn.BatchNorm2d(num_channels)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(num_channels)
    elif norm_type is None or norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Undefined norm type")

## KAN, créditos a Ali Kashefi (https://github.com/Ali-Stanford/PointNet_KAN_Graphic)
"""

Args:
    input_dim: 
    num_points: output_dim
    degree: 
    a: alpha en el polinomio de Jacaboi
    b: beta en el polinomio de Jacaboi
"""
class KAN(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree
        
        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        # Inicializamos los coeficientes
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))

        x = torch.tanh(x)

        jacobi = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, 1]= ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        
        # No preguntes.
        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, i] = (A*x + B)*jacobi[:, :, i-1].clone() + C*jacobi[:, :, i-2].clone()

        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)
        y = y.view(-1, self.output_dim)
        return y

class KANShared(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(KANShared, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous() 
        x = torch.tanh(x) 

        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        jacobi = jacobi.permute(0, 2, 3, 1)  
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs) 
        return y




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
    def __init__(self, dim, num_points, norm_type='batchnorm'):
        super(Tnet, self).__init__()

        self.dim = dim

        # Función de activación
        self.act = F.relu

        # Conv1d es una implementación sencilla de una 'MLP compartida'
        self.shared_mlp1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.shared_mlp2 = nn.Conv1d(64, 128, kernel_size=1)
        self.shared_mlp3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = get_norm_layer(norm_type, 64)
        self.bn2 = get_norm_layer(norm_type, 128)
        self.bn3 = get_norm_layer(norm_type, 1024)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

        # MLPs no compartidas
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)
        self.bn4 = get_norm_layer(norm_type, 512)
        self.bn5 = get_norm_layer(norm_type, 256)
    
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
        #iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        iden = torch.eye(self.dim).repeat(bs, 1, 1).to(x.device)
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
class PointNetClassifier(nn.Module):
    def __init__(self, dim, num_points, num_global_feats, num_classes, ignore_Tnet=False, norm_type='batchnorm', dropout=0.3):
        super(PointNetClassifier, self).__init__()
        self.ignore_Tnet = ignore_Tnet

        # Función de activación
        self.act = F.relu

        # T-Net en los puntos de la entrada
        self.input_transform = Tnet(dim, num_points)

        # Primera MLP compartida, transforma los puntos de la entrada en features
        self.shared_mlp1 = nn.Conv1d(3, 64, kernel_size=1)
        self.shared_mlp2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn1 = get_norm_layer(norm_type, 64)
        self.bn2 = get_norm_layer(norm_type, 64)

        # T-Net en las features
        self.feature_transform = Tnet(64, num_points)

        # Segunda MLP compartida, determina las features globales
        self.shared_mlp3 = nn.Conv1d(64, 64, kernel_size=1)
        self.shared_mlp4 = nn.Conv1d(64, 128, kernel_size=1)
        self.shared_mlp5 = nn.Conv1d(128, num_global_feats, kernel_size=1)
        self.bn3 = get_norm_layer(norm_type, 64)
        self.bn4 = get_norm_layer(norm_type, 128)
        self.bn5 = get_norm_layer(norm_type, num_global_feats)
        # Max pool para extraer las features globales
        # Devolver los indices nos permite ver los indices críticos que determinan las features globales
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

        # MLP para clasificación
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.bn_linear1 = get_norm_layer(norm_type, 512)
        self.bn_linear2 = get_norm_layer(norm_type, 256)
        self.dropout = nn.Dropout(p=dropout)
        #self.dropout = nn.Dropout(p=0.2)

        # Output layer
        self.linear3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Tamaño del batch, es decir cuantos ejemplos hay en el batch
        bs = x.shape[0]

        if not self.ignore_Tnet:
            # Transformación del input
            input_matrix = self.input_transform(x)
            # x = torch.bmm(x.transpose(2, 1), input_matrix).tranpose(2, 1)
            x = torch.transpose(torch.bmm(torch.transpose(x, 2, 1), input_matrix), 2, 1)

        # Paso a través de las primeras MLPs compartidas
        x = self.bn1(self.act(self.shared_mlp1(x)))
        x = self.bn2(self.act(self.shared_mlp2(x)))


        if not self.ignore_Tnet:
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
        if not self.ignore_Tnet:
            return x, critical_indexes, feature_matrix
        else:
            return x, critical_indexes, None




## PointNetKAN
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, num_points, num_classes, scaling=1.0, ignore_Tnet=False):
        super(PointNetKAN, self).__init__()

        self.ignore_Tnet = ignore_Tnet

        # T-Net en los puntos de la entrada
        if not ignore_Tnet:
            self.input_transform = Tnet(input_channels, num_points)

        self.jacobikan5 = KANShared(input_channels, int(num_points * scaling), 4)

        self.jacobikan6 = KAN(int(num_points * scaling), num_classes, 4)

        self.bn5 = nn.GroupNorm(num_channels=int(num_points * scaling), num_groups=8)

    def forward(self, x):
        if not self.ignore_Tnet:
            input_matrix = self.input_transform(x)
            #x = transpose(torch.bmm(torch.transpose(x, 2, 1), input_matrix), 2, 1)
            x = torch.transpose(torch.bmm(torch.transpose(x, 2, 1), input_matrix), 2, 1)

        x = self.jacobikan5(x)

        x = self.bn5(x)

        global_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)

        x = self.jacobikan6(global_feature)

        return x, None, None


## Wrapper para Test-time augmentation
"""
Wrappea un clasificador f, un merge-mode m, una lista de transformaciones \{T_1, \ldots, T_n\}, y un input x, genera
una secuencia de inputs \{T_1(x), \ldots, T_n(x)\}, tal que y = m(T_1(x), \ldots, T_n(x)).
Args:
    classifier          El clasificador al que wrappea. 
    transformations     Las transformaciones a aplicar al input (iterable).
    merge_mode          Método que utiliza para combinar predicciones en una.
"""
class TTAClassifier(nn.Module):
    def __init__(self,  classifier, transformations: list, merge_mode):
        super(TTAClassifier, self).__init__()
        self.classifier = classifier
        self.transformations = transformations
        self.merge_mode = merge_mode

    # Importante, este aplica softmax, no tiene mucho sentido aplicar el promedio sobre los logits.
    def forward(self, x):

        def apply_along_axis(function, x, axis):
            return torch.stack([
                function(x_i) for x_i in torch.unbind(x, dim=axis)
            ], dim=axis)

        
        Tx = [x] 
        x = torch.transpose(x, 2, 1)
        # Aplicamos las transformaciones.
        for T in self.transformations:
            transformed_batch = apply_along_axis(
                    lambda y: torch.from_numpy(T.transform(y.cpu().numpy())).to(dtype=torch.float32, device=y.device),
                    x,
                    0
            ) # (B, N, C)
            transformed_batch = torch.transpose(transformed_batch, 2, 1)
            Tx.append(transformed_batch)

        probs_list = list()
        for Tx_i in Tx:
            logits, _, _ = self.classifier(Tx_i) # (B, N_CLASSES)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

        probs_stacked = torch.stack(probs_list, dim=0) # (N_T, B, N_CLASSES)

        def geometric_mean(tensor, dim=0, eps=1e-9):
            tensor = tensor.clamp(min=eps)
            log_tensor = torch.log(tensor)
            mean_log = log_tensor.mean(dim=dim)
            return torch.exp(mean_log)
        
        if self.merge_mode == "mean":
            out = probs_stacked.mean(dim=0) # (B, N_CLASSES)
        elif self.merge_mode == "gmean":
            out = geometric_mean(probs_stacked, dim=0)
        elif self.merge_mode == "max":
            out = probs_stacked.max(dim=0).values


        return out, None, None




    def eval(self):
        self.classifier.eval()
        return super().eval()



        



# Clase para el cómputo de la pérdida
"""
Args:
    alpha            El peso de las clases para pérdida CrossEntropy.
    gamma            Peso para la pérdida focal.
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

        #self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")

    def forward(self, predictions, targets, A, is_train=True):
        # tamaño de batch
        batch_size = predictions.size(0)

        # computamos pérdida CE
        ce_loss = self.cross_entropy_loss(predictions, targets)
        
        # Probabilidades predichas
        pn = F.softmax(predictions, dim=1)

        # Probabilidad de clase target (ground truth)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # Factor de regularización matriz transformación
        reg = torch.tensor(0.0)
        #if self.reg_weight > 0 and is_train:
        if self.reg_weight > 0:
            I = torch.eye(A.size(1)).unsqueeze(0).repeat(A.size(0), 1, 1)
            if A.is_cuda:
                I = I.cuda()

            # Se computa una matriz I - A para cada elemento del batch
            diff = I - torch.bmm(A, A.transpose(2, 1)) # shape (B, 64, 64)
            norms = torch.linalg.norm(diff, dim=(1, 2)) # shape (B,)
            reg = norms.mean()
            reg = self.reg_weight * reg
        
        focal_loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average:
            return focal_loss.mean() + reg, reg
        else:
            return focal_loss.sum() + reg, reg
