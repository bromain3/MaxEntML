import torch
import torch.nn as nn

from ._NNregressorBase import _NNRegressorBase


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


def _buildMLP(inputDim, outputDim, hiddenLayers, activation):
    layers = []
    prev = inputDim
    for h in hiddenLayers:
        layers += [nn.Linear(prev, h), _ACTIVATIONS[activation]()]
        prev = h
    layers.append(nn.Linear(prev, outputDim))
    return nn.Sequential(*layers)


class MLPRegressorModel(_NNRegressorBase):
    """
    MLP torch pour la régression : architecture uniquement (predict / save / load dans _NNRegressorBase).

    Paramètres :
    - inputDim, outputDim : dimensions d'entrée / sortie du réseau
    - hiddenLayers : tailles des couches cachées, ex. (64, 64, 64)
    - activation   : "relu", "tanh", "gelu", "silu" ou "softplus"
    - dtype        : float32 par défaut ; float64 pour une précision maximale
    - device       : "cpu" ou "cuda"
    - seed         : graine pour l'initialisation des poids
    """
    def __init__(self, inputDim, outputDim, hiddenLayers=(64, 64), activation="tanh",
                 dtype=torch.float32, device="cpu", seed=None):
        self.hiddenLayers = tuple(hiddenLayers)
        self.activation = activation
        super().__init__(inputDim, outputDim, dtype=dtype, device=device, seed=seed)

    def _buildCore(self):
        return _buildMLP(self.inputDim, self.outputDim, self.hiddenLayers, self.activation)

    def _hyperparams(self):
        return {
            "hiddenLayers": self.hiddenLayers,
            "activation": self.activation,
        }