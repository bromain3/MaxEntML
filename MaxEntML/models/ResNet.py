import torch
import torch.nn as nn

from ._NNregressorBase import _NNRegressorBase


class _ResBlock(nn.Module):
    def __init__(self, featuresDim, hiddenDim, layerNorm, activationOnOutput=False):
        super().__init__()
        self.norm = nn.LayerNorm(featuresDim) if layerNorm else nn.Identity()
        self.linear1 = nn.Linear(featuresDim, hiddenDim)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(hiddenDim, featuresDim)
        self.actOut = nn.SiLU() if activationOnOutput else nn.Identity()

    def forward(self, x):
        residual = self.norm(x)
        residual = self.act1(self.linear1(residual))
        residual = self.linear2(residual)
        return self.actOut(x + residual)


class _ResNetCore(nn.Module):
    """
    backbone (Linear + SiLU) -> nbBlocks blocs résiduels -> couche de sortie.

    - useBackbone=False : pas de backbone, les blocs travaillent directement en dimension inputDim.
    - La couche de sortie est un Linear(featuresDim -> outputDim). Elle n'est omise (Identity)
      que si useBackbone=False ET outputDim == inputDim, ce qui reproduit ton ancien comportement.
    """
    def __init__(self, inputDim, outputDim, backboneDim, nbBlocks, hiddenDim,
                 layerNorm, useBackbone, activationOnOutput):
        super().__init__()
        if useBackbone:
            featuresDim = backboneDim
            self.backbone = nn.Linear(inputDim, backboneDim)
            self.backboneAct = nn.SiLU()
        else:
            featuresDim = inputDim
            self.backbone = nn.Identity()
            self.backboneAct = nn.Identity()

        self.resnet = nn.Sequential(*[
            _ResBlock(featuresDim, hiddenDim, layerNorm=layerNorm, activationOnOutput=activationOnOutput)
            for _ in range(nbBlocks)
        ])

        if useBackbone or outputDim != inputDim:
            self.outputLayer = nn.Linear(featuresDim, outputDim)
        else:
            self.outputLayer = nn.Identity()

    def forward(self, x):
        return self.outputLayer(self.resnet(self.backboneAct(self.backbone(x))))


class ResNetRegressorModel(_NNRegressorBase):
    """
    ResNet (MLP résiduel) pour la régression : architecture uniquement
    (predict / save / load dans _NNRegressorBase, entraînement avec `train`).

    Paramètres :
    - inputDim, outputDim : dimensions d'entrée / sortie
    - backboneDim  : largeur du flux résiduel (si useBackbone=True)
    - nbBlocks     : nombre de blocs résiduels
    - hiddenDim    : largeur interne de chaque bloc
    - layerNorm    : LayerNorm en entrée de chaque bloc
    - useBackbone  : Linear d'entrée (inputDim -> backboneDim) + couche de sortie
    - activationOnOutput : SiLU après chaque addition résiduelle
    - kaimingInit  : initialisation Kaiming (uniform, nonlinearity='relu') des Linear, biais à 0
    - dtype, device, seed : comme pour MLPNNRegressorModel
    """
    def __init__(self, inputDim, outputDim, backboneDim=64, nbBlocks=3, hiddenDim=128,
                 layerNorm=True, useBackbone=True, activationOnOutput=False, kaimingInit=False,
                 dtype=torch.float32, device="cpu", seed=None):
        self.backboneDim = backboneDim
        self.nbBlocks = nbBlocks
        self.hiddenDim = hiddenDim
        self.layerNorm = layerNorm
        self.useBackbone = useBackbone
        self.activationOnOutput = activationOnOutput
        self.kaimingInit = kaimingInit
        super().__init__(inputDim, outputDim, dtype=dtype, device=device, seed=seed)

    def _buildCore(self):
        core = _ResNetCore(
            self.inputDim, self.outputDim, self.backboneDim, self.nbBlocks, self.hiddenDim,
            layerNorm=self.layerNorm,
            useBackbone=self.useBackbone,
            activationOnOutput=self.activationOnOutput,
        )
        if self.kaimingInit:
            for m in core.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        return core

    def _hyperparams(self):
        # kaimingInit n'est pas sauvegardé : à la lecture, les poids viennent du stateDict
        return {
            "backboneDim": self.backboneDim,
            "nbBlocks": self.nbBlocks,
            "hiddenDim": self.hiddenDim,
            "layerNorm": self.layerNorm,
            "useBackbone": self.useBackbone,
            "activationOnOutput": self.activationOnOutput,
        }

    def enableFineTuning(self, freezeBackbone=False, freezeResnet=False, freezeOutput=False):
        parts = (
            (freezeBackbone, self.core.backbone),
            (freezeResnet, self.core.resnet),
            (freezeOutput, self.core.outputLayer),
        )
        for freeze, part in parts:
            if freeze:
                for p in part.parameters():
                    p.requires_grad = False

    def disableFineTuning(self):
        for p in self.core.parameters():
            p.requires_grad = True