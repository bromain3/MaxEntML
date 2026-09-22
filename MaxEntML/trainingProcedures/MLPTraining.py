import torch
import torch.nn as nn
from tqdm import tqdm


def train(MLPModel: "MLPRegressorModel", scaledInputData: "serie", scaledOutputData: "serie",
          lr=5e-3, weightDecay=0.0, epochs=500, batchSize=128,
          warmupSteps=2000, decayRate=0.92, decaySteps=1500,
          seed=None, progress=True):
    m = MLPModel
    if seed is not None:
        torch.manual_seed(seed)

    # Tout le dataset est mis une fois pour toutes sur le device
    X = scaledInputData.data.detach().to(device=m.device, dtype=m.dtype)
    Y = scaledOutputData.data.detach().to(device=m.device, dtype=m.dtype)
    m.yWasFlat = (Y.ndim == 1)
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    nSamples = X.shape[0]

    if X.shape[1] != m.inputDim or Y.shape[1] != m.outputDim:
        raise ValueError(
            f"Dimensions incompatibles : données ({X.shape[1]} -> {Y.shape[1]}), "
            f"modèle ({m.inputDim} -> {m.outputDim})"
        )

    m.inputColName = getattr(scaledInputData, "colName", None)
    m.outputColName = getattr(scaledOutputData, "colName", None)

    # Optimiseur + lr variable : warmup linéaire puis décroissance exponentielle
    optimizer = torch.optim.Adam(m.core.parameters(), lr=lr, weight_decay=weightDecay)

    def lrFactor(step):
        if step < warmupSteps:
            return (step + 1) / warmupSteps
        return decayRate ** ((step - warmupSteps) / decaySteps)

    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrFactor)

    # Boucle d'entraînement
    m.core.train()
    m.lossHistory = []
    bar = tqdm(range(epochs), desc="Training", disable=not progress)
    for epoch in bar:
        perm = torch.randperm(nSamples, device=m.device)
        totalLoss = torch.zeros((), device=m.device, dtype=m.dtype)

        for start in range(0, nSamples, batchSize):
            idx = perm[start:start + batchSize]
            bX, bY = X[idx], Y[idx]

            optimizer.zero_grad()
            loss = nn.functional.mse_loss(m.core(bX), bY)
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.detach() * bX.shape[0]

        epochLoss = totalLoss.item() / nSamples
        m.lossHistory.append(epochLoss)
        bar.set_postfix(loss=f"{epochLoss:.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    m.core.eval()
    return m.lossHistory