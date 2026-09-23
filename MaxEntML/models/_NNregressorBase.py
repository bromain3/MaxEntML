import inspect

import torch
import joblib

from MaxEntML.dataRepresentation.serie import serie


class _NNRegressorBase:
    """
    Base commune des modèles de régression torch : inférence + sauvegarde / chargement.

    Une sous-classe doit :
    - renseigner ses propres hyperparamètres d'architecture AVANT d'appeler super().__init__()
    - implémenter _buildCore()      -> renvoie le nn.Module (entrée inputDim, sortie outputDim)
    - implémenter _hyperparams()    -> dict des hyperparamètres d'architecture (== arguments du constructeur)

    predict() prend des données physiques : la normalisation / dénormalisation passe par
    inputScaler / outputScaler, à renseigner avant predict().
    L'entraînement se fait avec la fonction `train`.
    """
    def __init__(self, inputDim, outputDim, dtype=torch.float32, device="cpu", seed=None):
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.dtype = dtype
        self.device = torch.device(device)
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
        self.core = self._buildCore()
        self.core.to(device=self.device, dtype=self.dtype)

        self.inputScaler = None
        self.outputScaler = None
        self.inputColName = None
        self.outputColName = None
        self.yWasFlat = False
        self.lossHistory = []

    def _buildCore(self):
        raise NotImplementedError

    def _hyperparams(self):
        raise NotImplementedError

    def predict(self, XPhys):
        XScaled = self.inputScaler.transform(XPhys).data.detach()
        XScaled = XScaled.to(device=self.device, dtype=self.dtype)
        XScaled = XScaled.reshape(XScaled.shape[0], -1)

        self.core.eval()
        with torch.no_grad():
            pred = self.core(XScaled)
        pred = pred.cpu().numpy()
        if self.yWasFlat:
            pred = pred.ravel()

        pred = serie(pred, colName=self.outputColName, storingDtype=XPhys.storingDtype, storingDevice=XPhys.storingDevice)
        return self.outputScaler.inverse_transform(pred)

    def save(self, filename):
        joblib.dump({
            "version": 2,
            "hyperparams": {
                "inputDim": self.inputDim,
                "outputDim": self.outputDim,
                "dtype": self.dtype,
                "seed": self.seed,
                **self._hyperparams(),
            },
            "yWasFlat": self.yWasFlat,
            "stateDict": {k: v.cpu() for k, v in self.core.state_dict().items()},
            "lossHistory": self.lossHistory,
            "inputScaler": self.inputScaler,
            "outputScaler": self.outputScaler,
            "inputColName": self.inputColName,
            "outputColName": self.outputColName,
        }, filename)

    @classmethod
    def load(cls, filename, device="cpu"):
        payload = joblib.load(filename)
        hp = dict(payload["hyperparams"])

        # compatibilité avec les sauvegardes v1 (inputDim / outputDim hors de hyperparams)
        hp.setdefault("inputDim", payload.get("inputDim"))
        hp.setdefault("outputDim", payload.get("outputDim"))
        # on ne touche pas à la seed globale au chargement
        hp.pop("seed", None)
        # on ignore les clés que le constructeur n'accepte pas (ex. lr, epochs des anciennes versions)
        accepted = inspect.signature(cls.__init__).parameters
        hp = {k: v for k, v in hp.items() if k in accepted}

        obj = cls(device=device, **hp)
        obj.core.load_state_dict(payload["stateDict"])
        obj.core.to(device=obj.device, dtype=obj.dtype)
        obj.core.eval()
        obj.yWasFlat = payload["yWasFlat"]
        obj.lossHistory = payload["lossHistory"]
        obj.inputScaler = payload["inputScaler"]
        obj.outputScaler = payload["outputScaler"]
        obj.inputColName = payload["inputColName"]
        obj.outputColName = payload["outputColName"]
        return obj