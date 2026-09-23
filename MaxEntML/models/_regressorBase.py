import joblib

from MaxEntML.dataRepresentation.serie import serie


class BaseRegressorModel:
    """
    Classe de base factorisant la logique commune à tous les wrappers de
    modèles de régression scikit-learn (fit / predict / save / load).

    Chaque sous-classe doit, dans son __init__, construire self.core avec
    l'estimateur scikit-learn voulu (déjà paramétré, non entraîné).
    Elle peut éventuellement surcharger _corePayload()/_loadExtra() si
    elle a des attributs supplémentaires à persister (cf. kNNRegressorModel
    qui garde n_neighbors à part, à titre indicatif).
    """

    def __init__(self):
        self.core = None
        self.inputScaler = None
        self.outputScaler = None

        self.inputColName = None
        self.outputColName = None

    def fit(self, XScaled, YScaled):
        self.core.fit(XScaled.data.cpu().detach().numpy(), YScaled.data.cpu().detach().numpy())
        self.inputColName = XScaled.colName
        self.outputColName = YScaled.colName

    def predict(self, XPhys):
        XScaled = self.inputScaler.transform(XPhys).data.cpu().detach().numpy()
        pred = self.core.predict(XScaled)
        pred = serie(pred, colName=self.outputColName, storingDtype=XPhys.storingDtype, storingDevice=XPhys.storingDevice)
        return (self.outputScaler.inverse_transform(pred))

    def _corePayload(self):
        """A surcharger si des champs specifiques a la sous-classe doivent etre sauvegardes."""
        return {}

    def _loadExtra(self, payload):
        """A surcharger pour relire les champs ajoutes par _corePayload()."""
        pass

    def save(self, filename):
        payload = {
            "version": 1,
            "core": self.core,
            "inputScaler": self.inputScaler,
            "outputScaler": self.outputScaler,
            "inputColName": self.inputColName,
            "outputColName": self.outputColName,
        }
        payload.update(self._corePayload())
        joblib.dump(payload, filename)

    @classmethod
    def load(cls, filename):
        payload = joblib.load(filename)
        obj = cls.__new__(cls)
        BaseRegressorModel.__init__(obj)
        obj.core = payload["core"]
        obj.inputScaler = payload["inputScaler"]
        obj.outputScaler = payload["outputScaler"]
        obj.inputColName = payload["inputColName"]
        obj.outputColName = payload["outputColName"]
        obj._loadExtra(payload)
        return obj