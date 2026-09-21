from sklearn.neighbors import KNeighborsRegressor
import joblib


from MaxEntML.dataRepresentation.serie import serie

class kNNRegressorModel:
    """
    the knn is parametrized with:
    weights = distance so that the regression value is more impacted by closer points (seems resonable as there is no noise in our dataset. If we have an input qr that is in the dtb, we want the exact lambda, not an approximate)
    all other parameters are by default (default nearest neigbours computation even thoug for only two input data KDTree should be better
    and distance is by default euclidian distance
    """
    def __init__(self, n_neighbors):
        self.core = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
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
        return(self.outputScaler.inverse_transform(pred))

    def save(self, filename):
        joblib.dump({
            "version": 1,
            "n_neighbors": self.core.n_neighbors,
            "core": self.core,
            "inputScaler": self.inputScaler,
            "outputScaler": self.outputScaler,
            "inputColName": self.inputColName,
            "outputColName": self.outputColName,
        }, filename)

    @classmethod
    def load(cls, filename):
        payload = joblib.load(filename)
        obj = cls(payload["n_neighbors"])
        obj.core = payload["core"]
        obj.inputScaler = payload["inputScaler"]
        obj.outputScaler = payload["outputScaler"]
        obj.inputColName = payload["inputColName"]
        obj.outputColName = payload["outputColName"]
        return obj