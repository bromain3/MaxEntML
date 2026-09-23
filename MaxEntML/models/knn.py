from sklearn.neighbors import KNeighborsRegressor

from MaxEntML.models._regressorBase import BaseRegressorModel


class kNNRegressorModel(BaseRegressorModel):
    """
    le knn est parametre avec:
    weights = distance pour que la regression soit plus impactee par les points proches
    (semble raisonnable vu qu'il n'y a pas de bruit dans nos donnees. Si on a une entree qr
    qui est dans la dtb, on veut le lambda exact, pas une approximation)
    les autres parametres sont par defaut (calcul par defaut des plus proches voisins meme si
    pour seulement deux entrees KDTree serait sans doute mieux, et la distance est euclidienne par defaut)
    """

    def __init__(self, n_neighbors):
        super().__init__()
        self.core = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', leaf_size=100)

    def _corePayload(self):
        return {"n_neighbors": self.core.n_neighbors}