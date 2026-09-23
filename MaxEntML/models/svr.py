from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from MaxEntML.models._regressorBase import BaseRegressorModel


class svrRegressorModel(BaseRegressorModel):
    """
    Regression a vecteurs de support (noyau RBF par defaut). Contrairement aux
    modeles a base d'arbres, SVR est sensible a l'echelle des entrees (raison de
    plus pour bien passer par XScaled/inputScaler).

    SVR ne supporte nativement qu'une seule colonne de sortie : on l'enrobe
    systematiquement dans un MultiOutputRegressor, qui entraine un modele
    independant par colonne de YScaled (transparent si Y n'a qu'une seule
    colonne). C'est aussi ce qui apporte la parallelisation (n_jobs) que SVR
    seul n'a pas.
    """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', n_jobs=-1):
        super().__init__()
        base = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.core = MultiOutputRegressor(base, n_jobs=n_jobs)