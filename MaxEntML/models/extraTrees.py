from sklearn.ensemble import ExtraTreesRegressor

from MaxEntML.models._regressorBase import BaseRegressorModel


class extraTreesRegressorModel(BaseRegressorModel):
    """
    Variante de la foret aleatoire ou les seuils de split sont eux aussi tires
    aleatoirement (au lieu d'etre optimises), ce qui reduit encore la variance
    au prix d'un leger biais. Souvent plus rapide a entrainer qu'une RandomForest.
    """

    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=1,
                 n_jobs=-1, random_state=0):
        super().__init__()
        self.core = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
