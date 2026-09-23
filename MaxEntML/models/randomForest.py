from sklearn.ensemble import RandomForestRegressor

from MaxEntML.models._regressorBase import BaseRegressorModel

class randomForestRegressorModel(BaseRegressorModel):
    """
    Foret d'arbres de decision (bagging). Plus robuste au bruit et a la variance
    qu'un arbre seul, mais moins "exact" sur des points deja vus qu'un kNN pondere
    par la distance : chaque arbre est entraine sur un bootstrap des donnees.
    n_jobs=-1 pour paralleliser l'entrainement sur tous les coeurs disponibles.
    """

    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=1,
                 n_jobs=-1, random_state=0):
        super().__init__()
        self.core = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )