from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from MaxEntML.models._regressorBase import BaseRegressorModel


class gradientBoostingRegressorModel(BaseRegressorModel):
    """
    Boosting sequentiel d'arbres peu profonds. Souvent plus precis qu'une foret
    aleatoire sur des jeux de donnees de taille moderee, mais plus sensible au
    surapprentissage : learning_rate et n_estimators sont a ajuster ensemble
    (typiquement, baisser learning_rate et augmenter n_estimators).

    GradientBoostingRegressor ne supporte nativement qu'une seule colonne de
    sortie : on l'enrobe systematiquement dans un MultiOutputRegressor, qui
    entraine un modele independant par colonne de YScaled (transparent si Y
    n'a qu'une seule colonne).
    """

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3,
                 random_state=0, n_jobs=-1):
        super().__init__()
        base = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.core = MultiOutputRegressor(base, n_jobs=n_jobs)