from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from MaxEntML.models._regressorBase import BaseRegressorModel


class adaBoostRegressorModel(BaseRegressorModel):
    """
    Boosting qui reponderee iterativement les exemples les plus mal predits par
    un estimateur "faible" (par defaut un arbre peu profond). estimator_max_depth
    controle la complexite de l'arbre de base utilise a chaque iteration.

    AdaBoostRegressor ne supporte nativement qu'une seule colonne de sortie :
    on l'enrobe systematiquement dans un MultiOutputRegressor, qui entraine un
    modele independant par colonne de YScaled (transparent si Y n'a qu'une
    seule colonne).
    """

    def __init__(self, n_estimators=1000, learning_rate=0.05,
                 estimator_max_depth=20, n_jobs=-1):
        super().__init__()
        weak = DecisionTreeRegressor(max_depth=estimator_max_depth)
        base = AdaBoostRegressor(
            estimator=weak,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        self.core = MultiOutputRegressor(base, n_jobs=n_jobs)