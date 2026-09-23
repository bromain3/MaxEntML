from sklearn.tree import DecisionTreeRegressor

from MaxEntML.models._regressorBase import BaseRegressorModel


class decisionTreeRegressorModel(BaseRegressorModel):
    """
    Arbre de decision pour la regression. Sans limite de profondeur (max_depth=None)
    l'arbre peut surapprendre sur un petit jeu de donnees sans bruit, ce qui n'est pas
    forcement genant ici pour les memes raisons que pour le kNN (on veut le lambda exact
    si l'entree est deja dans la dtb) mais peut mal generaliser en interpolation :
    a ajuster via max_depth / min_samples_leaf si besoin.
    """

    def __init__(self, max_depth=None, min_samples_leaf=1, random_state=0):
        super().__init__()
        self.core = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
