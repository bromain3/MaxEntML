from MaxEntML.dtbProcessing._BaseTransform import _BaseTransform
from MaxEntML.dataRepresentation.serie import serie


class inverse(_BaseTransform):
    """
    Transformation y = 1 / x, appliquée à toutes les colonnes.

    Sans état, et involutive : la transformation inverse est la même fonction.

    Comportement en 0 : on suit l'arithmétique IEEE de Torch, sans protection.
        1 / 0  ->  +inf      1 / -0.0  ->  -inf      1 / nan  ->  nan
    Les jacobiennes valent -1/x^2 (resp. -1/y^2 pour J_1) et sortent donc -inf en 0.
    Attention : les inf se propagent aux scalers suivants d'un pipeline (par
    exemple un stdScaler ajusté sur des données contenant des inf aura une
    moyenne/variance non finie).

    Jacobiennes diagonales de forme (N, F), définies uniquement pour les entrées 2D.
    """

    def __checkdim(self, x):
        if x.dim() not in (2, 3):
            raise ValueError("Input must be 2D (N, F) or 3D (N, H, W)")

    def __checkdim_serie(self, x):
        self.__checkdim(x.data)

    def __check_not_matrix_for_jacobian(self, x):
        if x.data.dim() != 2:
            raise RuntimeError(
                "J and J_1 are only defined for 2D inputs (N, F). "
                "Flatten your input to (N, H*W) before using J / J_1."
            )

    def fit(self, x):
        self.__checkdim_serie(x)
        return self

    def transform(self, x):
        self.__checkdim_serie(x)
        return serie(1.0 / x.data, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def transformTorch(self, x):
        self.__checkdim(x)
        return 1.0 / x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return self.transform(x)  # x -> 1/x est sa propre inverse

    def inverse_transformTorch(self, x):
        return self.transformTorch(x)

    def J(self, x):
        """dy/dx = -1 / x^2, évaluée en x."""
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian(x)
        return -(1.0 / x.data) ** 2

    def J_1(self, x):
        """dx/dy = -1 / y^2, évaluée en y."""
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian(x)
        return -(1.0 / x.data) ** 2

    def __repr__(self) -> str:
        return "invScaler()"