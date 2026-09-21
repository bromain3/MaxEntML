from MaxEntML.dtbProcessing._BaseTransform import _BaseTransform
from MaxEntML.dataRepresentation.serie import serie

import torch


class minMaxScaler(_BaseTransform):
    """
    Mise à l'échelle min-max, sur le modèle de stdScaler :

        y = (x - min) / (max - min) * (hi - lo) + lo        avec featureRange = (lo, hi)

    Par défaut featureRange = (0, 1). Le min et le max sont calculés par colonne
    sur les valeurs FINIES uniquement (NaN et inf sont ignorés dans fit).

    Colonnes constantes (max == min, écart relatif < NULL_RELATIVE_RANGE, ou
    max - min < rangeCutoff) : sortie = lo, inverse = min, jacobiennes = 0
    (même convention que stdScaler).

    Les valeurs non finies en entrée restent non finies en sortie, y compris
    dans une colonne constante (elles ne sont pas remplacées par lo / min).

    Jacobiennes diagonales de forme (N, F), définies uniquement pour les entrées 2D.
    """

    NULL_RELATIVE_RANGE = 1e-12

    def __init__(self, featureRange=(0.0, 1.0), rangeCutoff=1e-28):
        lo, hi = featureRange
        if not lo < hi:
            raise ValueError(f"featureRange={featureRange} doit vérifier lo < hi")

        self.featureRange = (float(lo), float(hi))
        self.rangeCutoff = rangeCutoff

        self.data_min = None
        self.data_max = None
        self.range_ = None
        self.is_constant = None
        self._input_ndim = None

    def __checkdim(self, x):
        if x.dim() not in (2, 3):
            raise ValueError("Input must be 2D (N, F) or 3D (N, H, W)")

    def __checkdim_serie(self, x):
        self.__checkdim(x.data)

    def __changeDeviceTorch(self, x):
        self.data_min    = self.data_min.to(x.device).to(x.dtype)
        self.data_max    = self.data_max.to(x.device).to(x.dtype)
        self.range_      = self.range_.to(x.device).to(x.dtype)
        self.is_constant = self.is_constant.to(x.device)

    def __changeDevice(self, x):
        self.__changeDeviceTorch(x.data)

    def __check_not_matrix_for_jacobian(self):
        if self._input_ndim == 3:
            raise RuntimeError(
                "J and J_1 are not defined when the scaler was fitted on 3D inputs (N, H, W). "
                "Flatten your input to (N, H*W) before using J / J_1."
            )

    def __check_fitted(self):
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler not fitted")

    def fit(self, x):
        self.__checkdim_serie(x)
        self._input_ndim = x.data.dim()

        data = x.data
        finite = torch.isfinite(data)  # ignore NaN / inf dans les statistiques
        has_finite = finite.any(dim=0)
        inf = torch.full_like(data, float("inf"))

        data_min = torch.where(finite, data, inf).amin(dim=0)
        data_max = torch.where(finite, data, -inf).amax(dim=0)

        # colonne sans aucune valeur finie : statistiques NaN (la sortie sera NaN)
        nan = torch.full_like(data_min, float("nan"))
        self.data_min = torch.where(has_finite, data_min, nan)
        self.data_max = torch.where(has_finite, data_max, nan)
        self.range_ = self.data_max - self.data_min

        zero_range = (self.range_ == 0)
        scale = torch.maximum(self.data_min.abs(), self.data_max.abs())

        # écart relatif n'est calculé que là où range != 0 et scale != 0
        safe_rel = torch.where(zero_range | (scale == 0), torch.zeros_like(self.range_), self.range_ / scale)
        self.is_constant = zero_range | (safe_rel < self.NULL_RELATIVE_RANGE) | (self.range_ < self.rangeCutoff)

        return self

    # ------------------------------------------------------------------
    # transform / inverse_transform  →  délèguent aux versions Torch
    # ------------------------------------------------------------------

    def transform(self, x):
        self.__checkdim_serie(x)
        self.__check_fitted()
        self.__changeDevice(x)

        scaled = self.transformTorch(x.data)
        return serie(scaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def transformTorch(self, x):
        self.__checkdim(x)
        self.__check_fitted()
        self.__changeDeviceTorch(x)

        lo, hi = self.featureRange
        safe_range = torch.where(self.is_constant, torch.ones_like(self.range_), self.range_)
        scaled = (x - self.data_min) / safe_range * (hi - lo) + lo

        constant_finite = self.is_constant & torch.isfinite(x)
        scaled = torch.where(constant_finite, torch.full_like(scaled, lo), scaled)
        return scaled

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self.__checkdim_serie(x)
        self.__check_fitted()
        self.__changeDevice(x)

        unscaled = self.inverse_transformTorch(x.data)
        return serie(unscaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def inverse_transformTorch(self, x):
        self.__checkdim(x)
        self.__check_fitted()
        self.__changeDeviceTorch(x)

        lo, hi = self.featureRange
        safe_range = torch.where(self.is_constant, torch.ones_like(self.range_), self.range_)
        unscaled = (x - lo) / (hi - lo) * safe_range + self.data_min

        constant_finite = self.is_constant & torch.isfinite(x)
        unscaled = torch.where(constant_finite, self.data_min.expand_as(unscaled), unscaled)
        return unscaled

    # ------------------------------------------------------------------
    # Jacobiens — définis uniquement pour les entrées 2D
    # ------------------------------------------------------------------

    def J(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()
        self.__check_fitted()

        lo, hi = self.featureRange
        J_diag = torch.zeros_like(self.range_)
        mask = ~self.is_constant
        J_diag[mask] = (hi - lo) / self.range_[mask]

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()
        self.__check_fitted()

        lo, hi = self.featureRange
        J_diag = torch.zeros_like(self.range_)
        mask = ~self.is_constant
        J_diag[mask] = self.range_[mask] / (hi - lo)

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return f"minMaxScaler(featureRange={self.featureRange})"