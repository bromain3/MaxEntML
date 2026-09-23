# MaxEntML/dtbProcessing/arcsinhScaler.py

from MaxEntML.dtbProcessing._BaseTransform import _BaseTransform
from MaxEntML.dataRepresentation.serie import serie

import torch


class arcsinhScaler(_BaseTransform):
    """
    Transformation "arcsinh" (inverse hyperbolic sine), alternative lisse (C^inf,
    sans point anguleux) à symlogScaler : linéaire près de 0, logarithmique
    au-delà, continue et dérivable partout y compris en x=0.

        y = arcsinh(x / scale) = log( x/scale + sqrt((x/scale)^2 + 1) )
        x = scale * sinh(y)

    Contrairement à symlogScaler, il n'y a pas de zone plate strictement linéaire
    suivie d'un raccord log : la transition est progressive, ce qui évite tout
    artefact de gradient au raccord — utile ici car les lambda changent de signe
    exactement dans les zones à fort gradient (q*=0, bord du domaine).

    Paramètres
    ----------
    scale        : échelle de transition, PAR COLONNE. Si None (défaut), estimée
                   dans fit() avec la même heuristique que symlogScaler :
                   max(scaleMin, percentile(|x|, percentile)) sur les valeurs finies.
    percentile   : percentile (0-100) utilisé par l'heuristique (défaut 75, plus
                   élevé que symlogScaler pour ne garder qu'une échelle "de coeur
                   de distribution", robuste à la queue de la couche limite).
    scaleMin     : plancher de l'heuristique.

    Valeurs non finies : NaN reste NaN, ±inf reste ±inf. Elles sont ignorées
    dans l'estimation de scale.

    Jacobiennes diagonales de forme (N, F) (dépendent de x), définies uniquement
    pour les entrées 2D.
    """

    def __init__(self, scale=None, percentile=90.0, scaleMin=1e-2): # default 75.0 percentile
        if scale is not None and not scale > 0:
            raise ValueError(f"scale={scale} doit être > 0")
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile={percentile} doit être dans [0, 100]")
        if not scaleMin > 0:
            raise ValueError(f"scaleMin={scaleMin} doit être > 0")

        self.scale = scale
        self.percentile = float(percentile)
        self.scaleMin = scaleMin

        self.scaleFit = None  # scale effective, une par colonne (shape = data.shape[1:])
        self._input_ndim = None

    def __checkdim(self, x):
        if x.dim() not in (2, 3):
            raise ValueError("Input must be 2D (N, F) or 3D (N, H, W)")

    def __checkdim_serie(self, x):
        self.__checkdim(x.data)

    def __changeDeviceTorch(self, x):
        self.scaleFit = self.scaleFit.to(x.device).to(x.dtype)

    def __changeDevice(self, x):
        self.__changeDeviceTorch(x.data)

    def __check_not_matrix_for_jacobian(self):
        if self._input_ndim == 3:
            raise RuntimeError(
                "J and J_1 are not defined when the scaler was fitted on 3D inputs (N, H, W). "
                "Flatten your input to (N, H*W) before using J / J_1."
            )

    def __check_fitted(self):
        if self.scaleFit is None:
            raise RuntimeError("Scaler not fitted")

    # ------------------------------------------------------------------
    # Estimation de scale
    # ------------------------------------------------------------------

    def _estimate_scale(self, data):
        """max(scaleMin, percentile de |x| sur les valeurs finies), par colonne (interpolation linéaire)."""
        finite = torch.isfinite(data)
        inf = torch.full_like(data, float("inf"))
        sorted_abs = torch.where(finite, data.abs(), inf).sort(dim=0).values
        n = finite.sum(dim=0)

        pos = (n - 1).clamp(min=0).to(torch.float64) * (self.percentile / 100.0)
        lo = pos.floor().long()
        hi = pos.ceil().long()
        frac = (pos - lo).to(data.dtype)

        v_lo = sorted_abs.gather(0, lo.unsqueeze(0)).squeeze(0)
        v_hi = sorted_abs.gather(0, hi.unsqueeze(0)).squeeze(0)
        pct = v_lo + (v_hi - v_lo) * frac  # colonne sans valeur finie -> NaN

        return pct.clamp(min=self.scaleMin)

    def fit(self, x):
        self.__checkdim_serie(x)
        self._input_ndim = x.data.dim()

        data = x.data
        if self.scale is not None:
            self.scaleFit = torch.full(data.shape[1:], float(self.scale),
                                       dtype=data.dtype, device=data.device)
        else:
            self.scaleFit = self._estimate_scale(data)

        return self

    # ------------------------------------------------------------------
    # Briques élémentaires
    # ------------------------------------------------------------------

    def _forward(self, x):
        return torch.asinh(x / self.scaleFit)

    def _backward(self, y):
        return self.scaleFit * torch.sinh(y)

    def _dforward(self, x):
        """dy/dx en x = 1 / sqrt(scale^2 + x^2)."""
        s = self.scaleFit
        return 1.0 / torch.sqrt(s ** 2 + x ** 2)

    def _dbackward(self, y):
        """dx/dy en y = scale * cosh(y)."""
        return self.scaleFit * torch.cosh(y)

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
        return self._forward(x)

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
        return self._backward(x)

    # ------------------------------------------------------------------
    # Jacobiens — définis uniquement pour les entrées 2D
    # ------------------------------------------------------------------

    def J(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()
        self.__check_fitted()
        self.__changeDevice(x)

        return self._dforward(x.data).to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()
        self.__check_fitted()
        self.__changeDevice(x)

        return self._dbackward(x.data).to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return f"arcsinhScaler(scale={self.scale}, percentile={self.percentile})"