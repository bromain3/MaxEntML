from MaxEntML.dtbProcessing._BaseTransform import _BaseTransform
from MaxEntML.dataRepresentation.serie import serie

import math
import torch


class symlogScaler(_BaseTransform):
    """
    Transformation "bi-log" (symlog), identique à SymmetricalLogTransform de
    matplotlib (celle utilisée par SymLogNorm) : linéaire dans [-linthresh, +linthresh],
    logarithmique de part et d'autre, continue en 0 et en ±linthresh.

        a = linscale / (1 - 1/base)
        |x| <= linthresh :  y = a * x
        |x| >  linthresh :  y = sign(x) * linthresh * (a + log_base(|x| / linthresh))

    Ce scaler applique la transformation seule : il ne normalise PAS dans [0, 1]
    comme le fait SymLogNorm. Pour retrouver exactement SymLogNorm, le chaîner avec
    un minMaxScaler :   symlogScaler() + minMaxScaler()

    Paramètres
    ----------
    linthresh    : demi-largeur de la zone linéaire. Si None (défaut), estimée dans fit,
                   PAR COLONNE, avec l'heuristique de plotLambdasSymLog :
                   max(linthreshMin, percentile(|x|, percentile)) sur les valeurs finies.
    linscale     : nombre de décades occupées par chaque moitié de la zone linéaire.
    base         : base du logarithme (> 1).
    percentile   : percentile (0-100) utilisé par l'heuristique.
    linthreshMin : plancher de l'heuristique.

    Valeurs non finies : NaN reste NaN, ±inf reste ±inf (dans les deux sens). Elles
    sont ignorées dans l'estimation de linthresh.

    Jacobiennes diagonales de forme (N, F) (elles dépendent de x, donc de l'échantillon),
    définies uniquement pour les entrées 2D.
    """

    def __init__(self, linthresh=None, linscale=1.0, base=10.0, percentile=15.0, linthreshMin=1e-2):
        if linthresh is not None and not linthresh > 0:
            raise ValueError(f"linthresh={linthresh} doit être > 0")
        if not linscale > 0:
            raise ValueError(f"linscale={linscale} doit être > 0")
        if not base > 1:
            raise ValueError(f"base={base} doit être > 1")
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile={percentile} doit être dans [0, 100]")
        if not linthreshMin > 0:
            raise ValueError(f"linthreshMin={linthreshMin} doit être > 0")

        self.linthresh = linthresh
        self.linscale = float(linscale)
        self.base = float(base)
        self.percentile = float(percentile)
        self.linthreshMin = linthreshMin

        self._log_base = math.log(self.base)
        self._linscale_adj = self.linscale / (1.0 - 1.0 / self.base)

        self.linthreshFit = None  # linthresh effectif, un par colonne (shape = data.shape[1:])
        self._input_ndim = None

    def __checkdim(self, x):
        if x.dim() not in (2, 3):
            raise ValueError("Input must be 2D (N, F) or 3D (N, H, W)")

    def __checkdim_serie(self, x):
        self.__checkdim(x.data)

    def __changeDeviceTorch(self, x):
        self.linthreshFit = self.linthreshFit.to(x.device).to(x.dtype)

    def __changeDevice(self, x):
        self.__changeDeviceTorch(x.data)

    def __check_not_matrix_for_jacobian(self):
        if self._input_ndim == 3:
            raise RuntimeError(
                "J and J_1 are not defined when the scaler was fitted on 3D inputs (N, H, W). "
                "Flatten your input to (N, H*W) before using J / J_1."
            )

    def __check_fitted(self):
        if self.linthreshFit is None:
            raise RuntimeError("Scaler not fitted")

    # ------------------------------------------------------------------
    # Estimation de linthresh
    # ------------------------------------------------------------------

    def _estimate_linthresh(self, data):
        """max(linthreshMin, percentile de |x| sur les valeurs finies), par colonne (interpolation linéaire)."""
        finite = torch.isfinite(data)
        inf = torch.full_like(data, float("inf"))
        sorted_abs = torch.where(finite, data.abs(), inf).sort(dim=0).values  # les non finis vont à la fin
        n = finite.sum(dim=0)

        pos = (n - 1).clamp(min=0).to(torch.float64) * (self.percentile / 100.0)
        lo = pos.floor().long()
        hi = pos.ceil().long()
        frac = (pos - lo).to(data.dtype)

        v_lo = sorted_abs.gather(0, lo.unsqueeze(0)).squeeze(0)
        v_hi = sorted_abs.gather(0, hi.unsqueeze(0)).squeeze(0)
        pct = v_lo + (v_hi - v_lo) * frac  # colonne sans valeur finie -> NaN

        return pct.clamp(min=self.linthreshMin)

    def fit(self, x):
        self.__checkdim_serie(x)
        self._input_ndim = x.data.dim()

        data = x.data
        if self.linthresh is not None:
            self.linthreshFit = torch.full(data.shape[1:], float(self.linthresh),
                                           dtype=data.dtype, device=data.device)
        else:
            self.linthreshFit = self._estimate_linthresh(data)

        return self

    # ------------------------------------------------------------------
    # Briques élémentaires
    # ------------------------------------------------------------------

    def _forward(self, x):
        lt, a = self.linthreshFit, self._linscale_adj
        ax = x.abs()
        linear = ax <= lt
        ax_log = torch.where(linear, lt, ax)  # évite log(0) dans la branche non retenue (gradients sains)
        log_part = torch.sign(x) * lt * (a + torch.log(ax_log / lt) / self._log_base)
        return torch.where(linear, x * a, log_part)

    def _backward(self, y):
        lt, a = self.linthreshFit, self._linscale_adj
        ay = y.abs()
        linear = ay <= lt * a
        ay_exp = torch.where(linear, lt * a, ay)  # évite un exp inutile dans la branche non retenue
        exp_part = torch.sign(y) * lt * torch.pow(self.base, ay_exp / lt - a)
        return torch.where(linear, y / a, exp_part)

    def _dforward(self, x):
        """dy/dx en x."""
        lt, a = self.linthreshFit, self._linscale_adj
        ax = x.abs()
        linear = ax <= lt
        ax_log = torch.where(linear, lt, ax)
        return torch.where(linear, torch.full_like(ax, a), lt / (ax_log * self._log_base))

    def _dbackward(self, y):
        """dx/dy en y."""
        lt, a = self.linthreshFit, self._linscale_adj
        ay = y.abs()
        linear = ay <= lt * a
        ay_exp = torch.where(linear, lt * a, ay)
        return torch.where(linear, torch.full_like(ay, 1.0 / a),
                           self._log_base * torch.pow(self.base, ay_exp / lt - a))

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
        return (f"symlogScaler(linthresh={self.linthresh}, linscale={self.linscale}, "
                f"base={self.base})")