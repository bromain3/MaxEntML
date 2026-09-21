from MaxEntML.dtbProcessing._BaseTransform import _BaseTransform
from MaxEntML.dataRepresentation.serie import serie

import math
import torch


class qr2xyScaler(_BaseTransform):
    """
    Changement de variables (q, r) -> (u, v), inverse de la paramétrisation
    utilisée dans dtbGeneration pour construire la grille :

        q = q_max * sinh(beta_q * u) / sinh(beta_q)        (q = q_max * u si beta_q == 0)
        r = r_low(q) + (r_max - r_low(q)) * v**gamma_r ,   r_low(q) = 1 + q**2 + eps

    avec u in [-1, 1] et v in [0, 1]. Le scaler est sans état (tous les
    paramètres sont fixés à la construction) : fit() ne fait que valider les
    colonnes.

    Convention : l'entrée de transform() est (N, 2) avec colName == ['q', 'r'],
    la sortie est (N, 2) avec colName == ['u', 'v']. inverse_transform() fait
    l'inverse. Les versions Torch supposent colonne 0 = q/u et colonne 1 = r/v.

    ATTENTION - jacobiennes : contrairement aux autres scalers, la jacobienne
    n'est PAS diagonale (v dépend de q et de r). J et J_1 renvoient donc des
    matrices complètes de forme (N, 2, 2), avec J[n, i, j] = d(sortie_i)/d(entrée_j).

        J   (évaluée en x = (q, r)) = d(u, v) / d(q, r)
        J_1 (évaluée en y = (u, v)) = d(q, r) / d(u, v)

    Les deux matrices sont triangulaires inférieures (du/dr = 0 et dq/dv = 0).
    """

    IN_NAMES = ['q', 'r']
    OUT_NAMES = ['x', 'y']

    def __init__(self, qr2xyParams):
        q_max, r_max, eps, beta_q, gamma_r = qr2xyParams.q_max, qr2xyParams.r_max, qr2xyParams.eps, qr2xyParams.beta_q, qr2xyParams.gamma_r
        if q_max <= 0:
            raise ValueError(f"q_max={q_max} doit être > 0")
        if gamma_r <= 0:
            raise ValueError(f"gamma_r={gamma_r} doit être > 0")
        if r_max <= 1.0 + q_max ** 2 + eps:
            raise ValueError(
                f"r_max={r_max} doit être > 1 + q_max^2 + eps = {1.0 + q_max ** 2 + eps}"
            )

        self.q_max = q_max
        self.r_max = r_max
        self.eps = eps
        self.beta_q = beta_q
        self.gamma_r = gamma_r

        self._sinh_beta = math.sinh(beta_q) if beta_q != 0 else None

    # ------------------------------------------------------------------
    # Vérifications
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shape(data):
        if data.dim() != 2 or data.shape[1] != 2:
            raise ValueError("Input must be 2D (N, 2)")

    def _check_serie(self, x, expected_names):
        self._check_shape(x.data)
        if list(x.colName) != expected_names:
            raise ValueError(f"Expected columns {expected_names}, got {list(x.colName)}")

    # ------------------------------------------------------------------
    # Briques élémentaires (tenseurs Torch, colonnes séparées)
    # ------------------------------------------------------------------

    def _q_of_u(self, u):
        if self.beta_q == 0:
            return self.q_max * u
        return self.q_max * torch.sinh(self.beta_q * u) / self._sinh_beta

    def _u_of_q(self, q):
        if self.beta_q == 0:
            return q / self.q_max
        return torch.asinh(q * (self._sinh_beta / self.q_max)) / self.beta_q

    def _r_low(self, q):
        return 1.0 + q ** 2 + self.eps

    def _r_of_qv(self, q, v):
        r_low = self._r_low(q)
        return r_low + (self.r_max - r_low) * v.clamp(min=0.0).pow(self.gamma_r)

    def _v_of_qr(self, q, r):
        r_low = self._r_low(q)
        w = ((r - r_low) / (self.r_max - r_low)).clamp(min=0.0)  # évite NaN si r < r_low (arrondis)
        return w.pow(1.0 / self.gamma_r)

    def _forward(self, data):
        q, r = data[:, 0], data[:, 1]
        return torch.stack([self._u_of_q(q), self._v_of_qr(q, r)], dim=1)

    def _backward(self, data):
        u, v = data[:, 0], data[:, 1]
        q = self._q_of_u(u)
        return torch.stack([q, self._r_of_qv(q, v)], dim=1)

    def _jacobian_entries(self, q, u, v):
        """
        Coefficients de J_1 = d(q, r)/d(u, v) = [[a, 0], [c, d]] :
            a = dq/du
            c = dr/du = 2 q (1 - v^gamma) * a
            d = dr/dv = (r_max - r_low) * gamma * v^(gamma - 1)
        v est borné par eps machine pour éviter d = 0 / inf en v = 0.
        """
        if self.beta_q == 0:
            a = torch.full_like(q, self.q_max)
        else:
            a = self.q_max * self.beta_q * torch.cosh(self.beta_q * u) / self._sinh_beta

        v_safe = v.clamp(min=torch.finfo(v.dtype).eps)
        c = 2.0 * q * (1.0 - v_safe.pow(self.gamma_r)) * a
        d = (self.r_max - self._r_low(q)) * self.gamma_r * v_safe.pow(self.gamma_r - 1.0)
        return a, c, d

    # ------------------------------------------------------------------
    # API _BaseTransform
    # ------------------------------------------------------------------

    def fit(self, x):
        self._check_serie(x, self.IN_NAMES)
        return self

    def transform(self, x):
        self._check_serie(x, self.IN_NAMES)
        return serie(
            self._forward(x.data),
            self.OUT_NAMES,
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def transformTorch(self, x):
        self._check_shape(x)
        return self._forward(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self._check_serie(x, self.OUT_NAMES)
        return serie(
            self._backward(x.data),
            self.IN_NAMES,
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def inverse_transformTorch(self, x):
        self._check_shape(x)
        return self._backward(x)

    # ------------------------------------------------------------------
    # Jacobiennes — matrices complètes (N, 2, 2), triangulaires inférieures
    # ------------------------------------------------------------------

    def J(self, x):
        """d(u, v)/d(q, r) évaluée en x = (q, r). Inverse de la matrice [[a, 0], [c, d]]."""
        raise ValueError('As the jacobian is not diagonal, _PipelineTransform should be patched for suporting it if we really want to use it')
        self._check_serie(x, self.IN_NAMES)

        data = x.data
        q, r = data[:, 0], data[:, 1]
        u = self._u_of_q(q)
        v = self._v_of_qr(q, r)
        a, c, d = self._jacobian_entries(q, u, v)

        J = torch.zeros(data.shape[0], 2, 2, device=data.device, dtype=data.dtype)
        J[:, 0, 0] = 1.0 / a
        J[:, 1, 0] = -c / (a * d)
        J[:, 1, 1] = 1.0 / d

        return J.to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        raise ValueError('As the jacobian is not diagonal, _PipelineTransform should be patched for suporting it if we really want to use it')
        """d(q, r)/d(u, v) évaluée en y = (u, v)."""
        self._check_serie(x, self.OUT_NAMES)

        data = x.data
        u, v = data[:, 0], data[:, 1]
        q = self._q_of_u(u)
        a, c, d = self._jacobian_entries(q, u, v)

        J_inv = torch.zeros(data.shape[0], 2, 2, device=data.device, dtype=data.dtype)
        J_inv[:, 0, 0] = a
        J_inv[:, 1, 0] = c
        J_inv[:, 1, 1] = d

        return J_inv.to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return (f"qr2xyScaler(q_max={self.q_max}, r_max={self.r_max}, eps={self.eps}, "
                f"beta_q={self.beta_q}, gamma_r={self.gamma_r})")