from MaxEntML.dtbProcessing._BaseTransform import _BaseTransform
from MaxEntML.dataRepresentation.serie import serie

import copy
from collections import Counter

import torch


class featureAugmentation(_BaseTransform):
    """
    Expansion de features : pour chaque colonne, on garde la colonne d'origine
    et on ajoute le résultat de chaque scaler de sa liste, appliqué séparément
    (en parallèle, PAS en chaîne) à la colonne d'origine.

        columnFeaturesScaler([[scalerA, scalerB], [scalerC, scalerD]])

    sur un dataset [x, y] donne [x, A(x), B(x), y, C(y), D(y)].

    columnScalers[i] est la liste des scalers appliqués à la colonne i. Un
    élément peut être un scaler seul (équivaut à [scaler]) ou None / [] (la
    colonne est juste conservée). len(columnScalers) doit être égal au nombre
    de colonnes des données. Chaque scaler est ajusté (fit) sur la colonne
    d'origine, sous forme (N, 1).

    Noms de colonnes en sortie : la colonne d'origine garde son nom, les
    colonnes ajoutées s'appellent "<classe du scaler>(<nom>)", par exemple
    "stdScaler(x)". Si un même nom apparaît plusieurs fois (deux scalers de la
    même classe sur la même colonne), on suffixe "#1", "#2", ...

    Inverse : les features ajoutées sont redondantes, donc inverse_transform
    reconstruit les données uniquement à partir des colonnes d'origine
    (les colonnes ajoutées sont ignorées).

    Jacobiennes : la transformation change le nombre de colonnes (F -> F_out),
    donc J et J_1 sont des matrices complètes, pas des diagonales :
        J   : (N, F_out, F)  avec J[n, i, j] = d(sortie_i) / d(entrée_j)
              (une seule entrée non nulle par ligne : la colonne source)
        J_1 : (N, F, F_out)  dérivée de l'inverse ci-dessus : matrice de
              sélection, 1 sur les colonnes d'origine, 0 ailleurs.

    Contraintes sur les scalers internes : entrée (N, 1) -> sortie (N, 1), et
    jacobienne diagonale (N, 1). Pas de qrToUvScaler ni de columnFeaturesScaler
    imbriqué.
    """

    def __init__(self, columnScalers):
        self.columnScalers = [self._build_column_scalers(s) for s in columnScalers]

        self._identity_idx = []  # indice, dans la sortie, de la colonne d'origine de chaque colonne d'entrée
        offset = 0
        for scalers in self.columnScalers:
            self._identity_idx.append(offset)
            offset += 1 + len(scalers)

        self.nIn = len(self.columnScalers)
        self.nOut = offset

    @staticmethod
    def _build_column_scalers(scalers):
        if scalers is None:
            return []
        if isinstance(scalers, _BaseTransform):
            scalers = [scalers]
        return [copy.deepcopy(s) for s in scalers]  # copie : une même instance peut servir à plusieurs colonnes

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _check_in(self, data):
        if data.ndim != 2 or data.shape[1] != self.nIn:
            raise ValueError(f"Input must be 2D (N, {self.nIn}), got {tuple(data.shape)}")

    def _check_out(self, data):
        if data.ndim != 2 or data.shape[1] != self.nOut:
            raise ValueError(f"Input must be 2D (N, {self.nOut}), got {tuple(data.shape)}")

    @staticmethod
    def _as_column(t, scaler, method):
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(
                f"{method}() of {type(scaler).__name__} must return a single column (N, 1), "
                f"got {tuple(t.shape)}"
            )
        return t

    def _column(self, x, i):
        """Sous-serie (N, 1) contenant la colonne i."""
        return serie(
            x.data[:, i:i + 1],
            [x.colName[i]],
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def _out_names(self, colNames):
        names = []
        for i, scalers in enumerate(self.columnScalers):
            names.append(colNames[i])
            names.extend(f"{type(s).__name__}({colNames[i]})" for s in scalers)

        counts = Counter(names)
        seen = Counter()
        final = []
        for n in names:
            if counts[n] > 1:
                seen[n] += 1
                final.append(f"{n}#{seen[n]}")
            else:
                final.append(n)
        return final

    # ------------------------------------------------------------------
    # API _BaseTransform
    # ------------------------------------------------------------------

    def fit(self, x):
        self._check_in(x.data)
        for i, scalers in enumerate(self.columnScalers):
            col = self._column(x, i)
            for s in scalers:
                s.fit(col)
        return self

    def transform(self, x):
        self._check_in(x.data)
        outs = []
        for i, scalers in enumerate(self.columnScalers):
            col = self._column(x, i)
            outs.append(col.data)
            for s in scalers:
                outs.append(self._as_column(s.transform(col).data, s, "transform"))

        return serie(
            torch.cat(outs, dim=1),
            self._out_names(x.colName),
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def transformTorch(self, x):
        self._check_in(x)
        outs = []
        for i, scalers in enumerate(self.columnScalers):
            col = x[:, i:i + 1]
            outs.append(col)
            for s in scalers:
                outs.append(self._as_column(s.transformTorch(col), s, "transformTorch"))
        return torch.cat(outs, dim=1)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self._check_out(x.data)
        return serie(
            x.data[:, self._identity_idx],
            [x.colName[j] for j in self._identity_idx],
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def inverse_transformTorch(self, x):
        self._check_out(x)
        return x[:, self._identity_idx]

    # ------------------------------------------------------------------
    # Jacobiennes — matrices complètes (voir docstring)
    # ------------------------------------------------------------------

    def J(self, x):
        raise ValueError('As the jacobian is not diagonal, _PipelineTransform should be patched for suporting it if we really want to use it')
        self._check_in(x.data)
        data = x.data

        J = torch.zeros(data.shape[0], self.nOut, self.nIn, dtype=data.dtype, device=data.device)
        for i, scalers in enumerate(self.columnScalers):
            row = self._identity_idx[i]
            J[:, row, i] = 1.0  # colonne d'origine

            col = self._column(x, i)
            for k, s in enumerate(scalers):
                J_k = self._as_column(s.J(col), s, "J")  # jacobienne diagonale (N, 1)
                J[:, row + 1 + k, i] = J_k[:, 0]

        return J

    def J_1(self, x):
        raise ValueError('As the jacobian is not diagonal, _PipelineTransform should be patched for suporting it if we really want to use it')
        self._check_out(x.data)
        data = x.data

        J_inv = torch.zeros(data.shape[0], self.nIn, self.nOut, dtype=data.dtype, device=data.device)
        for i, j in enumerate(self._identity_idx):
            J_inv[:, i, j] = 1.0

        return J_inv

    def __repr__(self) -> str:
        cols = ", ".join("[" + ", ".join(type(s).__name__ for s in scalers) + "]"
                         for scalers in self.columnScalers)
        return f"columnFeaturesScaler([{cols}])"