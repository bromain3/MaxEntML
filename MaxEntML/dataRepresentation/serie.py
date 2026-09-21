import numpy as np
import torch

class serie:
    def __init__(self, data, colName, storingDevice=None, storingDtype=None):
        self.colName = colName
        self.ncols = len(colName)
        self.data = self._to_tensor(data, device=storingDevice, dtype=storingDtype)
        self.data = self._normalize_shape(self.data)
        self.storingDevice = storingDevice
        self.storingDtype = storingDtype


    def save(self, path):
        """
        Sauvegarde la série dans un fichier (format torch, ex: 'ma_serie.pt').
        Les données sont stockées sur CPU pour rester portables ; device et dtype
        sont enregistrés sous forme de chaînes.
        """
        payload = {
            "version": 1,
            "colName": list(self.colName),
            "data": self.data.detach().cpu().clone(),  # clone : évite de sauver tout le storage d'une vue
            "storingDevice": None if self.storingDevice is None else str(self.storingDevice),
            "storingDtype": None if self.storingDtype is None else str(self.storingDtype).replace("torch.", ""),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device=None):
        """
        Charge une série sauvegardée avec `save`.
        - device=None : utilise le device d'origine (repli sur CPU si CUDA indisponible).
        - device='cpu' / 'cuda' / ... : force le device de destination.
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)

        savedDtype = payload["storingDtype"]
        storingDtype = getattr(torch, savedDtype) if savedDtype is not None else None

        target = device if device is not None else payload["storingDevice"]
        if target is not None and str(target).startswith("cuda") and not torch.cuda.is_available():
            target = "cpu"

        data = payload["data"]
        if target is not None:
            data = data.to(target)
            target = torch.device(target)

        return cls._from_raw(data, payload["colName"], target, storingDtype)

    def _to_tensor(self, data, device=None, dtype=None):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype) if (device or dtype) else data

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=device, dtype=dtype)

        return torch.tensor(data, device=device, dtype=dtype)

    def _normalize_shape(self, t):
        # scalar
        if t.dim() == 0:
            return t.unsqueeze(0).unsqueeze(1)

        # 1D case → disambiguation
        if t.dim() == 1:
            if self.ncols == 1:
                # single feature → column vector (N,1)
                return t.unsqueeze(1)

            elif self.ncols == len(t):
                # single sample → row vector (1,N)
                return t.unsqueeze(0)

            else:
                raise ValueError(f"Ambiguous shape: len(data)={len(t)} vs ncols={self.ncols}")

        return t

    @classmethod
    def _from_raw(cls, tensor, colName, storingDevice, storingDtype):
        """Constructeur rapide : skip validation, pour usage interne uniquement."""
        obj = object.__new__(cls)
        obj.colName = colName
        obj.ncols = len(colName)
        obj.data = tensor
        obj.storingDevice = storingDevice
        obj.storingDtype = storingDtype
        return obj

    @property
    def nSamples(self):
        return self.data.shape[0]

    @property
    def nFeatures(self):
        return self.ncols

    def to(self, device):
        self.data = self.data.to(device, non_blocking=True)
        self.storingDevice = self.data.device
        self.storingDtype = self.data.dtype
        return self

    def contiguous(self):
        self.data = self.data.contiguous()
        return self

    def __getitem__(self, key):
        #raise TypeError()  # if key is not of the right type
        #raise IndexError()  # if key is numerical but outside the range of allowed values (used by for loops to stop)
        #raise KeyError()  # if key is a special type and cannot be accessed
        if isinstance(key, str):
            idx = self.colName.index(key)
            # CAS 1 : plusieurs colonnes 2D
            if self.data.dim() >= 2 and self.nSamples > 1:
                return self.data[:, idx]
            # CAS 2 : une seule colonne ND
            if self.data.dim() > 2 and self.ncols == 1:
                return self.data  # shape (N, ...)
            # CAS 3 : plusieurs colones mais une seule ligne (retour scalaire)
            if self.nSamples == 1:
                return self.data[0, idx].detach().cpu().item()
            else:
                raise ValueError("shape/columns not coherent")

        elif isinstance(key, (int, slice, list, tuple, torch.Tensor, np.ndarray)):
            return serie(self.data[key], self.colName, storingDevice=self.storingDevice, storingDtype=self.storingDtype)

        else:
            raise KeyError(f"type {type(key)} not supported as serie key")

    def drop(self, cols):
        """
        Remove columns whose names are in `cols`.
        Keeps original column order.
        """
        if isinstance(cols, str):
            cols = [cols]

        # vérification
        missing = [c for c in cols if c not in self.colName]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        # colonnes conservées dans l'ordre original
        keep_idx = [i for i, c in enumerate(self.colName) if c not in cols]
        keep_cols = [self.colName[i] for i in keep_idx]

        # CAS 1 : structure classique (N, F, ...)
        if self.data.dim() >= 2:
            new_data = self.data[:, keep_idx]

        # CAS 2 : série avec une seule colonne ND
        elif self.ncols == 1:
            if len(keep_idx) == 0:
                raise ValueError("Cannot drop the only column")
            new_data = self.data

        else:
            raise ValueError("Unsupported tensor shape")

        return serie._from_raw(
            new_data,
            keep_cols,
            self.storingDevice,
            self.storingDtype
        )

    def clone(self):
        return serie._from_raw(
            self.data.clone(),
            self.colName.copy(),
            self.storingDevice,
            self.storingDtype
        )

def _columnsView(s):
    """
    Retourne le tenseur de `s` sous la forme (N, F, ...) où la dim 1 est toujours l'axe des colonnes.
    - Structure classique (N, F, ...) avec F == ncols : renvoyée telle quelle.
    - Colonne unique ND (N, ...) avec ncols == 1 : on insère l'axe colonne -> (N, 1, ...).
    Si ncols == 1 et shape[1] == 1, la série est considérée comme classique (pas d'ambiguïté).
    """
    t = s.data
    if t.dim() >= 2 and t.shape[1] == s.ncols:
        return t
    if s.ncols == 1:
        return t.unsqueeze(1)
    raise ValueError(
        f"Incoherent serie: data shape {tuple(t.shape)} vs ncols={s.ncols}"
    )


def agregateColumns(a):
    """
    Agrège horizontalement une liste de séries de même nSamples.
    Retourne une nouvelle série contenant les colonnes de toutes les séries,
    dans l'ordre de la liste.

    Gère :
    - les séries classiques (N, F, ...)
    - les séries à colonne unique ND (N, ...)
    - les séries d'un seul élément (retourne une copie)

    Erreurs :
    - TypeError  : entrée invalide
    - ValueError : nSamples différents, dimensions ND incompatibles,
                   noms de colonnes dupliqués, série incohérente
    """
    if not isinstance(a, (list, tuple)) or len(a) == 0:
        raise TypeError("Input must be a non-empty list")

    if not all(isinstance(x, serie) for x in a):
        raise TypeError("Input must be a list of series")

    first = a[0]

    # 1) Même nombre d'échantillons
    for i, s in enumerate(a):
        if s.nSamples != first.nSamples:
            raise ValueError(
                f"All series must have the same number of samples "
                f"(serie {i}: {s.nSamples}, serie 0: {first.nSamples})"
            )

    # 2) Noms de colonnes uniques (à l'intérieur d'une série et entre séries)
    allNames = [name for s in a for name in s.colName]
    seen, duplicates = set(), set()
    for name in allNames:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Duplicate column names: {sorted(duplicates)}")

    # 3) Mise au format (N, F, ...) + alignement device/dtype sur la première série
    ref = _columnsView(first)
    tensors = [
        _columnsView(s).to(device=ref.device, dtype=ref.dtype)
        for s in a
    ]

    # 4) Dimensions ND (au-delà de N et F) identiques
    refTrailing = tensors[0].shape[2:]
    for i, t in enumerate(tensors):
        if t.shape[2:] != refTrailing:
            raise ValueError(
                f"Incompatible trailing dimensions: serie {i} has {tuple(t.shape[2:])}, "
                f"serie 0 has {tuple(refTrailing)}"
            )

    # 5) Concaténation le long de l'axe des colonnes
    data = torch.cat(tensors, dim=1)

    return serie._from_raw(data, allNames, first.storingDevice, first.storingDtype)

def concat(a):
    if not isinstance(a, (list, tuple)) or len(a) == 0:
        raise TypeError("Input must be a non-empty list")

    # CAS 1 : liste de series
    if all(isinstance(x, serie) for x in a):
        first = a[0]
        for s in a:
            if s.colName != first.colName:
                raise ValueError("All series must have the same columns")
        data = torch.vstack([s.data for s in a])
        return serie._from_raw(data, first.colName, first.storingDevice, first.storingDtype)

    else:
        raise TypeError("Input must be a list of series")