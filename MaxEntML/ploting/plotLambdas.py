import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# Utilitaires communs
# ============================================================================

def _extract_data(qr, lambdas):
    """Convertit qr['q'], qr['r'] ou qr['x'], qr['y'] et lambdas (tenseurs) en tableaux numpy et renvoie les étiquettes d'axes."""
    if 'q' in qr.colName and 'r' in qr.colName:
        q_flat = qr['q'].data.cpu().detach().numpy().ravel()
        r_flat = qr['r'].data.cpu().detach().numpy().ravel()
        xlabel, ylabel = "$q^*$", "$r^*$"
    elif 'x' in qr.colName and 'y' in qr.colName:
        q_flat = qr['x'].data.cpu().detach().numpy().ravel()
        r_flat = qr['y'].data.cpu().detach().numpy().ravel()
        xlabel, ylabel = "$x$", "$y$"
    else:
        raise ValueError('Impossible to extract q and r or x and y from the data')

    lambdas_flat = lambdas.data.cpu().detach().numpy()
    if lambdas_flat.ndim == 1:
        lambdas_flat = lambdas_flat[:, None]
    return q_flat, r_flat, lambdas_flat, xlabel, ylabel


def _levels_from_norm(norm: mcolors.Normalize, n_levels: int) -> np.ndarray:
    """
    Niveaux de contour équi-répartis dans l'espace *transformé* par la norme
    (log ou symlog) et non dans l'espace linéaire des données.
    """
    levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, n_levels)), dtype=float)
    levels[0], levels[-1] = norm.vmin, norm.vmax  # évite les erreurs d'arrondi aux bornes
    return np.unique(levels)


# --- Fabriques de normes / validité, partagées par les versions 2D et 3D ---

def _linear_norm_factory(vals):
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not vmax > vmin:
        return None
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _log_norm_factory(vals):
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not vmax > vmin:
        return None
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _make_symlog_norm_factory(linthresh, linscale):
    def factory(vals):
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if not vmax > vmin:
            return None
        lt = linthresh if linthresh is not None else max(
            1e-2, float(np.percentile(np.abs(vals), 15)))
        return mcolors.SymLogNorm(linthresh=lt, linscale=linscale,
                                  vmin=vmin, vmax=vmax, base=10)
    return factory


def _valid_finite(L_k):
    return np.isfinite(L_k)


def _valid_positive(L_k):
    with np.errstate(invalid="ignore"):
        return np.isfinite(L_k) & (L_k > 0)


# ============================================================================
# Version 2D (contours, tricontourf)
# ============================================================================

def _plot_phase_space(ax, q_flat, r_flat, xlabel="$q^*$", ylabel="$r^*$"):
    """Subplot de l'espace des phases (adapté aux étiquettes dynamiques)."""
    ax.scatter(q_flat, r_flat, s=8, c="tab:blue", zorder=3, label="Points")

    if xlabel == "$q^*$":
        q_ref = np.linspace(np.min(q_flat) - 0.1, np.max(q_flat) + 0.1, 400)
        ax.plot(q_ref, 1.0 + q_ref**2, "r--", lw=1.2, label=r"Boundary ($r^* = 1 + q^{*2}$)")
        ax.vlines(0, 1.0, np.max(r_flat), colors="k", linestyles=":", alpha=0.6,
                  label="Junk Line ($q^*=0$)")
        title_str = r"Physical Phase Space $(q^*, r^*)$"
    else:
        title_str = f"Physical Phase Space ({xlabel.strip('$')}, {ylabel.strip('$')})"

    ax.set_title(title_str, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def _plot_lambdas_normed(qr, lambdas, *, norm_factory, valid_fn, cmap, n_levels,
                         chunk_size, scale_name):
    """Coeur commun aux versions 2D."""
    q_flat, r_flat, lambdas_flat, xlabel, ylabel = _extract_data(qr, lambdas)
    n_lambdas = lambdas_flat.shape[1]

    # --- Triangulation unique, partagée par tous les subplots ---
    tri = mtri.Triangulation(q_flat, r_flat)
    flat_mask = mtri.TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio=0.01)
    tri.set_mask(flat_mask)

    figs = []
    for start in range(0, n_lambdas, chunk_size):
        stop = min(start + chunk_size, n_lambdas)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        figs.append(fig)

        for slot, k in enumerate(range(start, stop)):
            ax = axes[slot]
            L_k = lambdas_flat[:, k]

            # Points admissibles pour l'échelle choisie (fini, et > 0 pour le log)
            valid_pts = valid_fn(L_k)
            tri_mask = flat_mask | (~valid_pts)[tri.triangles].any(axis=1)

            norm = None
            if not tri_mask.all():
                # Norme calculée uniquement sur les sommets des triangles affichés
                used = np.unique(tri.triangles[~tri_mask])
                norm = norm_factory(L_k[used])

            if norm is None:
                ax.text(0.5, 0.5, "Aucune valeur valide\n(ou valeurs constantes)",
                        ha="center", va="center", transform=ax.transAxes)
            else:
                tri_k = mtri.Triangulation(q_flat, r_flat, triangles=tri.triangles,
                                           mask=tri_mask)
                levels = _levels_from_norm(norm, n_levels)
                safe_L = np.where(valid_pts, L_k, norm.vmin)
                ax.tricontourf(tri_k, safe_L, levels=levels, cmap=cmap, norm=norm)

                # Colorbar pilotée par la norme -> graduations log / symlog correctes
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                fig.colorbar(sm, ax=ax)

            ax.set_title(f"Multiplier $\\lambda_{{{k}}}$ ({scale_name})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.4)

        # Cache les subplots inutilisés (dernier paquet incomplet)
        for slot in range(stop - start, 5):
            axes[slot].axis("off")

        _plot_phase_space(axes[5], q_flat, r_flat, xlabel=xlabel, ylabel=ylabel)

        fig.suptitle(f"Multipliers $\\lambda_{{{start}}}$ – $\\lambda_{{{stop - 1}}}$ "
                     f"— {scale_name} scale", fontsize=13, fontweight="bold")
        fig.tight_layout()

    plt.show()
    return figs


def plotLambdasSymLog(qr: "serie", lambdas: "serie", chunk_size: int = 5,
                      linthresh: float | None = None, linscale: float = 1.0,
                      n_levels: int = 30, cmap: str = "coolwarm"):
    """
    Chaque lambda_k est affiché avec une échelle bi-log (SymLogNorm, base 10) :
    logarithmique de part et d'autre de 0, linéaire dans [-linthresh, +linthresh].
    Gère donc les valeurs positives, négatives et nulles.
    """
    return _plot_lambdas_normed(
        qr, lambdas,
        norm_factory=_make_symlog_norm_factory(linthresh, linscale),
        valid_fn=_valid_finite,
        cmap=cmap, n_levels=n_levels, chunk_size=chunk_size,
        scale_name="SymLog",
    )


def plotLambdasLog(qr: "serie", lambdas: "serie", chunk_size: int = 5,
                   n_levels: int = 30, cmap: str = "viridis"):
    """
    Chaque lambda_k est affiché avec une échelle log classique (LogNorm, base 10).
    Les points où lambda_k <= 0 (ou non fini) ne sont pas représentables : les
    triangles qui les contiennent sont masqués.
    """
    return _plot_lambdas_normed(
        qr, lambdas,
        norm_factory=_log_norm_factory,
        valid_fn=_valid_positive,
        cmap=cmap, n_levels=n_levels, chunk_size=chunk_size,
        scale_name="Log",
    )


def plotLambdas(qr: "serie", lambdas: "serie", chunk_size: int = 5,
                n_levels: int = 30, cmap: str = "viridis"):
    """
    Chaque lambda_k est affiché avec une échelle linéaire (Normalize), niveaux
    équi-répartis entre le min et le max de lambda_k. Sert de référence pour
    comparer avec les échelles SymLog et Log.
    """
    return _plot_lambdas_normed(
        qr, lambdas,
        norm_factory=_linear_norm_factory,
        valid_fn=_valid_finite,
        cmap=cmap, n_levels=n_levels, chunk_size=chunk_size,
        scale_name="Linear",
    )


# ============================================================================
# Version 3D (surfaces, plot_trisurf)
# ============================================================================

def _plot_phase_space_3d(ax, q_flat, r_flat, xlabel="$q^*$", ylabel="$r^*$"):
    """Version 3D du subplot phase space, aplati sur z=0 pour servir de repère."""
    ax.scatter(q_flat, r_flat, 0, s=8, c="tab:blue", zorder=3, label="Points")

    if xlabel == "$q^*$":
        q_ref = np.linspace(np.min(q_flat) - 0.1, np.max(q_flat) + 0.1, 400)
        ax.plot(q_ref, 1.0 + q_ref**2, 0, "r--", lw=1.2, label=r"Boundary ($r^* = 1 + q^{*2}$)")
        title_str = r"Physical Phase Space $(q^*, r^*)$"
    else:
        title_str = f"Physical Phase Space ({xlabel.strip('$')}, {ylabel.strip('$')})"

    ax.set_title(title_str, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_zticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def _plot_lambdas_normed_3d(qr, lambdas, *, norm_factory, valid_fn, cmap, n_levels,
                            chunk_size, scale_name):
    """Equivalent 3D de _plot_lambdas_normed."""
    q_flat, r_flat, lambdas_flat, xlabel, ylabel = _extract_data(qr, lambdas)
    n_lambdas = lambdas_flat.shape[1]

    tri = mtri.Triangulation(q_flat, r_flat)
    flat_mask = mtri.TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio=0.01)
    tri.set_mask(flat_mask)

    figs = []
    for start in range(0, n_lambdas, chunk_size):
        stop = min(start + chunk_size, n_lambdas)

        fig = plt.figure(figsize=(16, 9))
        axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
        figs.append(fig)

        for slot, k in enumerate(range(start, stop)):
            ax = axes[slot]
            L_k = lambdas_flat[:, k]

            valid_pts = valid_fn(L_k)
            tri_mask = flat_mask | (~valid_pts)[tri.triangles].any(axis=1)

            norm = None
            if not tri_mask.all():
                used = np.unique(tri.triangles[~tri_mask])
                norm = norm_factory(L_k[used])

            if norm is None:
                ax.text2D(0.5, 0.5, "Aucune valeur valide\n(ou valeurs constantes)",
                          ha="center", va="center", transform=ax.transAxes)
            else:
                tri_k = mtri.Triangulation(q_flat, r_flat, triangles=tri.triangles,
                                           mask=tri_mask)
                safe_L = np.where(valid_pts, L_k, norm.vmin)
                surf = ax.plot_trisurf(tri_k, safe_L, cmap=cmap, norm=norm,
                                       linewidth=0.1, antialiased=True)
                fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

            ax.set_title(f"Multiplier $\\lambda_{{{k}}}$ ({scale_name})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_zlabel(f"$\\lambda_{{{k}}}$", fontsize=10)

        for slot in range(stop - start, 5):
            axes[slot].axis("off")

        _plot_phase_space_3d(axes[5], q_flat, r_flat, xlabel=xlabel, ylabel=ylabel)

        fig.suptitle(f"Multipliers $\\lambda_{{{start}}}$ – $\\lambda_{{{stop - 1}}}$ "
                     f"— {scale_name} scale (3D)", fontsize=13, fontweight="bold")
        fig.tight_layout()

    plt.show()
    return figs


def plotLambdasSymLog3D(qr: "serie", lambdas: "serie", chunk_size: int = 5,
                        linthresh: float | None = None, linscale: float = 1.0,
                        n_levels: int = 30, cmap: str = "coolwarm"):
    """Equivalent 3D (surface) de plotLambdasSymLog."""
    return _plot_lambdas_normed_3d(
        qr, lambdas,
        norm_factory=_make_symlog_norm_factory(linthresh, linscale),
        valid_fn=_valid_finite,
        cmap=cmap, n_levels=n_levels, chunk_size=chunk_size,
        scale_name="SymLog",
    )


def plotLambdasLog3D(qr: "serie", lambdas: "serie", chunk_size: int = 5,
                     n_levels: int = 30, cmap: str = "viridis"):
    """Equivalent 3D (surface) de plotLambdasLog."""
    return _plot_lambdas_normed_3d(
        qr, lambdas,
        norm_factory=_log_norm_factory,
        valid_fn=_valid_positive,
        cmap=cmap, n_levels=n_levels, chunk_size=chunk_size,
        scale_name="Log",
    )


def plotLambdas3D(qr: "serie", lambdas: "serie", chunk_size: int = 5,
                  n_levels: int = 30, cmap: str = "viridis"):
    """Equivalent 3D (surface) de plotLambdas (échelle linéaire)."""
    return _plot_lambdas_normed_3d(
        qr, lambdas,
        norm_factory=_linear_norm_factory,
        valid_fn=_valid_finite,
        cmap=cmap, n_levels=n_levels, chunk_size=chunk_size,
        scale_name="Linear",
    )