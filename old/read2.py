import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.tri import Triangulation


# ============================================================================
# Dynamic Norm & Colorbar Formatting Strategy
# ============================================================================

def get_norm_and_locator(vals: np.ndarray):
    """
    Select an optimal matplotlib Norm, Colormap, Locator, and Formatter
    for signed or strictly non-negative data spanning varying dynamic ranges.
    """
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    abs_vals = np.abs(vals)
    nonzero_abs = abs_vals[abs_vals > 1e-12]

    has_pos = np.any(vals > 1e-8)
    has_neg = np.any(vals < -1e-8)

    dyn_range = (np.max(abs_vals) / np.min(nonzero_abs)) if len(nonzero_abs) > 0 else 1.0

    if has_pos and has_neg:
        cmap = "coolwarm"
        if dyn_range > 50:
            linthresh = max(1e-2, float(np.percentile(abs_vals, 15)))
            norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
            locator = mticker.SymmetricalLogLocator(linthresh=linthresh, base=10)
            formatter = mticker.LogFormatterSciNotation(base=10)
        else:
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            locator = mticker.MaxNLocator(nbins=7)
            formatter = mticker.ScalarFormatter()

    elif has_neg:
        cmap = "viridis"
        if dyn_range > 50:
            linthresh = max(1e-3, abs(vmax)) if abs(vmax) > 1e-12 else 1e-3
            norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
            locator = mticker.SymmetricalLogLocator(linthresh=linthresh, base=10)
            formatter = mticker.LogFormatterSciNotation(base=10)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            locator = mticker.MaxNLocator(nbins=7)
            formatter = mticker.ScalarFormatter()

    else:
        cmap = "plasma"
        if dyn_range > 50:
            vmin_pos = max(vmin, 1e-10)
            norm = mcolors.LogNorm(vmin=vmin_pos, vmax=vmax)
            locator = mticker.LogLocator(base=10)
            formatter = mticker.LogFormatterSciNotation(base=10)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            locator = mticker.MaxNLocator(nbins=7)
            formatter = mticker.ScalarFormatter()

    return norm, cmap, locator, formatter


def build_contour_levels(norm: mcolors.Normalize, vmin: float, vmax: float, n_levels: int = 35) -> np.ndarray:
    """Generate monotonically increasing levels tailored to the chosen Norm."""
    if isinstance(norm, mcolors.SymLogNorm):
        tr_min = norm(vmin)
        tr_max = norm(vmax)
        tr_levels = np.linspace(tr_min, tr_max, n_levels)
        levels = norm.inverse(tr_levels)
    elif isinstance(norm, mcolors.LogNorm):
        levels = np.geomspace(norm.vmin, norm.vmax, n_levels)
    elif isinstance(norm, mcolors.TwoSlopeNorm):
        neg = np.linspace(vmin, 0, n_levels // 2, endpoint=False)
        pos = np.linspace(0, vmax, n_levels // 2)
        levels = np.unique(np.concatenate([neg, pos]))
    else:
        levels = np.linspace(vmin, vmax, n_levels)

    return np.unique(levels)


# ============================================================================
# Main Plotting Function
# ============================================================================

def plot_grid_lambdas(input_file: str = "grid_with_lambdas_clean.csv",
                      plot_absolute: bool = False,
                      outpath: str | None = None):
    """
    Reads grid CSV, dynamically reconstructs 2D grid structure with NaN masking,
    and visualizes multiplier contours and physical phase space.
    """
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    u_flat = data[:, 0]
    v_flat = data[:, 1]
    q_flat = data[:, 2]
    r_flat = data[:, 3]
    lambdas_flat = data[:, 4:9]

    # Reconstruct original unique grid indices from u, v computational coordinates
    u_unique = np.unique(u_flat)
    v_unique = np.unique(v_flat)
    Nu = len(u_unique)
    Nv = len(v_unique)

    # Re-map scatter data into structured 2D matrices initialized with NaNs
    Q = np.full((Nv, Nu), np.nan)
    R = np.full((Nv, Nu), np.nan)
    L_2d = np.full((Nv, Nu, 5), np.nan)

    u_idx = np.searchsorted(u_unique, u_flat)
    v_idx = np.searchsorted(v_unique, v_flat)

    Q[v_idx, u_idx] = q_flat
    R[v_idx, u_idx] = r_flat
    for k in range(5):
        L_2d[v_idx, u_idx, k] = lambdas_flat[:, k]

    # Clean non-NaN 1D data vectors for Triangulation
    valid_mask = ~np.isnan(lambdas_flat[:, 0]) & ~np.isnan(q_flat) & ~np.isnan(r_flat)
    q_valid = q_flat[valid_mask]
    r_valid = r_flat[valid_mask]

    # Boundary and range setup
    q_min, q_max = float(np.nanmin(q_flat)), float(np.nanmax(q_flat))
    r_max = float(np.nanmax(r_flat))
    q_line = np.linspace(q_min - 0.1, q_max + 0.1, 400)
    r_line = 1.0 + q_line ** 2

    # Construct Triangulation and mask elements outside realizability limit
    tri = Triangulation(q_valid, r_valid)
    def bad_triangle(tri_idx: int) -> bool:
        i, j, k = tri.triangles[tri_idx]
        qc = (q_valid[i] + q_valid[j] + q_valid[k]) / 3.0
        rc = (r_valid[i] + r_valid[j] + r_valid[k]) / 3.0
        return rc < 1.0 + qc * qc

    tri.set_mask(np.array([bad_triangle(t) for t in range(tri.triangles.shape[0])]))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()

    # --- Panels 0 to 4: Dual Multipliers lambda_0 through lambda_4 ---
    for k in range(5):
        ax = axes[k]
        lam_k = lambdas_flat[valid_mask, k]
        vals = np.abs(lam_k) if plot_absolute else lam_k

        norm, cmap, locator, formatter = get_norm_and_locator(vals)
        vmin, vmax = float(vals.min()), float(vals.max())
        levels = build_contour_levels(norm, vmin, vmax, n_levels=35)

        try:
            cf = ax.tricontourf(tri, vals, levels=levels, norm=norm, cmap=cmap, extend="both")
        except Exception:
            # Fallback to structured grid with masked NaNs
            L_k = L_2d[:, :, k]
            masked_L = np.ma.masked_invalid(np.abs(L_k) if plot_absolute else L_k)
            cf = ax.contourf(Q, R, masked_L, levels=levels, norm=norm, cmap=cmap, extend="both")

        # Overlays
        ax.plot(q_line, r_line, "k--", lw=1.2, label=r"$r^*=1+q^{*2}$")
        ax.fill_between(q_line, r_line, -5, color="0.85", zorder=0)
        ax.vlines(0, 3, r_max, colors="r", linestyles="dashdot", lw=1.3, label="Junk line")
        ax.plot(0, 3, "kx", ms=8, mew=2)

        ax.set_xlim(q_min, q_max)
        ax.set_ylim(0, r_max)
        ax.set_xlabel(r"$q^*$")
        ax.set_ylabel(r"$r^*$")

        label_symbol = rf"$|\lambda_{{{k}}}|$" if plot_absolute else rf"$\lambda_{{{k}}}$"
        mode_str = "absolute" if plot_absolute else "signed"
        ax.set_title(f"Multiplier {label_symbol} ({mode_str})", fontsize=11, fontweight="bold")

        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cbar.locator = locator
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.set_label(label_symbol)

    # --- Panel 5: Physical Phase Space Grid ---
    ax_grid = axes[5]
    for i in range(Nu):
        ax_grid.plot(Q[:, i], R[:, i], "C0-", alpha=0.3, lw=0.8)
    for j in range(Nv):
        ax_grid.plot(Q[j, :], R[j, :], "C0-", alpha=0.3, lw=0.8)

    ax_grid.scatter(q_valid, r_valid, s=8, c="tab:blue", zorder=3, label="Valid Grid Points")
    ax_grid.plot(q_line, r_line, "k--", lw=1.2, label=r"Boundary ($r^* = 1 + q^{*2}$)")
    ax_grid.vlines(0, 3, r_max, colors="r", linestyles="dashdot", lw=1.3, label="Junk Line ($q^*=0$)")
    ax_grid.plot(0, 3, "kx", ms=8, mew=2)

    ax_grid.set_xlim(q_min, q_max)
    ax_grid.set_ylim(0, r_max)
    ax_grid.set_title("Physical Phase Space $(q^*, r^*)$ Grid", fontsize=11, fontweight="bold")
    ax_grid.set_xlabel(r"$q^*$")
    ax_grid.set_ylabel(r"$r^*$")
    ax_grid.grid(True, linestyle=":", alpha=0.4)
    ax_grid.legend(loc="upper right", fontsize=8, framealpha=0.85)

    title_type = "Absolute" if plot_absolute else "Signed"
    sym_range = r"$|\lambda_0| \dots |\lambda_4|$" if plot_absolute else r"$\lambda_0 \dots \lambda_4$"
    fig.suptitle(f"{title_type} MaxEnt Multipliers {sym_range} across $(q^*, r^*)$ Phase Space", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if outpath:
        plt.savefig(outpath, dpi=300)
        print(f"Plot saved successfully to '{outpath}'.")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot lambda contours and physical grid with adaptive colorbars.")
    parser.add_argument("--input", type=str, default="grid_with_lambdas_clean.csv", help="Input CSV path.")
    parser.add_argument("--plot-absolute", action="store_true", help="Plot |lambda_k| absolute magnitudes.")
    parser.add_argument("--output", type=str, default=None, help="Output image file path (optional).")

    args = parser.parse_args()
    plot_grid_lambdas(input_file=args.input, plot_absolute=args.plot_absolute, outpath=args.output)