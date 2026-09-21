import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Dynamic Norm Strategy for Pre-Scaling
# ============================================================================

def get_norm(vals: np.ndarray) -> mcolors.Normalize:
    """
    Replicates the dynamic colorbar normalization strategy from read2.py.
    Selects SymLogNorm, LogNorm, TwoSlopeNorm, or Normalize based on data dynamic range and sign.
    """
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    abs_vals = np.abs(vals)
    nonzero_abs = abs_vals[abs_vals > 1e-12]

    has_pos = np.any(vals > 1e-8)
    has_neg = np.any(vals < -1e-8)

    dyn_range = (np.max(abs_vals) / np.min(nonzero_abs)) if len(nonzero_abs) > 0 else 1.0

    if has_pos and has_neg:
        if dyn_range > 50:
            linthresh = max(1e-2, float(np.percentile(abs_vals, 15)))
            return mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
        else:
            return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    elif has_neg:
        if dyn_range > 50:
            linthresh = max(1e-3, abs(vmax)) if abs(vmax) > 1e-12 else 1e-3
            return mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
        else:
            return mcolors.Normalize(vmin=vmin, vmax=vmax)

    else:
        if dyn_range > 50:
            vmin_pos = max(vmin, 1e-10)
            return mcolors.LogNorm(vmin=vmin_pos, vmax=vmax)
        else:
            return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ============================================================================
# Linear Plotting Diagnostic Visualization
# ============================================================================

def plot_scaled_results(scaled_data: np.ndarray, outpath: str | None = None):
    """
    Plots the scaled data using simple linear colorbar mapping with no dynamic 
    range or specialized norm handling.
    """
    u_flat = scaled_data[:, 0]
    v_flat = scaled_data[:, 1]
    q_flat = scaled_data[:, 2]
    r_flat = scaled_data[:, 3]
    lambdas_flat = scaled_data[:, 4:9]

    # Reconstruct original unique grid dimensions
    u_unique = np.unique(u_flat)
    v_unique = np.unique(v_flat)
    Nu = len(u_unique)
    Nv = len(v_unique)

    Q = np.full((Nv, Nu), np.nan)
    R = np.full((Nv, Nu), np.nan)
    L_2d = np.full((Nv, Nu, 5), np.nan)

    u_idx = np.searchsorted(u_unique, u_flat)
    v_idx = np.searchsorted(v_unique, v_flat)

    Q[v_idx, u_idx] = q_flat
    R[v_idx, u_idx] = r_flat
    for k in range(5):
        L_2d[v_idx, u_idx, k] = lambdas_flat[:, k]

    valid_mask = ~np.isnan(lambdas_flat[:, 0]) & ~np.isnan(q_flat) & ~np.isnan(r_flat)
    q_valid = q_flat[valid_mask]
    r_valid = r_flat[valid_mask]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()

    # Create simple triangulation
    tri = Triangulation(q_valid, r_valid)

    # --- Panels 0 to 4: Linear Multipliers lambda_0 to lambda_4 ---
    for k in range(5):
        ax = axes[k]
        lam_k = lambdas_flat[valid_mask, k]

        # Standard linear mapping: uniform spacing, default colormap
        try:
            cf = ax.tricontourf(tri, lam_k, levels=35, cmap="coolwarm", extend="both")
        except Exception:
            L_k = L_2d[:, :, k]
            masked_L = np.ma.masked_invalid(L_k)
            cf = ax.contourf(Q, R, masked_L, levels=35, cmap="coolwarm", extend="both")

        ax.set_xlabel(r"$q^*$ (scaled)")
        ax.set_ylabel(r"$r^*$ (scaled)")
        ax.set_title(f"Scaled Multiplier $\lambda_{{{k}}}$ (Linear)", fontsize=11, fontweight="bold")

        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cbar.set_label(f"Scaled $\lambda_{{{k}}}$")

    # --- Panel 5: Scaled Phase Space Grid ---
    ax_grid = axes[5]
    for i in range(Nu):
        ax_grid.plot(Q[:, i], R[:, i], "C0-", alpha=0.3, lw=0.8)
    for j in range(Nv):
        ax_grid.plot(Q[j, :], R[j, :], "C0-", alpha=0.3, lw=0.8)

    ax_grid.scatter(q_valid, r_valid, s=8, c="tab:blue", zorder=3, label="Grid Points")
    ax_grid.set_title("Scaled Phase Space $(q^*, r^*)$ Grid", fontsize=11, fontweight="bold")
    ax_grid.set_xlabel(r"$q^*$ (scaled)")
    ax_grid.set_ylabel(r"$r^*$ (scaled)")
    ax_grid.grid(True, linestyle=":", alpha=0.4)
    ax_grid.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle("Scaled Dataset Diagnostic (Standard Linear Mapping)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if outpath:
        plt.savefig(outpath, dpi=300)
        print(f"Scaled visualization saved to '{outpath}'.")
    else:
        plt.show()


# ============================================================================
# Main Data Transformation & Processing Pipeline
# ============================================================================

def scale_grid_dataset(input_file: str,
                       output_file: str,
                       scale_qr: bool,
                       scale_lambdas: bool,
                       lambda_scaling_mode: str,
                       plot_results: bool,
                       plot_outpath: str | None):
    """
    Loads clean grid dataset, applies feature scaling on q*, r* and/or lambda components,
    and optionally renders diagnostic subplots using standard linear colormaps.
    """
    # 1. Read header and raw matrix data
    with open(input_file, "r") as f:
        header_line = f.readline().strip()
    header = header_line.lstrip("# ").strip()

    data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    scaled_data = data.copy()

    # 2. Scale q* (col 2) and r* (col 3) using StandardScaler
    if scale_qr:
        print("Applying StandardScaler to (q*, r*)...")
        qr_scaler = StandardScaler()
        scaled_data[:, 2:4] = qr_scaler.fit_transform(data[:, 2:4])

    # 3. Scale Multipliers lambda_0 through lambda_4 (cols 4 to 8)
    if scale_lambdas:
        lambdas_raw = data[:, 4:9]
        lambdas_processed = np.zeros_like(lambdas_raw)

        if lambda_scaling_mode == "colormap_and_standard":
            print("Applying Colormap Norm + StandardScaler to lambdas...")
            for k in range(5):
                lam_k = lambdas_raw[:, k]
                norm = get_norm(lam_k)
                normed_k = norm(lam_k)
                normed_k = np.ma.filled(normed_k, fill_value=0.0)
                
                std_scaler = StandardScaler()
                lambdas_processed[:, k] = std_scaler.fit_transform(normed_k.reshape(-1, 1)).flatten()

        elif lambda_scaling_mode == "standard_only":
            print("Applying StandardScaler only to lambdas...")
            std_scaler = StandardScaler()
            lambdas_processed = std_scaler.fit_transform(lambdas_raw)

        scaled_data[:, 4:9] = lambdas_processed

    # 4. Save transformed dataset
    np.savetxt(output_file, scaled_data, delimiter=",", fmt="%.8e", header=header, comments="")
    print(f"Scaled dataset saved successfully to '{output_file}'.")

    # 5. Optional diagnostic plot (Standard Linear Mapping)
    if plot_results:
        plot_scaled_results(scaled_data, outpath=plot_outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale grid coordinates (q*, r*) and MaxEnt multipliers.")
    
    # File arguments
    parser.add_argument("-i", "--input", type=str, default="grid_with_lambdas_clean.csv", help="Input clean CSV file.")
    parser.add_argument("-o", "--output", type=str, default="grid_with_lambdas_scaled.csv", help="Output scaled CSV file.")

    # Independent scaling flags
    parser.add_argument("--scale-qr", action="store_true", help="Enable StandardScaler for (q*, r*).")
    parser.add_argument("--scale-lambdas", action="store_true", help="Enable scaling for lambda multipliers.")
    
    # Lambda scaling strategy choice
    parser.add_argument("--lambda-scaling-mode", type=str, choices=["colormap_and_standard", "standard_only"],
                        default="colormap_and_standard",
                        help="Choose lambda transformation: 'colormap_and_standard' (Norm mapping + z-score) or 'standard_only' (z-score only).")

    # Plotting arguments
    parser.add_argument("--plot", action="store_true", help="Plot the scaled dataset results using standard linear mapping.")
    parser.add_argument("--plot-output", type=str, default=None, help="Path to save output diagnostic figure (optional).")

    args = parser.parse_args()
    
    scale_grid_dataset(
        input_file=args.input,
        output_file=args.output,
        scale_qr=args.scale_qr,
        scale_lambdas=args.scale_lambdas,
        lambda_scaling_mode=args.lambda_scaling_mode,
        plot_results=args.plot,
        plot_outpath=args.plot_output
    )