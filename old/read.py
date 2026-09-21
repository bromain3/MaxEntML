import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_lambdas_and_grid(input_file: str = "grid_with_lambdas.csv", output_plot: str | None = None):
    """
    Reads aligned CSV data and plots contours for lambda_0..lambda_4 and the physical grid.
    """
    # 1. Load CSV data
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    u_flat = data[:, 0]
    v_flat = data[:, 1]
    q_flat = data[:, 2]
    r_flat = data[:, 3]
    lambdas_flat = data[:, 4:9]

    # 2. Reconstruct original 2D grid dimensions (Nv, Nu)
    Nu = len(np.unique(u_flat))
    Nv = len(np.unique(v_flat))

    Q = q_flat.reshape(Nv, Nu)
    R = r_flat.reshape(Nv, Nu)

    # 3. Create 2x3 subfigure layout
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    # 4. Plot contours for lambda_0 through lambda_4 (Subplots 1 to 5)
    for k in range(5):
        ax = axes[k]
        L_k = lambdas_flat[:, k].reshape(Nv, Nu)

        # Mask invalid/non-converged solver values (NaNs)
        masked_L = np.ma.masked_invalid(L_k)

        contour = ax.contourf(Q, R, masked_L, levels=30, cmap="viridis")
        fig.colorbar(contour, ax=ax)

        ax.set_title(f"Multiplier $\\lambda_{k}$", fontsize=11, fontweight="bold")
        ax.set_xlabel("$q^*$", fontsize=10)
        ax.set_ylabel("$r^*$", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)

    # 5. Subplot 6: Physical Phase Space Grid
    ax_grid = axes[5]
    for i in range(Nu):
        ax_grid.plot(Q[:, i], R[:, i], "C0-", alpha=0.3, lw=0.8)
    for j in range(Nv):
        ax_grid.plot(Q[j, :], R[j, :], "C0-", alpha=0.3, lw=0.8)

    ax_grid.scatter(q_flat, r_flat, s=8, c="tab:blue", zorder=3, label="Grid Points")

    # Add reference boundary and junk line
    q_ref = np.linspace(np.min(q_flat) - 0.1, np.max(q_flat) + 0.1, 400)
    ax_grid.plot(q_ref, 1.0 + q_ref**2, "r--", lw=1.2, label=r"Boundary ($r^* = 1 + q^{*2}$)")
    ax_grid.vlines(0, 1.0, np.max(r_flat), colors="k", linestyles=":", alpha=0.6, label="Junk Line ($q^*=0$)")

    ax_grid.set_title("Physical Phase Space $(q^*, r^*)$", fontsize=11, fontweight="bold")
    ax_grid.set_xlabel("$q^*$", fontsize=10)
    ax_grid.set_ylabel("$r^*$", fontsize=10)
    ax_grid.grid(True, linestyle=":", alpha=0.4)
    ax_grid.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.tight_layout()

    if output_plot:
        plt.savefig(output_plot, dpi=300)
        print(f"Plot saved successfully to '{output_plot}'.")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Lambda Contours and Physical Grid.")
    parser.add_argument("--input", type=str, default="grid_with_lambdas.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None, help="Optional output image file path (e.g. plot.png)")
    
    args = parser.parse_args()
    plot_lambdas_and_grid(args.input, args.output)