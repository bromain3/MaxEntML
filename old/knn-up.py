import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.tri import Triangulation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================================
# Dynamic Norm Generator for Error Contours
# ============================================================================

def get_error_norm(vals: np.ndarray):
    """
    Selects an optimal norm and colorbar locator for visualization of error magnitude.
    """
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    nonzero_vals = vals[vals > 1e-12]

    if len(nonzero_vals) > 0 and (vmax / np.min(nonzero_vals)) > 50:
        vmin_pos = max(vmin, 1e-6)
        norm = mcolors.LogNorm(vmin=vmin_pos, vmax=vmax)
        locator = mticker.LogLocator(base=10)
        formatter = mticker.LogFormatterSciNotation(base=10)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        locator = mticker.MaxNLocator(nbins=7)
        formatter = mticker.ScalarFormatter()

    return norm, locator, formatter


# ============================================================================
# Inverse Transformation Helpers
# ============================================================================

def inverse_transform_lambdas(lambdas_scaled: np.ndarray, scaler_payload: dict) -> np.ndarray:
    """
    Reverts scaled multipliers back to physical values using saved joblib scalers.
    """
    if not scaler_payload.get("scale_lambdas", False):
        return lambdas_scaled.copy()

    mode = scaler_payload.get("lambda_scaling_mode", "standard_only")
    lambdas_phys = np.zeros_like(lambdas_scaled)

    if mode == "colormap_and_standard":
        scalers = scaler_payload["lambda_scalers"]
        norms = scaler_payload["lambda_norms"]
        for k in range(5):
            std_scaler = scalers[k]
            normed_k = std_scaler.inverse_transform(lambdas_scaled[:, k].reshape(-1, 1)).flatten()
            norm = norms[k]
            lambdas_phys[:, k] = norm.inverse(normed_k)

    elif mode == "standard_only":
        std_scaler = scaler_payload["lambda_scalers"][0]
        lambdas_phys = std_scaler.inverse_transform(lambdas_scaled)

    return lambdas_phys


def inverse_transform_qr(qr_scaled: np.ndarray, scaler_payload: dict) -> np.ndarray:
    """
    Reverts scaled phase space coordinates (q*, r*) back to physical values.
    """
    if scaler_payload.get("scale_qr", False) and scaler_payload.get("qr_scaler") is not None:
        return scaler_payload["qr_scaler"].inverse_transform(qr_scaled)
    return qr_scaled.copy()


# ============================================================================
# Main KNN Training & Visualization Pipeline
# ============================================================================

def run_knn_pipeline(input_file: str,
                     scaled_data: bool = False,
                     scalers_file: str = "scalers.joblib",
                     feature_type: str = "qr",
                     test_size: float = 0.25,
                     n_neighbors: int = 5,
                     weights: str = "distance",
                     error_metric: str = "relative",
                     random_state: int = 42,
                     outpath: str | None = None):
    """
    Trains KNN regressor on chosen feature set, predicts lambdas, inverse-transforms
    data if scaled, and plots approximation error contours over physical (q*, r*) space.
    """
    # 1. Load dataset
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    u_flat = data[:, 0]
    v_flat = data[:, 1]
    q_flat = data[:, 2]
    r_flat = data[:, 3]
    lambdas_flat = data[:, 4:9]

    # Handle Scalers loading & Physical Coordinates recovery for plotting
    if scaled_data:
        print(f"Loading scaler configuration from '{scalers_file}'...")
        scaler_payload = joblib.load(scalers_file)

        # Recover physical coordinates for plotting boundary and physical error calculations
        qr_phys = inverse_transform_qr(np.column_stack((q_flat, r_flat)), scaler_payload)
        q_phys, r_phys = qr_phys[:, 0], qr_phys[:, 1]
        lambdas_phys = inverse_transform_lambdas(lambdas_flat, scaler_payload)
    else:
        scaler_payload = None
        q_phys, r_phys = q_flat.copy(), r_flat.copy()
        lambdas_phys = lambdas_flat.copy()

    # Select input features X (using coordinates as provided in dataset)
    if feature_type == "uv":
        X = np.column_stack((u_flat, v_flat))
        feature_name = "$(u, v)$ Computational Coordinates"
    else:
        X = np.column_stack((q_flat, r_flat))
        feature_name = "$(q^*, r^*)$ Phase Space Coordinates" + (" (Scaled)" if scaled_data else "")

    Y = lambdas_flat

    # 2. Train / Test Split (75% train, 25% test)
    (X_train, X_test, 
     Y_train, Y_test, 
     q_train, q_test, 
     r_train, r_test, 
     Y_train_phys, Y_test_phys) = train_test_split(
        X, Y, q_phys, r_phys, lambdas_phys, test_size=test_size, random_state=random_state
    )

    print(f"Dataset Split: {len(X_train)} Train Samples ({100*(1-test_size):.0f}%), "
          f"{len(X_test)} Test Samples ({100*test_size:.0f}%)")

    # 3. Train KNN Regressor
    print(f"Training KNN Regressor (K={n_neighbors}, weights='{weights}') using {feature_name}...")
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, Y_train)

    # Predict on test set
    Y_pred = knn.predict(X_test)

    # 4. Convert predictions back to physical scale if operating on scaled data
    if scaled_data and scaler_payload is not None:
        Y_pred_phys = inverse_transform_lambdas(Y_pred, scaler_payload)
    else:
        Y_pred_phys = Y_pred

    # Compute Approximation Errors on Physical Target Values
    if error_metric == "relative":
        errors = np.abs(Y_test_phys - Y_pred_phys) / (np.abs(Y_test_phys) + 1e-8)
        metric_label = "Relative Error"
    else:
        errors = np.abs(Y_test_phys - Y_pred_phys)
        metric_label = "Absolute Error"

    # Print Summary Performance Report
    print("\n" + "=" * 55)
    print(f"{'Multiplier':<12} | {'MAE':<12} | {'RMSE':<12} | {'Max Error':<12}")
    print("-" * 55)
    for k in range(5):
        mae = mean_absolute_error(Y_test_phys[:, k], Y_pred_phys[:, k])
        rmse = np.sqrt(mean_squared_error(Y_test_phys[:, k], Y_pred_phys[:, k]))
        max_err = np.max(np.abs(Y_test_phys[:, k] - Y_pred_phys[:, k]))
        print(f"lambda_{k:<5} | {mae:<12.4e} | {rmse:<12.4e} | {max_err:<12.4e}")
    print("=" * 55 + "\n")

    # 5. Build Triangulation on Physical Test Set (q*, r*) points
    tri_test = Triangulation(q_test, r_test)

    # Mask triangles crossing outside realizability limit r* = 1 + q*^2
    def bad_triangle(tri_idx: int) -> bool:
        i, j, k = tri_test.triangles[tri_idx]
        qc = (q_test[i] + q_test[j] + q_test[k]) / 3.0
        rc = (r_test[i] + r_test[j] + r_test[k]) / 3.0
        return rc < 1.0 + qc * qc

    tri_test.set_mask(np.array([bad_triangle(t) for t in range(tri_test.triangles.shape[0])]))

    # Setup boundary line for overlays in physical space
    q_min, q_max = float(q_phys.min()), float(q_phys.max())
    r_max = float(r_phys.max())
    q_line = np.linspace(q_min - 0.1, q_max + 0.1, 400)
    r_line = 1.0 + q_line ** 2

    # 6. Plot Subfigures (5 Error panels + 1 Train/Test distribution panel)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()

    # Panels 0-4: Error Contours for lambda_0 to lambda_4
    for k in range(5):
        ax = axes[k]
        err_k = errors[:, k]

        norm, locator, formatter = get_error_norm(err_k)
        vmin, vmax = float(err_k.min()), float(err_k.max())

        if isinstance(norm, mcolors.LogNorm):
            levels = np.geomspace(norm.vmin, norm.vmax, 35)
        else:
            levels = np.linspace(vmin, vmax, 35)

        cf = ax.tricontourf(tri_test, err_k, levels=levels, norm=norm, cmap="inferno", extend="both")

        # Overlays
        ax.plot(q_line, r_line, "k--", lw=1.2, label=r"$r^*=1+q^{*2}$")
        ax.fill_between(q_line, r_line, -5, color="0.85", zorder=0)

        ax.set_xlim(q_min, q_max)
        ax.set_ylim(0, r_max)
        ax.set_xlabel(r"$q^*$")
        ax.set_ylabel(r"$r^*$")
        ax.set_title(f"$\lambda_{{{k}}}$ {metric_label}", fontsize=11, fontweight="bold")

        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cbar.locator = locator
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.set_label(f"|$\lambda_{{{k}}} - \hat{{\lambda}}_{{{k}}}$|" if error_metric == "absolute" else "Relative Error")

    # Panel 5: Train vs Test Data Distribution in Physical (q*, r*) space
    ax_split = axes[5]
    ax_split.scatter(q_train, r_train, s=12, c="tab:blue", alpha=0.5, label=f"Train ({len(q_train)})")
    ax_split.scatter(q_test, r_test, s=16, c="tab:red", alpha=0.8, marker="x", label=f"Test ({len(q_test)})")
    ax_split.plot(q_line, r_line, "k--", lw=1.2, label=r"$r^*=1+q^{*2}$")

    ax_split.set_xlim(q_min, q_max)
    ax_split.set_ylim(0, r_max)
    ax_split.set_title("Train / Test Physical Spatial Distribution", fontsize=11, fontweight="bold")
    ax_split.set_xlabel(r"$q^*$")
    ax_split.set_ylabel(r"$r^*$")
    ax_split.grid(True, linestyle=":", alpha=0.4)
    ax_split.legend(loc="upper right", fontsize=8, framealpha=0.85)

    mode_str = " (Scaled Data Mode)" if scaled_data else " (Unscaled Data Mode)"
    fig.suptitle(f"KNN Multiplier Error using Features: {feature_name} (K={n_neighbors}){mode_str}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if outpath:
        plt.savefig(outpath, dpi=300)
        print(f"Error contour plot saved successfully to '{outpath}'.")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KNN approximation accuracy for MaxEnt multipliers.")

    # Input/Output path options
    parser.add_argument("-i", "--input", type=str, default="grid_with_lambdas_scaled.csv", help="Input dataset path.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output plot image path (optional).")

    # Scaled data options
    parser.add_argument("--scaled-data", action="store_true", default=True, help="Flag to indicate input dataset uses scaled features/lambdas.")
    parser.add_argument("--scalers-file", type=str, default="scalers.joblib", help="Path to joblib file containing fitted scaling transformations.")

    # Feature selection flag
    parser.add_argument("--feature-type", type=str, choices=["qr", "uv"], default="uv",
                        help="Input features for KNN: 'qr' for physical/scaled phase space or 'uv' for computational.")

    # KNN and evaluation hyper-parameters
    parser.add_argument("--k-neighbors", type=int, default=5, help="Number of neighbors K for KNN regressor.")
    parser.add_argument("--weights", type=str, choices=["uniform", "distance"], default="distance", help="KNN weighting scheme.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test set fraction (default: 0.25 for 75/25 split).")
    parser.add_argument("--error-metric", type=str, choices=["absolute", "relative"], default="absolute", help="Error metric for visualization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split reproducibility.")

    args = parser.parse_args()

    run_knn_pipeline(
        input_file=args.input,
        scaled_data=args.scaled_data,
        scalers_file=args.scalers_file,
        feature_type=args.feature_type,
        test_size=args.test_size,
        n_neighbors=args.k_neighbors,
        weights=args.weights,
        error_metric=args.error_metric,
        random_state=args.seed,
        outpath=args.output
    )