"""Run the Kirkwood/Neumann numerical solver on the same d-grids as the SSA configs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from configs import baseline_config, increased_dispersal_config, reduced_dispersal_config
from kirkwood_solver import solve

CONFIGS = [
    baseline_config(),
    increased_dispersal_config(),
    reduced_dispersal_config(),
]

HERE = Path(__file__).resolve().parent

def main():
    out_dir = HERE / "kirkwood_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    by_cfg: dict[str, dict[str, np.ndarray]] = {}

    for cfg in CONFIGS:
        d_vals = cfg.get_d_values()
        densities = []
        mean_field = []
        deviations = []
        iterations = []
        steps = []
        converged = []
        n_grids = []

        for d in d_vals:
            s = solve(
                cfg.b,
                cfg.d_prime,
                cfg.sigma,
                cfg.sigma,
                d,
                cfg.L,
            )
            mf = cfg.n_expected(float(d))
            ngrid = int(2 ** np.ceil(np.log2(8 * cfg.L / min(cfg.sigma, cfg.sigma))))
            row = {
                "config": cfg.name,
                "sigma": cfg.sigma,
                "L": cfg.L,
                "b": cfg.b,
                "d_prime": cfg.d_prime,
                "d": float(d),
                "n_kirkwood": float(s["N"]),
                "n_mean_field": float(mf),
                "deviation": float(s["N"] - mf),
                "deviation_per_d": float((s["N"] - mf) / d) if d > 0 else np.nan,
                "iterations": int(s["it"]),
                "step": float(s["step"]),
                "converged": bool(s["converged"]),
                "fft_grid_n": ngrid,
            }
            all_rows.append(row)
            densities.append(s["N"])
            mean_field.append(mf)
            deviations.append(s["N"] - mf)
            iterations.append(s["it"])
            steps.append(s["step"])
            converged.append(s["converged"])
            n_grids.append(ngrid)

        by_cfg[cfg.name] = {
            "d": d_vals,
            "n": np.asarray(densities),
            "mf": np.asarray(mean_field),
            "diff": np.asarray(deviations),
            "iterations": np.asarray(iterations),
            "step": np.asarray(steps),
            "converged": np.asarray(converged),
            "fft_grid_n": np.asarray(n_grids),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(d_vals, densities, "o-", label="Kirkwood")
        ax.plot(d_vals, mean_field, "r--", label=r"Mean field: $(b-d)/d'$", linewidth=2)
        ax.set_xlabel(r"$d$ (intrinsic death rate)")
        ax.set_ylabel(r"Equilibrium density $n$")
        ax.set_title(rf"First Moment: Density vs $d$ ($\sigma$={cfg.sigma})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(d_vals, deviations, "o-")
        ax.axhline(0.0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$d$ (intrinsic death rate)")
        ax.set_ylabel(r"$n_\text{K} - n_\text{MF}$")
        ax.set_title("Deviation from Mean Field")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{cfg.name}: numerical moment solver", y=1.02, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"kirkwood_density_deviation_{cfg.name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
