"""
Driver script: Lorenz63 Ensemble Predictability Experiment

Produces a 3-panel figure showing how predictability depends on where
you start on the Lorenz attractor. Saves lorenz_ensemble_predictability.png.
"""
import numpy as np
from .lorenz63 import Lorenz63
from .plotting import plot_ensemble_panels

def main():

    # --- Configuration ---
    DT = 0.01
    SPINUP_STEPS = 500000       # let transients die out
    REFERENCE_STEPS = 20000   # long trajectory for background attractor
    ENSEMBLE_STEPS = 50     # how far to integrate each ensemble
    N_MEMBERS = 30            # ensemble size
    PERTURBATION_SCALE = 0.3  # std dev of initial perturbations
    SAVE_PATH = "lorenz-ens_predictability.png"


    # Step 1 — Create model
    model = Lorenz63()

    # Step 2 — Generate reference trajectory
    spinup    = model.run(np.array([1.0, 1.0, 1.0]), DT, SPINUP_STEPS)
    reference = model.run(spinup[-1], DT, REFERENCE_STEPS)

    # Step 3 — Create initial condition clouds
    np.random.seed(42)
    deep_left_state = np.array([-15, 1, 45])
    high_left_state = np.array([-8, -3, 34])
    saddle_state    = np.array([0, 0, 18])
    ics_deep   = deep_left_state + np.random.randn(N_MEMBERS, 3) * PERTURBATION_SCALE
    ics_high   = high_left_state + np.random.randn(N_MEMBERS, 3) * PERTURBATION_SCALE
    ics_saddle = saddle_state    + np.random.randn(N_MEMBERS, 3) * PERTURBATION_SCALE

    # Step 4 — Run ensembles
    ensemble_deep   = model.run_ensemble(ics_deep,   DT, ENSEMBLE_STEPS)
    ensemble_high   = model.run_ensemble(ics_high,   DT, ENSEMBLE_STEPS)
    ensemble_saddle = model.run_ensemble(ics_saddle, DT, ENSEMBLE_STEPS)

    # Step 5 — Plot
    fig, axes = plot_ensemble_panels(
        [ensemble_deep, ensemble_high, ensemble_saddle],
        reference,
        ["(a) Deep left lobe", "(b) High left lobe", "(c) Saddle region"],
        save_path=SAVE_PATH,
    )
    print(f"Figure saved to {SAVE_PATH}")
if __name__ == "__main__":
    main()