# lorenz_ens

Lorenz63 ensemble predictability experiment for ATOC 4815/5815.

## Usage

Without installing:

```bash
python run.py
```

After packaging (lab exercise):

```bash
pip install -e .
run-lorenz
```

Both generate `lorenz_ensemble_predictability.png`.

## Files

- `lorenz63.py` — Lorenz63 model class
- `integrators.py` — Forward Euler integrator
- `plotting.py` — Ensemble visualization
- `run_lorenz_ensemble.py` — Driver script
