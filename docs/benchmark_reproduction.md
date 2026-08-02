# Nonlinear benchmark reproduction

Run these commands from the repository root. Results are written to
`example/experiments/artifacts/data`, and figures are written below
`example/experiments/artifacts/figures`.

## Reproduction environment

Use Python 3.12 and the locked reference environment before comparing saved
benchmark values:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -c "import activesysid; print(activesysid.__file__)"
```

The final command must print this checkout's
`src/activesysid/__init__.py`. Do not compare a local source run against a
separately installed release. In particular, the locked JAX, Flax,
`jax-sysid`, NumPy, and SciPy versions are part of the benchmark definition.

The commands use explicit `--save-exp-type` labels so reruns do not overwrite
the historical result files distributed with the repository. Remove that
option only when replacing an existing artifact intentionally.

## Oxidation: five-method comparison

The five methods are Passive, IDWuy, GSx, iGS, and IDW.

Without the output penalty (`delta=1e4`, `alpha=1e2`):

```powershell
python example/experiments/EXP_CMP_AL_sysid_Oxidation.py `
    --method-set 5 `
    --const 0 `
    --delta-set 1e4 `
    --alpha-set 1e2 `
    --save-exp-type rerun_oxidation_cmp_al5_no_penalty
```

With the ordinary output penalty (`delta=3e4`, `alpha=3e2`):

```powershell
python example/experiments/EXP_CMP_AL_sysid_Oxidation.py `
    --method-set 5 `
    --const 1 `
    --delta-set 3e4 `
    --alpha-set 3e2 `
    --save-exp-type rerun_oxidation_cmp_al5_penalty
```

These settings correspond to the historical
`oxidation_RNN_cmp_al5_noise_1_scale_1_const_0_.pkl` and
`oxidation_RNN_cmp_al5_noise_1_scale_1_const_1_.pkl` configurations.

## Oxidation: IDW sensitivity

The stored unconstrained delta sweep fixed `alpha=1e2`:

```powershell
python example/experiments/EXP_CMP_AL_sysid_Oxidation.py `
    --exp-type cmp_delta `
    --const 0 `
    --delta-set 1e2,1e3,1e4,1e5,1e6 `
    --alpha-set 1e2 `
    --save-exp-type rerun_oxidation_cmp_delta_no_penalty
```

The stored unconstrained alpha sweep fixed `delta=1e4`:

```powershell
python example/experiments/EXP_CMP_AL_sysid_Oxidation.py `
    --exp-type cmp_alpha `
    --const 0 `
    --delta-set 1e4 `
    --alpha-set 1,10,100,1000,10000 `
    --save-exp-type rerun_oxidation_cmp_alpha_no_penalty
```

Sensitivity modes compare Passive and IDW, so `--method-set` is not needed.

## Unbalanced Disk: constrained comparisons

The four methods are Passive, GSx, iGS, and IDW. The default constrained
benchmark uses the output bound below. A different positive symmetric bound
can be supplied through `--constraint-bound`.

Output bound `[-4.5, 4.5]` (`delta=1e5`, `alpha=1`):

```powershell
python example/experiments/EXP_CMP_AL_sysid_Unbalanced_Disk.py `
    --method-set 4 `
    --const 1 `
    --constraint-bound 4.5 `
    --delta-set 1e5 `
    --alpha-set 1 `
    --history-refresh-interval 10 `
    --save-exp-type rerun_disk_cmp_al4_const4p5
```

Unique save labels are necessary because the persistence suffix records only
whether constraints are enabled; it does not encode the numerical bound.

## Configuration reference

| Benchmark configuration | Methods | `delta` | `alpha` |
|---|---|---:|---:|
| Oxidation, unconstrained | Passive, IDWuy, GSx, iGS, IDW | `1e4` | `1e2` |
| Oxidation, constrained | Passive, IDWuy, GSx, iGS, IDW | `3e4` | `3e2` |
| Unbalanced Disk, bound `4.5` | Passive, GSx, iGS, IDW | `1e5` | `1` |

All full comparisons use seeds `0` through `9`. Oxidation uses initial,
maximum, and test sizes `60/500/2000`; Unbalanced Disk uses
`60/2000/2000`.

Explicit `--delta-set` and `--alpha-set` values override the constrained or
unconstrained defaults.

## Quick smoke runs

Use one unsaved repetition to check the environment before launching a full
benchmark. These commands retain the benchmark data budgets, optimizer
settings, constraints, and tuned IDW weights.

```powershell
python example/experiments/EXP_CMP_AL_sysid_Oxidation.py `
    --method-set 5 `
    --const 0 `
    --delta-set 1e4 `
    --alpha-set 1e2 `
    --n-exp 1 `
    --no-save `
    --no-plot
```

```powershell
python example/experiments/EXP_CMP_AL_sysid_Unbalanced_Disk.py `
    --method-set 4 `
    --const 1 `
    --constraint-bound 4.5 `
    --delta-set 1e5 `
    --alpha-set 1 `
    --history-refresh-interval 10 `
    --n-exp 1 `
    --no-save `
    --no-plot
```
