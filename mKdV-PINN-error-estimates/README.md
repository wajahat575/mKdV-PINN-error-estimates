# Error estimates for a PINN solving the modified Korteweg-de Vries equation

Code for the paper

> H. W. A. Riaz, A. Alshuhail and Y. Yildirim, *Error estimates for a
> physics-informed neural network solving the modified Korteweg-de Vries
> equation*.

The paper proves error bounds for a tanh network with two hidden layers trained
as a physics-informed neural network (PINN) for

    v_t + alpha v^2 v_x + beta v_xxx = 0

on a bounded interval, with boundary data taken from the solution. This
repository contains the code for every numerical experiment of the paper and
the script that produces its figures, tables and reported numbers.

## Contents

```
code/
  pinn_mkdv.py   PINN trainer; every experiment is a named job
  figures.py     figures, tables and the numbers reported in the paper
requirements.txt pinned versions of the Python packages
```

## Installation

The code runs on a CPU and needs Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Train the networks. Each job writes its results to `code/outputs/<job>.npz`
   and the final weights of its network to `code/outputs/weights/<job>.pt`.

   ```bash
   python3 code/pinn_mkdv.py --list                          # job names
   python3 code/pinn_mkdv.py --jobs all --workers 4          # all 45 jobs
   python3 code/pinn_mkdv.py --jobs single_s0 amp_a2.0_s0    # selected jobs
   python3 code/pinn_mkdv.py --jobs all --skip-existing      # resume a run
   python3 code/pinn_mkdv.py --jobs all --quick              # short trial run
   ```

   Each job uses one CPU thread, and a single soliton job takes roughly twenty
   minutes. Jobs save a checkpoint every 500 Adam iterations and continue
   exactly from it after an interruption. All randomness is seeded, so a rerun
   with the versions in `requirements.txt` on the same machine gives the same
   results; on other hardware the last digits may differ. The quick mode
   shortens every job and writes to `code/outputs_quick/`.

2. Make the figures and tables once all jobs have finished:

   ```bash
   python3 code/figures.py           # reads code/outputs
   python3 code/figures.py --quick   # reads code/outputs_quick
   ```

   This writes the figures to `figures/` (`.png` and `.pdf`, 300 dpi) and, to
   `tables/`, the table bodies (`tab_*.tex`) and a file of LaTeX macros with
   every number reported in the paper (`results.tex`).

## Jobs

| Jobs | Experiment in the paper |
|------|-------------------------|
| `single_s0` to `single_s4` | single soliton with amplitude 1, five initializations |
| `amp_a<a>_s<seed>` | amplitude study, amplitudes 0.5 to 3, three initializations each |
| `meq_n<n>_s<seed>` | study of the number of equation points, three initializations each |
| `breather_plain`, `breather_causal` | breather with standard and causal training |
| `param_A`, `param_B`, `param_C`, `param_C_wide` | parameterized solitons (uncertainty quantification) |
| `param_*_unscaled` | the same four cases without rescaled parameter inputs |

The amplitude study at amplitude 1 and the study of the number of equation
points at `n = 20` (the benchmark grid) reuse the single soliton runs
`single_s0` to `single_s2`.

## Using a trained network

After training, a network can be rebuilt from its weights:

```python
import sys; sys.path.insert(0, "code")
import torch, pinn_mkdv as P

model = P.load_weights("single_s0")   # code/outputs/weights/single_s0.pt
x = torch.linspace(-20.0, 20.0, 5).reshape(-1, 1)
t = torch.zeros_like(x)
with torch.no_grad():
    print(model(torch.cat([x, t], 1)))
```

The inputs are `(x, t)` for the soliton and breather jobs. For the `param_*`
jobs they are `(x, t, p_1, ..., p_m)`, with the parameters in the order of
`PARAM_CASES` in `code/pinn_mkdv.py`, each mapped affinely from its range to
[-1, 1] (raw values for the `*_unscaled` jobs).

## Citation

If you use this code, please cite the paper above. The full reference will be
added here after publication.
