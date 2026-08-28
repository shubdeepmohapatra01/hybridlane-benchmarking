<!--
SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
SPDX-License-Identifier: BSD-2-Clause
-->

# Running the CV-DV scaling study on GPU

Everything here is specific to running `cvdv_vs_dv/` on NVIDIA hardware. The
physics is identical to the CPU path; only the execution backend differs.

## Why this directory exists

This repository is a fork of [pnnl/hybridlane](https://github.com/pnnl/hybridlane),
and the fork is kept **byte-identical to upstream in every shared file**. All of
our work lives in directories upstream has never heard of — `hpc/`,
`cvdv_vs_dv/`, `sandia/` — so `git merge upstream/main` can never conflict.

That is why the CUDA dependencies are pinned in `requirements-hpc.txt` rather
than as an optional extra in `pyproject.toml`. An extra would be tidier in
isolation, but `pyproject.toml` is an upstream file, and editing it is the one
thing that would create a merge surface. None of this is intended for upstream.

## Reproducing on any CUDA machine

Requires an NVIDIA GPU with **good float64 throughput** — see the table below.

```bash
python -m pip install -r hpc/requirements-hpc.txt
python -m pip install -e .
python hpc/gpu_smoke_test.py          # ~20 s; verifies the environment
python -m cvdv_vs_dv.run_scaling_sweeps 8 --force
```

Run sweeps as a **module** (`python -m cvdv_vs_dv.run_scaling_sweeps`), not as a
path (`python cvdv_vs_dv/run_scaling_sweeps.py`). Running the file directly puts
`cvdv_vs_dv/` on `sys.path` instead of the repository root, and the package
imports fail.

Results are written to `cvdv_vs_dv/data/*.npz` and **skipped if already
present** — pass `--force` to recompute. The committed `.npz` files are
CPU-generated baselines; keep them, they are the reference the GPU numbers are
validated against.

### The `jax<0.11` pin is load-bearing

JAX 0.11 removed `jax.core.is_concrete`, which PennyLane 0.45 still calls in
`workflow/resolution.py` on every qnode invocation. With JAX 0.11 installed,
every sweep dies with an `AttributeError` on its first optimizer step — after
importing cleanly and correctly reporting the GPU, which makes it look like a
runtime bug rather than a dependency one. Do not relax the bound without
checking PennyLane's JAX support first.

### Choosing a GPU: float64 is the only thing that matters

The propagator is dense batched float64 matrix-vector products. Consumer and
inference-oriented cards run fp64 at 1/32–1/64 rate and are **slower than a
laptop** for this workload, regardless of their headline FLOPS.

| GPU | fp64 | Verdict |
|---|---|---|
| A100, H100, H200 | full rate | Ideal |
| A30, P100 | half rate | Good — A30 measured 9.1–9.5 TFLOP/s |
| L40S, L40, A10, RTX 2080, GTX 1080 | ~1/32–1/64 | **Avoid** |

Energies are compared against cached values at 1e-15, so float32 would silently
invalidate them. `cvdv_vs_dv.backend.verify()` makes a missing GPU or disabled
x64 a hard error rather than a silent fallback — a job that quietly runs on CPU
produces correct numbers far too slowly and looks identical in the output files.

## On NC State's Hazel cluster

`hazel.sh` wraps the workflow. It assumes an `ssh hazel` alias in
`~/.ssh/config`; `ControlMaster` with `ControlPersist` is worth configuring so
Duo prompts once per session rather than per command.

```bash
./hpc/hazel.sh connect                 # shell on the login node
./hpc/hazel.sh submit <script.py>      # sbatch on an A30 (default 4 h)
./hpc/hazel.sh status                  # your jobs
./hpc/hazel.sh pull                    # results back to this repo
./hpc/hazel.sh gpus                    # what is free right now
```

### Cluster-specific gotchas

**Interactive jobs cannot reach the fp64-capable cards.** Hazel forces
`srun`/`salloc` onto QOS `short_gpu`, which is bound to the `gpu_partners`
partition — A10 and L40S only, both fp64-crippled. Batch jobs may use any QOS,
so **`sbatch` is the only route to an A30 or A100**.

**A30, not A100.** A100 is a single node with 4 cards and is effectively always
allocated; A30 offers 8 cards across four nodes with near-zero queue wait and
comparable fp64 throughput. Prefer A100 only when 24 GB is not enough.

**GPU type is mandatory and lowercase.** `--gres=gpu:a30:1`. An untyped request
is rejected; `si --gpus` displays names capitalised but Slurm wants lowercase.

**Set `HYQ_DEVICE=gpu` in batch jobs.** Without it `backend.select()` resolves
`auto`, which permits a CPU fallback — spending the whole wall-clock allocation
at CPU speed with output files that look identical.

**Install from a login node.** Compute nodes have no route to the internet.

**Environments belong in `/usr/local/usrapps`, never `/share` or `/home`.**
`/share` deletes files not accessed in 30 days, which corrupts a conda
environment one rarely-imported module at a time. `/home` has a ~10,000 file
inode quota that a single JAX+CUDA install exceeds.

**Use `python -m pip`, never bare `pip`.** If the environment lacks its own pip,
bare `pip` silently resolves to the system one and installs into `~/.local`,
which both blows the inode quota and shadows the environment afterwards.
