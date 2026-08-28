<!--
SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
SPDX-License-Identifier: BSD-2-Clause
-->

# Hazel cheat sheet

Account-specific: Unity ID `smohapa5`, group `hzhou`. See `hpc/README.md` for
the reasoning behind these; this file is just the commands.

## Every session

Paste this whole block after connecting. Nothing here survives a disconnect —
module loads, conda activation, and environment variables all reset.

```bash
ssh hazel                                                        # from your Mac
```

```bash
cd /share/hzhou/smohapa5/hybridlane-benchmarking
module load conda
eval "$(conda shell.bash hook)"
conda activate /usr/local/usrapps/hzhou/smohapa5/vqe-gpu
```

Confirm before doing anything that installs or submits:

```bash
which python        # must be .../vqe-gpu/bin/python, NOT /usr/bin/python3
```

If it shows the system Python, the environment is not active. Installing would
go to `~/.local` and blow the home inode quota; submitting would run the job
with the wrong interpreter.

## Submitting jobs

Always `sbatch`. Interactive `srun`/`salloc` are forced onto QOS `short_gpu`,
which only reaches fp64-crippled cards (A10/L40S).

The environment must be active **when you submit** — `--export=ALL` passes your
PATH to the job, which is how the job finds the right Python.

### The n=8 sweep (the one that works)

```bash
sbatch --partition=gpu --gres=gpu:a30:1 --ntasks=4 --mem=32G --time=02:00:00 \
  --export=ALL,JAX_ENABLE_X64=1,HYQ_DEVICE=gpu \
  --output=logs/sweep_n8_%j.out \
  --wrap="python -m cvdv_vs_dv.run_scaling_sweeps 8 --force"
```

### Larger sizes — more time, more memory

```bash
sbatch --partition=gpu --gres=gpu:a30:1 --ntasks=4 --mem=64G --time=12:00:00 \
  --export=ALL,JAX_ENABLE_X64=1,HYQ_DEVICE=gpu \
  --output=logs/sweep_n12_%j.out \
  --wrap="python -m cvdv_vs_dv.run_scaling_sweeps 12 --force"
```

### A CPU run, for an honest GPU-vs-CPU comparison

Same code, same data, `HYQ_DEVICE=cpu`. Run it *after* pulling GPU results, or
it overwrites them.

```bash
sbatch --partition=compute --ntasks=8 --mem=32G --time=08:00:00 \
  --export=ALL,JAX_ENABLE_X64=1,HYQ_DEVICE=cpu \
  --output=logs/sweep_n8_cpu_%j.out \
  --wrap="python -m cvdv_vs_dv.run_scaling_sweeps 8 --force"
```

### The 20-second environment check

Run after any dependency change, before committing to a long job.

```bash
sbatch --partition=gpu --gres=gpu:a30:1 --ntasks=1 --mem=16G --time=00:15:00 \
  --export=ALL,JAX_ENABLE_X64=1 --output=logs/smoke_%j.out \
  --wrap="python hpc/gpu_smoke_test.py"
```

### Flags that are not optional

| Flag | Why |
|---|---|
| `--gres=gpu:a30:1` | Type is mandatory and **lowercase**; untyped is rejected |
| `--export=ALL,...` | Passes your PATH so the job finds the conda env |
| `HYQ_DEVICE=gpu` | Without it, `auto` allows a silent CPU fallback |
| `python -m cvdv...` | Module form; a file path breaks package imports |
| `--force` | Sweeps skip datasets whose `.npz` already exists |
| `--output=logs/..._%j.out` | `%j` = job ID, so runs don't overwrite each other |

## Monitoring

```bash
squeue -u $USER                          # your jobs: PD pending, R running, gone = done
tail -f logs/sweep_n8_<jobid>.out        # follow live; Ctrl+C stops watching, not the job
scancel <jobid>                          # kill a job
seff <jobid>                             # elapsed time and peak memory, after it finishes
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -X | tail -5
si --gpus                                # what is free right now
```

Batch jobs survive disconnection — close the laptop, come back later.

Do not trust Slurm's `Exit code: 0:0` in the summary; `--wrap` swallows Python's
exit status. Read the log. Likewise `GPU stats: SM 0%` is sampled at job *end*,
when the GPU is idle by definition, so it does not mean the GPU went unused.

## Moving files

```bash
# From your Mac:
./hpc/hazel.sh push        # rsync working tree up (includes uncommitted work)
./hpc/hazel.sh pull        # bring cvdv_vs_dv/data/ and logs/ back

# On Hazel, if you pushed to GitHub instead:
git pull
```

`git clone`/`git pull` only carries **committed** work; `hazel.sh push` carries
everything. `/share` deletes files untouched for 30 days and is not backed up,
so pull results promptly.

**Before pulling, remember `data/*.npz` in git are CPU baselines.** Pulling
overwrites them with GPU output. Compare first if you still want the reference.

## Installing packages

Login node only — compute nodes have no internet.

```bash
python -m pip install --no-cache-dir -r hpc/requirements-hpc.txt
python -m pip install --no-cache-dir -e .
```

Never bare `pip`. If the env lacks its own pip it silently resolves to the
system one and installs into `~/.local`.

## When the home quota fills

Symptom: `Disk quota exceeded` writing even a tiny file. The limit is ~10,000
**files**, not bytes.

```bash
find ~ -xdev | wc -l                     # how many files
cd ~; find ~ -xdev -printf '%h\n' | cut -d/ -f1-4 | sort | uniq -c | sort -rn | head
rm -rf ~/.vscode-server ~/.cache ~/.local    # the usual culprits; all regenerable
```

`.vscode-server` alone was 14,678 files. Do not run VS Code Remote against a
login node.
