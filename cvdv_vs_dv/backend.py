# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
"""
Pick the JAX execution backend once, before anything imports JAX.

The scaling study runs in two places. Small instances (4-8 variables) run on a
laptop CPU, which is where the demo notebooks live and where a reader will
reproduce them. The large ones (12 and 16 variables) are a different order of
cost -- one 16-variable optimizer run is ~30 CPU-minutes and a sweep is days --
so they belong on NC State's **Hazel** cluster, on a GPU node. The physics code
is identical either way; only this module differs, which is the point.

**This module must be imported before ``jax``.** JAX reads its platform choice
from the environment at import time and caches it, so a later
``jax.config.update`` cannot move work onto a GPU that JAX has already decided
does not exist. Every entry point that touches the simulation therefore starts
with::

    from cvdv_vs_dv import backend
    backend.select()          # or backend.select("gpu")

:func:`select` is idempotent and safe to call from a worker process.

**Choosing the device.** In precedence order:

1. the ``device`` argument to :func:`select`,
2. the ``HYQ_DEVICE`` environment variable (``cpu`` / ``gpu`` / ``auto``),
3. auto-detection -- GPU when JAX can see one, else CPU.

Auto-detection is the default because the same script then runs unmodified on a
laptop and under ``sbatch`` on Hazel; the Slurm script sets ``HYQ_DEVICE=gpu``
explicitly anyway, so that a job which lands on a node whose GPU is unavailable
**fails loudly** rather than silently spending its wall-clock allocation on CPU.

**Threads: measured, and mostly not controllable.** A single JAX CPU process
spreads across roughly six cores here, so running five worker processes on a
twelve-core machine oversubscribes it about 2.5x. The obvious fix is to bound
each worker's thread count -- and it **does not work**: with
``OMP_NUM_THREADS``, the MKL/OpenBLAS equivalents, and XLA's
``intra_op_parallelism_threads`` all set to 1, 2 and 6 in turn, the measured
per-step time was 49.7 / 49.5 / 49.6 ms and the process used ~16 cores' worth of
CPU time in every case. XLA's CPU backend runs its own thread pool that these
knobs do not reach.

``threads=`` is still honoured for the OMP-family variables, because those do
bind NumPy and BLAS, but do not expect it to bound XLA. The lever that actually
works on CPU is **running fewer worker processes**, and the real fix is the GPU
path below. This is recorded here so the next person does not spend an
afternoon rediscovering it.

Float64 is enabled on both backends. The knapsack energies are compared against
cached values at 1e-15 elsewhere in this repo, so dropping to float32 for GPU
speed would silently invalidate those comparisons; if that trade is ever wanted
it should be a deliberate, measured change rather than a side effect of moving
machines.
"""

from __future__ import annotations

import os
import shutil
import subprocess

#: Set once :func:`select` has run, so repeat calls are cheap and consistent.
_SELECTED: str | None = None

VALID = ("cpu", "gpu", "auto")


def _gpu_is_visible() -> bool:
    """True when this machine appears to have a usable NVIDIA GPU.

    Deliberately does **not** import JAX: importing it is what freezes the
    platform choice, so the detection has to happen first. ``nvidia-smi`` is
    the cheapest reliable signal, and Slurm's ``CUDA_VISIBLE_DEVICES`` is
    checked first because on a cluster it is the authoritative statement of
    what this job was actually allocated.
    """
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allocated is not None:
        return allocated.strip() not in ("", "-1", "NoDevFiles")
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def resolve(device: str | None = None) -> str:
    """Resolve the requested device to ``"cpu"`` or ``"gpu"`` without acting on it."""
    choice = (device or os.environ.get("HYQ_DEVICE") or "auto").lower()
    if choice not in VALID:
        raise ValueError(f"device must be one of {VALID}, got {choice!r}")
    if choice == "auto":
        return "gpu" if _gpu_is_visible() else "cpu"
    return choice


def select(device: str | None = None, threads: int | None = None,
           verbose: bool = True) -> str:
    """Fix the JAX backend for this process. Call before importing ``jax``.

    Args:
        device: ``"cpu"``, ``"gpu"``, ``"auto"`` or None (see module docstring).
        threads: CPU threads for NumPy/BLAS in this process. Does **not** bound
            XLA's own pool -- see the module docstring. Ignored on GPU.
        verbose: print the resolved configuration. Worth leaving on -- a sweep
            that quietly ran on the wrong backend looks identical in the output
            files and is only visible in the wall-clock time.

    Returns:
        The resolved device, ``"cpu"`` or ``"gpu"``.
    """
    global _SELECTED
    resolved = resolve(device)

    if _SELECTED is not None:
        if resolved != _SELECTED and verbose:
            print(f"[backend] already fixed to {_SELECTED!r}; ignoring "
                  f"request for {resolved!r} (JAX caches this at import)")
        return _SELECTED

    if resolved == "gpu":
        # An *explicit* gpu request must fail here rather than fall back. A
        # batch job that quietly runs on CPU produces correct numbers about a
        # hundred times too slowly, uses its whole wall-clock allocation, and
        # looks identical in the output files -- the failure is only visible in
        # the timing, long after the allocation is spent. "auto" may fall back,
        # because that is what it was asked to do.
        explicit = (device or os.environ.get("HYQ_DEVICE") or "auto").lower() == "gpu"
        if explicit and not _gpu_is_visible():
            raise RuntimeError(
                "device='gpu' was requested but no GPU is visible. On Hazel this "
                "usually means the job did not get the card it asked for: check "
                "--gres=gpu:<type>:<n> (the type is mandatory), `si --gpus` for "
                "availability, and CUDA_VISIBLE_DEVICES inside the job. Pass "
                "device='auto' to allow a CPU fallback."
            )
        # "cuda" first, with CPU retained as a fallback *device*, not a
        # fallback platform: the process still runs if a stray op has no GPU
        # kernel, but the default placement is the GPU.
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
        # Cuts peak memory: JAX otherwise preallocates ~75% of the card, which
        # collides with anything else sharing the node.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        if threads is not None:
            if threads < 1:
                raise ValueError(f"threads must be >= 1, got {threads}")
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[var] = str(threads)
            # XLA's CPU backend runs its own pool, which the OMP variables above
            # do not reach -- this is the flag that actually bounds it.
            flags = os.environ.get("XLA_FLAGS", "")
            os.environ["XLA_FLAGS"] = (
                f"{flags} --xla_cpu_multi_thread_eigen={'true' if threads > 1 else 'false'}"
                f" intra_op_parallelism_threads={threads}"
            ).strip()

    os.environ.setdefault("JAX_ENABLE_X64", "true")
    _SELECTED = resolved
    if verbose:
        detail = f", {threads} threads" if (resolved == "cpu" and threads) else ""
        print(f"[backend] {resolved}{detail}, float64 on", flush=True)
    return resolved


def describe() -> dict:
    """What JAX actually ended up with. Import-safe only *after* :func:`select`."""
    import jax

    devices = jax.devices()
    return {
        "requested": _SELECTED,
        "platform": devices[0].platform,
        "n_devices": len(devices),
        "devices": [str(d) for d in devices],
        "x64": jax.config.jax_enable_x64,
    }


def verify(strict: bool = True) -> dict:
    """Assert that JAX actually ended up where :func:`select` asked it to go.

    **This is a separate call from :func:`select` because it has to be.**
    ``select`` runs before JAX is imported -- it must, since JAX fixes its
    platform at import and caches it -- so all it can check is whether the
    *machine* has a GPU (via ``CUDA_VISIBLE_DEVICES``, else ``nvidia-smi``).
    That leaves a gap it cannot close: a node can have a perfectly good A100
    while the installed JAX has no CUDA plugin, in which case
    ``JAX_PLATFORMS="cuda,cpu"`` silently resolves to the CPU device and the job
    runs to completion at CPU speed. The only way to detect that is to look at
    ``jax.devices()`` *after* the import, which is what this does.

    Call it once, immediately after the first JAX import, in any entry point
    that is going to spend real time computing.

    Args:
        strict: raise on mismatch. Pass False to get the dict and decide
            yourself -- a notebook rendering cached results does not care.

    Raises:
        RuntimeError: a GPU was requested and JAX is not on one.
    """
    info = describe()
    if strict and info["requested"] == "gpu" and info["platform"] != "gpu":
        raise RuntimeError(
            f"a GPU was requested but JAX is running on {info['platform']!r} "
            f"(devices: {info['devices']}).\n"
            "The machine may well have a GPU -- this means *JAX* cannot use it. "
            "Two usual causes:\n"
            "  1. CUDA-enabled JAX is not installed. This project declares plain "
            "`jax<0.11`, which is CPU-only; install the extra with\n"
            "       uv sync --extra cuda\n"
            "     and confirm with `python -c \"import jax; print(jax.devices())\"`.\n"
            "  2. The job did not get the card it asked for -- check "
            "--gres=gpu:<type>:<n> (the type is mandatory on Hazel), `si --gpus`, "
            "and CUDA_VISIBLE_DEVICES inside the job.\n"
            "Left unchecked this runs to completion at CPU speed and the output "
            "files look identical, so it is made fatal here on purpose."
        )
    if strict and not info["x64"]:
        raise RuntimeError(
            "float64 is disabled. Energies in this study are compared against "
            "cached values at 1e-15, so float32 would silently invalidate them."
        )
    return info


def report() -> str:
    """One line naming the backend, for a notebook or a job log."""
    info = describe()
    line = (f"JAX on {info['platform']} -- {info['n_devices']} device(s): "
            f"{', '.join(info['devices'])}, float64={info['x64']}")
    if info["requested"] == "gpu" and info["platform"] != "gpu":
        line += ("\n  WARNING: a GPU was requested but JAX is on "
                 f"{info['platform']}. On Hazel this usually means the job did "
                 "not get the GPU it asked for -- check --gres and `si --gpus`.")
    return line
