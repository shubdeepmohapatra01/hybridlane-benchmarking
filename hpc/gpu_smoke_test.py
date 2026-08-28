"""Confirm a Hazel GPU node is configured the way the VQE resource study needs.

Run inside a GPU allocation (see hpc/hazel.sh smoke). Checks, in order of how
badly each one silently corrupts results:

  1. JAX actually sees a GPU (not a silent CPU fallback).
  2. x64 is enabled -- without it every array is float32 and the 1e-12
     validation against the qnode cannot pass.
  3. TF32 matmuls are off -- on A100/H100 JAX may quietly compute float32
     matmuls at ~10 bits of mantissa, which looks like a correctness bug.
  4. float64 matmul throughput is sane -- L40S/A10/RTX cards run fp64 at
     1/32-1/64 rate and would be slower than a laptop.
"""

import sys

FAIL = []


def check(label, ok, detail, fatal=True):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok and fatal:
        FAIL.append(label)


print("== JAX / GPU smoke test ==")

import jax
import jax.numpy as jnp

print(f"  jax {jax.__version__}")

devices = jax.devices()
gpus = [d for d in devices if d.platform == "gpu"]
check("GPU visible", bool(gpus), f"{devices}")
if not gpus:
    print("\nNo GPU. Are you inside a --gres=gpu allocation? Is jax[cuda12] installed?")
    sys.exit(1)

print(f"  device: {gpus[0].device_kind}")

check(
    "x64 enabled",
    jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64,
    f"zeros(float64).dtype = {jnp.zeros(1, dtype=jnp.float64).dtype}"
    " (if float32, set jax_enable_x64)",
)

check(
    "complex128 preserved",
    jnp.zeros(1, dtype=jnp.complex128).dtype == jnp.complex128,
    f"{jnp.zeros(1, dtype=jnp.complex128).dtype}",
)

# TF32 shows up as ~1e-3 relative error on a float32 matmul that should be ~1e-7.
a = jnp.ones((256, 256), dtype=jnp.float32) * 1.0001
err32 = float(jnp.abs((a @ a)[0, 0] - 256 * 1.0001**2) / (256 * 1.0001**2))
check(
    "float32 matmul precision",
    err32 < 1e-5,
    f"rel err {err32:.2e} (>1e-5 means TF32 is on; set "
    'jax.config.update("jax_default_matmul_precision", "highest"))',
    fatal=False,
)

# Throughput: 4096^3 fp64 matmul. A100 ~ tenths of a second; a fp64-crippled
# card takes many seconds.
import time

n = 4096
x = jnp.asarray(jnp.ones((n, n), dtype=jnp.float64))
jax.block_until_ready(x @ x)  # warm up / compile
t0 = time.perf_counter()
jax.block_until_ready(x @ x)
dt = time.perf_counter() - t0
tflops = 2 * n**3 / dt / 1e12
check(
    "float64 throughput",
    tflops > 2.0,
    f"{tflops:.1f} TFLOP/s ({dt*1000:.0f} ms) -- under ~2 means an fp64-crippled card",
    fatal=False,
)

print()
if FAIL:
    print(f"FAILED: {', '.join(FAIL)}")
    sys.exit(1)
print("All critical checks passed.")
