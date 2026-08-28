"""
ECD-VQE on the ``default.hybrid`` (native, JAX-jittable) simulator, as a
faster alternative to the ``bosonicqiskit.hybrid`` backend in
``ecd_vqe_sandia.py``.

Same 3-item knapsack problem, same windowed+penalized cost, same
``ECDLayer`` ansatz -- only the simulation backend and optimizer differ:

* **Backend**: ``default.hybrid`` applies each op's native Fock-matrix
  representation directly (no translation through the external
  bosonic-qiskit/Qiskit library), and supports ``interface="jax"`` with
  ``diff_method="backprop"`` -- i.e. exact autodiff gradients through the
  simulation, computed in one backward pass, instead of the
  finite-difference gradients ``scipy.optimize.approx_fprime`` needs (which
  costs ``8*N_DEPTH + 1`` extra simulations per gradient). Combined with
  ``jax.jit``, this is the source of the expected speedup.

* **Optimizer**: ``optax.adam`` with exact gradients, instead of
  ``scipy.optimize.minimize(method="BFGS")`` with finite-difference
  gradients.

**Wire-ordering gotcha (important, not a physics bug)**: ``default.hybrid``
returns ``hqml.state()`` as a plain row-major (C-order) reshape of the
per-wire dimensions in wire declaration order -- i.e. the *first* wire
(``q``) is the *most significant* index. ``bosonicqiskit.hybrid`` (via
Qiskit's own statevector convention) is the opposite: the first wire is
*least* significant. Verified directly: for the same ``ECDLayer`` circuit
and parameters, the two devices produce bit-for-bit identical *sorted*
probability distributions -- same physics -- just addressed by different
flat-index conventions. :func:`build_cost_diagonal_jax` and
:func:`decode_top_state_jax` use the big-endian convention to match; do not
reuse ``ecd_vqe_sandia.py``'s little-endian ``_wire_strides`` with state
vectors from this module, or the cost/decoding will silently point at the
wrong basis states.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml

import hybridlane as hqml
from hybridlane import ECDLayer, random_ecd_params

import ecd_vqe_sandia as _base
from ecd_vqe_sandia import WIRES, decode_fock, knapsack_cost

# NOTE: problem-dependent constants (VALUES, WEIGHTS, MAX_WEIGHT, L_VAL,
# N_BITS_M0, N_BITS_M1, N_ITEMS, PRIMARY_LEVELS, AUX_LEVELS, N_DEPTH, H_OPT,
# and the Fock cutoffs) are deliberately NOT imported by value here -- they are
# read off ``_base`` inside the functions that use them, so
# ``ecd_vqe_sandia.set_problem(...)`` takes effect in this module too. Only
# genuinely problem-independent names (WIRES) and functions are bound at import.
#
# The cutoffs are **per mode**: an instance carrying more item bits than slack
# bits needs a larger window on m0 than on m1, and paying the larger cutoff on
# both modes would square a cost that only needs to be paid once.

jax.config.update("jax_enable_x64", True)
qml.decomposition.enable_graph()

# Problem-dependent constants are forwarded to ``ecd_vqe_sandia`` on attribute
# access (PEP 562) rather than bound at import. Two things this buys:
#   * ``sj.VALUES`` and friends keep working for callers that expect them here
#     (the comparison/timing notebooks read them off this module).
#   * They follow ``ecd_vqe_sandia.set_problem(...)`` instead of freezing at
#     whatever problem happened to be selected when this module was imported.
_FORWARDED = (
    "PROBLEM",
    "N_ITEMS",
    "VALUES",
    "WEIGHTS",
    "MAX_WEIGHT",
    "L_VAL",
    "H_OPT",
    "TARGET",
    "N_DEPTH",
    "N_BITS_M0",
    "N_BITS_M1",
    "N_VARS",
    "PRIMARY_LEVELS",
    "AUX_LEVELS",
    "MAX_FOCK_M0",
    "MAX_FOCK_M1",
    "OPT_MAX_FOCK_M0",
    "OPT_MAX_FOCK_M1",
    "OPTIMAL_PATH",
)


def __getattr__(name):
    if name in _FORWARDED:
        return getattr(_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_FORWARDED))

__all__ = [
    *_FORWARDED,
    "WIRES",
    "resolve_fock",
    "decode_fock",
    "knapsack_cost",
    "get_qnode",
    "build_cost_diagonal_jax",
    "energy_jax",
    "evaluate_jax",
    "run_sandia_vqe_jax",
    "golden_angle_init",
    "optimize_problem",
    "regenerate_all",
    "decode_top_state_jax",
]


# ---------------------------------------------------------------------------
# Simulation plumbing
# ---------------------------------------------------------------------------


def _circuit_fn(params, ndepth, wires):
    p_r = qml.math.reshape(params, (ndepth, 8))
    for d in range(ndepth):
        ECDLayer(*p_r[d], wires=list(wires))
    return hqml.state()


def resolve_fock(max_fock=None, stage="verify"):
    """Normalize a cutoff argument to a ``(m0, m1)`` pair.

    ``None`` takes the active problem's cutoffs for ``stage`` (``"verify"`` or
    ``"opt"``); an int applies the same cutoff to both modes; a pair passes
    through.
    """
    if max_fock is None:
        if stage == "opt":
            return _base.OPT_MAX_FOCK_M0, _base.OPT_MAX_FOCK_M1
        return _base.MAX_FOCK_M0, _base.MAX_FOCK_M1
    if isinstance(max_fock, (int, np.integer)):
        return int(max_fock), int(max_fock)
    f0, f1 = max_fock
    return int(f0), int(f1)


def get_qnode(max_fock=None, wires=WIRES):
    """Build a jax-interfaced, backprop-differentiable QNode on ``default.hybrid``.

    Uses ``wire_dims`` rather than a blanket ``fock_level`` so m0 and m1 can be
    truncated independently.
    """
    f0, f1 = resolve_fock(max_fock)
    dev = qml.device(
        "default.hybrid", wire_dims={wires[0]: 2, wires[1]: f0, wires[2]: f1}
    )
    return qml.QNode(_circuit_fn, dev, interface="jax", diff_method="backprop")


def build_cost_diagonal_jax(max_fock=None, wires=WIRES, penalty_coeff=5.0):
    """Windowed + penalized cost diagonal, indexed in ``default.hybrid``'s
    big-endian convention: ``idx = q * F0 * F1 + n0 * F1 + n1``.

    Mirrors ``ecd_vqe_sandia.build_cost_diagonal`` exactly in cost logic;
    only the index-to-(q, n0, n1) decoding differs. Returns (h_diag, in_window)
    as JAX arrays.
    """
    f0, f1 = resolve_fock(max_fock)
    dim = 2 * f0 * f1
    idx = jnp.arange(dim)
    q = idx // (f0 * f1)
    n0 = (idx // f1) % f0
    n1 = idx % f1

    x = [(n0 >> b) & 1 for b in range(_base.N_BITS_M0)] + [q]
    y = [(n1 >> b) & 1 for b in range(_base.N_BITS_M1)]

    value = sum(_base.VALUES[i] * x[i] for i in range(_base.N_ITEMS))
    weight = sum(_base.WEIGHTS[i] * x[i] for i in range(_base.N_ITEMS))
    slack = sum(2**b * y[b] for b in range(_base.N_BITS_M1))
    cost = -value + _base.L_VAL * (_base.MAX_WEIGHT - weight - slack) ** 2

    in_window = (n0 < _base.PRIMARY_LEVELS) & (n1 < _base.AUX_LEVELS)
    h_diag = jnp.where(in_window, cost, 0.0)

    excess0 = jnp.clip(n0 - (_base.PRIMARY_LEVELS - 1), 0, None)
    excess1 = jnp.clip(n1 - (_base.AUX_LEVELS - 1), 0, None)
    h_diag = h_diag + penalty_coeff * (excess0**2 + excess1**2)

    return h_diag, in_window


def energy_jax(params, ndepth, h_diag, qnode, wires=WIRES):
    """<H> for a single parameter vector (jax-traceable)."""
    sv = qnode(params, ndepth, wires)
    probs = jnp.real(sv * jnp.conj(sv))
    return jnp.real(jnp.dot(probs, h_diag))


def evaluate_jax(params, ndepth, h_diag, in_window, qnode, wires=WIRES):
    """Return (energy, confinement) -- the jax-backend analogue of
    ``ecd_vqe_sandia.evaluate``. ``in_window`` may be a boolean or 0/1 mask;
    a multiply-and-sum is used (rather than boolean fancy-indexing) so this
    stays jit-traceable with a static output shape."""
    sv = qnode(params, ndepth, wires)
    probs = jnp.real(sv * jnp.conj(sv))
    energy = jnp.real(jnp.dot(probs, h_diag))
    confinement = jnp.sum(probs * in_window)
    return energy, confinement


def decode_top_state_jax(probs, max_fock=None):
    """Decode the most probable basis state, given big-endian-ordered probs."""
    f0, f1 = resolve_fock(max_fock)
    best_idx = int(jnp.argmax(probs))
    q = best_idx // (f0 * f1)
    n0 = (best_idx // f1) % f0
    n1 = best_idx % f1
    return q, n0, n1, float(probs[best_idx])


# ---------------------------------------------------------------------------
# VQE runner (JAX autodiff + optax, in place of scipy BFGS + finite differences)
# ---------------------------------------------------------------------------


def run_sandia_vqe_jax(
    h_diag,
    in_window,
    max_fock=None,
    ndepth=None,
    wires=WIRES,
    n_restarts=5,
    maxiter=300,
    seed=42,
    learning_rate=1e-2,
):
    """Adam (optax) + exact backprop gradients against a windowed cost diagonal.

    Same return shape as ``ecd_vqe_sandia.run_sandia_vqe``: dict with keys
    ``params``, ``energy``, ``confinement``, ``energy_history``,
    ``confinement_history`` (from the winning restart), ``wall_time``.
    """
    ndepth = _base.N_DEPTH if ndepth is None else ndepth
    qnode = get_qnode(max_fock, wires)
    optimizer = optax.adam(learning_rate)
    in_window_mask = in_window.astype(h_diag.dtype)

    def energy_fn(params):
        return energy_jax(params, ndepth, h_diag, qnode, wires)

    @jax.jit
    def train_step(params, opt_state):
        value, grad = jax.value_and_grad(energy_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Confinement piggybacks on the same forward pass's state (via a
        # second qnode call, since value_and_grad only returns the scalar
        # energy) but is folded into this single jitted step rather than a
        # separate eager call, which is what actually dominated runtime.
        _, confinement = evaluate_jax(params, ndepth, h_diag, in_window_mask, qnode, wires)
        return params, opt_state, value, confinement

    best_params, best_energy = None, np.inf
    best_energy_hist, best_confinement_hist = [], []

    t0 = time.time()
    for restart in range(n_restarts):
        # Same physically-motivated initialization (and the same seed
        # sequence) as ecd_vqe_sandia.run_sandia_vqe, just converted to a
        # jax array -- so any difference in outcome reflects the
        # backend/optimizer, not the starting point.
        rng = np.random.default_rng(seed + restart)
        params = jnp.asarray(random_ecd_params(ndepth, rng=rng))
        opt_state = optimizer.init(params)

        e_hist, c_hist = [], []
        for _ in range(maxiter):
            params, opt_state, value, confinement = train_step(params, opt_state)
            e_hist.append(float(value))
            c_hist.append(float(confinement))

        final_energy, final_confinement = evaluate_jax(
            params, ndepth, h_diag, in_window_mask, qnode, wires
        )
        final_energy, final_confinement = float(final_energy), float(final_confinement)
        print(
            f"  restart {restart + 1}/{n_restarts}  energy={final_energy:.4f}  "
            f"confinement={final_confinement:.4f}"
        )

        if final_energy < best_energy:
            best_energy = final_energy
            best_params = params
            best_energy_hist = e_hist
            best_confinement_hist = c_hist

    wall_time = time.time() - t0
    final_energy, final_confinement = evaluate_jax(
        best_params, ndepth, h_diag, in_window_mask, qnode, wires
    )
    final_energy, final_confinement = float(final_energy), float(final_confinement)

    return {
        "params": np.asarray(best_params).flatten(),
        "energy": final_energy,
        "confinement": final_confinement,
        "energy_history": best_energy_hist,
        "confinement_history": best_confinement_hist,
        "wall_time": wall_time,
    }


# ---------------------------------------------------------------------------
# Deterministic optimizer -- the recipe that actually works here
# ---------------------------------------------------------------------------

# Golden angle. Successive multiples of it are maximally spread on the circle,
# which is what makes the per-layer phase schedule below asymmetric without
# using an RNG.
_GOLDEN_ANGLE = 2.399963229728653


def golden_angle_init(ndepth, beta=0.8):
    """Deterministic (RNG-free) initial parameters for ``ndepth`` ECD layers.

    Why not simply start at ``params = 0``? That is the identity circuit, which
    leaves the state in the vacuum -- and it is an exact stationary point: the
    gradient there is *identically zero* (verified numerically, ``|grad| = 0``
    to machine precision at every depth), so an optimizer started there can
    never move. Some symmetry breaking is required.

    Why not a *symmetric* near-identity start (all phases zero, ``beta`` small)?
    That does have non-zero gradient, but it confines every displacement to a
    single quadrature and every qubit rotation to a single axis. Empirically it
    stalls badly -- E = +6.36 on knapsack4a where the schedule below reaches
    -8.23.

    So: fixed displacement magnitude ``beta`` on both modes, with the layer
    index advancing each phase by an irrational multiple of 2*pi. Fully
    reproducible, no RNG, and asymmetric enough to escape the symmetric basin.
    Sweep ``beta`` over a few values and keep the best -- see
    :func:`optimize_problem`.

    Args:
        ndepth: Number of :class:`~hybridlane.ECDLayer` layers.
        beta: Initial displacement magnitude for both qumodes.

    Returns:
        Flat array of shape ``(8 * ndepth,)``, matching ``ECDLayer``'s
        ``beta1_mag, beta1_arg, theta1, phi1, beta2_mag, beta2_arg, theta2, phi2``.
    """
    tau = 2 * np.pi
    return np.array(
        [
            [
                beta, (k * _GOLDEN_ANGLE) % tau, np.pi / 3, (k * 1.2) % tau,
                beta, (k * _GOLDEN_ANGLE + 1.0) % tau, np.pi / 3, (k * 0.7) % tau,
            ]
            for k in range(ndepth)
        ]
    ).flatten()


def optimize_problem(
    ndepth=None,
    opt_max_fock=None,
    verify_max_fock=None,
    betas=(0.6, 0.8, 1.0),
    n_iters=8000,
    penalty_coeff=5.0,
    wires=WIRES,
    verbose=True,
):
    """Re-derive optimal parameters for the **currently selected** problem.

    Works for any instance -- call ``ecd_vqe_sandia.set_problem(name)`` first,
    or use :func:`regenerate_all` to sweep every one of them.

    Deterministic end to end: no RNG anywhere, so repeated runs reproduce the
    same parameters bit for bit. The only search is a small sweep over the
    initial displacement magnitude ``betas``, keeping the lowest-energy result.

    Optimization runs at ``opt_max_fock`` (default: the active problem's
    ``OPT_MAX_FOCK_M0/M1``, i.e. 32 for the original instances) and the returned
    metrics are evaluated at ``verify_max_fock`` (default: the problem's
    ``MAX_FOCK_M0/M1``, i.e. 64 for the original instances). This
    is safe and was checked explicitly: for the converged parameters the two
    cutoffs agree to machine precision (dE ~ 4e-15, dConfinement ~ 3e-16), and
    128 is identical again -- occupation above n = 32 is ~1e-19. Optimizing at
    the smaller cutoff is ~4x cheaper per step.

    Returns:
        dict with ``params``, ``energy``, ``confinement``,
        ``energy_history``, ``confinement_history`` (per-iteration traces from
        the winning seed, for the convergence plots), ``p_optimal``,
        ``top_state``, ``beta`` and ``ndepth``.
    """
    ndepth = _base.N_DEPTH if ndepth is None else ndepth
    opt_max_fock = resolve_fock(opt_max_fock, stage="opt")
    verify_max_fock = resolve_fock(verify_max_fock)

    h_opt, iw_opt = build_cost_diagonal_jax(opt_max_fock, wires, penalty_coeff)
    h_ver, iw_ver = build_cost_diagonal_jax(verify_max_fock, wires, penalty_coeff)
    qn_opt, qn_ver = get_qnode(opt_max_fock, wires), get_qnode(verify_max_fock, wires)
    iw_opt_mask = iw_opt.astype(h_opt.dtype)

    # Decaying step size: 0.02 -> 0.005 -> 0.001 across the run.
    schedule = optax.piecewise_constant_schedule(
        0.02, {int(n_iters * 0.5): 0.25, int(n_iters * 0.82): 0.2}
    )
    optimizer = optax.adam(schedule)

    def energy_fn(params):
        return energy_jax(params, ndepth, h_opt, qn_opt, wires)

    def energy_and_confinement(params):
        """(energy, confinement) from a *single* forward pass.

        Returned as ``(value, aux)`` so ``value_and_grad(..., has_aux=True)``
        differentiates only the energy while carrying the confinement along for
        free -- tracking it does not cost a second simulation.
        """
        sv = qn_opt(params, ndepth, wires)
        probs = jnp.real(sv * jnp.conj(sv))
        return jnp.real(jnp.dot(probs, h_opt)), jnp.sum(probs * iw_opt_mask)

    def _scan_body(carry, _):
        params, opt_state = carry
        (energy, confinement), grad = jax.value_and_grad(
            energy_and_confinement, has_aux=True
        )(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        return (optax.apply_updates(params, updates), opt_state), (energy, confinement)

    # The whole inner loop runs on-device as one scan: no Python-level loop and
    # no per-step host sync, which is also what makes recording the full
    # histories essentially free.
    @jax.jit  # compiled once, reused for every beta -- recompiling per seed dominates otherwise
    def run_seed(params):
        opt_state = optimizer.init(params)
        (params, _), (e_hist, c_hist) = jax.lax.scan(
            _scan_body, (params, opt_state), None, length=n_iters
        )
        return params, e_hist, c_hist

    t0 = time.time()
    best_params, best_energy, best_beta = None, np.inf, None
    best_e_hist, best_c_hist = None, None
    for beta in betas:
        params, e_hist, c_hist = run_seed(jnp.asarray(golden_angle_init(ndepth, beta)))
        e = float(energy_fn(params))
        if verbose:
            print(f"  beta={beta}: E={e:.4f}", flush=True)
        if e < best_energy:
            best_params, best_energy, best_beta = np.asarray(params), e, beta
            best_e_hist = np.asarray(e_hist)
            best_c_hist = np.asarray(c_hist)

    probs = np.abs(np.asarray(qn_ver(best_params, ndepth, wires))) ** 2
    energy = float(np.dot(probs, np.asarray(h_ver)))
    confinement = float(np.sum(probs * np.asarray(iw_ver).astype(float)))
    tq, tm0, tm1 = _base.TARGET
    vf0, vf1 = verify_max_fock
    p_optimal = float(probs[tq * vf0 * vf1 + tm0 * vf1 + tm1])
    q, n0, n1, _ = decode_top_state_jax(probs, verify_max_fock)

    if verbose:
        print(
            f"  -> E={energy:.4f} (H_opt={_base.H_OPT})  P(optimal)={p_optimal:.4f}  "
            f"top=|q={q},m0={n0},m1={n1}>  confinement={confinement:.4f}  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    return {
        "params": best_params,
        "energy": energy,
        "confinement": confinement,
        # Per-iteration traces from the winning seed, for the convergence plots.
        # Recorded at ``opt_max_fock``; the final values are re-evaluated at
        # ``verify_max_fock`` above, and the two agree to ~1e-15, so the trace
        # and the reported endpoint are consistent.
        "energy_history": best_e_hist,
        "confinement_history": best_c_hist,
        "p_optimal": p_optimal,
        "top_state": (q, n0, n1),
        "beta": best_beta,
        "ndepth": ndepth,
    }


def regenerate_all(names=None, save=True, **kwargs):
    """Re-derive and save optimal parameters for every problem in the module.

    This is the "someone wants to reproduce the parameters" entry point. Each
    instance writes its own ``ecd_vqe_sandia_optimal_<name>.npz``, so none
    clobbers another. Restores the originally-selected problem when done.

    Also runnable straight from the shell::

        python ecd_vqe_sandia_jax.py
    """
    original = _base.PROBLEM
    names = _base.problem_names() if names is None else names
    results = {}
    try:
        for name in names:
            _base.set_problem(name)
            print(f"=== {name}  (target |q={_base.TARGET[0]},m0={_base.TARGET[1]},"
                  f"m1={_base.TARGET[2]}>, depth {_base.N_DEPTH}) ===", flush=True)
            res = optimize_problem(**kwargs)
            if save:
                path = _base.save_optimal_result(res, ndepth=res["ndepth"])
                print(f"  saved -> {path}", flush=True)
            results[name] = res
    finally:
        _base.set_problem(original)
    return results


if __name__ == "__main__":
    regenerate_all()
