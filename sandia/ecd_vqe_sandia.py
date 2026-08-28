"""
ECD-VQE at high Fock cutoff, to isolate truncation artifacts from a binary
knapsack problem encoded on 1 qubit + 2 qumodes.

**Three problems live in this module**, selected by the ``PROBLEM`` constant.
All three encode the same way: the first ``N_BITS_PER_MODE`` item variables in
binary on the primary qumode m0, the last item variable on the qubit, and the
slack variable's binary expansion on the auxiliary qumode m1. That is 3 item
bits on m0 plus 1 on the qubit for the 4-item problems -- 4 item variables
across the three wires ``ECDLayer`` is defined on (it takes exactly 1 qubit and
2 qumodes, so there is no fourth wire to put a variable on).

======================  =========  ============  ========  
problem                 target     tot. photons  P(opt)     
======================  =========  ============  ========  
knapsack3               |0,1,0>    1             ~1.0       
knapsack4a              |1,3,2>    5             0.44       
knapsack4b (active)     |1,1,1>    2             0.996      
======================  =========  ============  ======== 

*knapsack3*: values=[7,3,4], weights=[3,2,3], W_max=3,
lambda=2, optimum x=[1,0,0]. Its optimum packs the knapsack exactly (weight ==
W_max), so the optimal slack is **zero** and m1's target state is the vacuum.

*knapsack4a*: values=[2,3,1,5], weights=[1,1,4,1], W_max=5, lambda=2, optimum x=[1,1,0,1],
slack=2 -> target |q=1, m0=3, m1=2>, H_opt=-10. 

*knapsack4b*: values=[4,1,2,5], weights=[1,3,4,2], W_max=4, lambda=4. Optimum
x=[1,0,0,1] -- items 0 and 3 -- value=9, weight=3, slack=1. H_opt=-9.

**Windowed + penalized cost.** The knapsack's binary variables still only
live in the low-lying occupation numbers (the "window" of interest: Fock
0-7 per mode, since each mode carries 3 bits here). :func:`build_cost_diagonal`
scores the physical QUBO cost inside that window, and adds a quadratic
confinement penalty -- ``penalty_coeff * excess^2``, where ``excess`` is the
occupation past the window edge -- outside it. This actively discourages the
optimizer from parking amplitude past the window (which would otherwise
alias to a bogus, unintended variable assignment), while staying smooth and
differentiable everywhere, unlike the hard Fock-cutoff wall this whole
exercise is trying to avoid. Confinement (in-window probability) is tracked
throughout optimization as a diagnostic.

"""

import os

import bosonic_qiskit as bq
import numpy as np
import pennylane as qml
from scipy.optimize import approx_fprime, minimize, nnls

import hybridlane as hqml
from hybridlane import ECDLayer, random_ecd_params
from hybridlane import wires as sa
from hybridlane.devices.bosonic_qiskit import gates as bq_gates
from hybridlane.devices.bosonic_qiskit.simulate import make_cv_circuit
from hybridlane.measurements import FockTruncation

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

# Three instances live here. Switch by changing ``PROBLEM`` below -- everything
# downstream (cost diagonal, decoding, saved-bundle filename) follows from it.
#
#   "knapsack3"  -- the original 3-item problem. Optimal slack 0, so m1 stays in
#                   the vacuum for every layer. Converges trivially, but does not
#                   exercise the auxiliary mode at all.
#   "knapsack4a" -- 4-item, target |q=1, m0=3, m1=2>. Aux mode excited,
#                   but the ansatz *plateaus*: P(optimal) saturates near 0.44 at
#                   depth 7 and does not improve with depth 9 / 12k iterations,
#                   nor with larger lambda (lambda=4 is strictly worse).
#   "knapsack4b" -- 4-item, target |q=1, m0=1, m1=1>. Aux mode excited *and*
#                   converges: P(optimal) = 0.996 at depth 7.
#

PROBLEM = "knapsack4b"

_PROBLEMS = {
    # name: (n_items, values, weights, W_max, lambda, bit layout, H_opt,
    #        target (q, m0, m1), depth)
    #
    # Bit layout is ``n_bits_m0`` item bits on m0 (plus one item on the qubit)
    # and ``n_bits_m1`` slack bits on m1, so the instance carries
    # ``n_bits_m0 + 1 + n_bits_m1`` binary variables. ``n_bits_per_mode=b`` is
    # accepted as shorthand for the symmetric case ``n_bits_m0 = n_bits_m1 = b``,
    # which is what the three original instances use.
    "knapsack3": dict(
        n_items=3, values=[7, 3, 4], weights=[3, 2, 3], max_weight=3, l_val=2,
        n_bits_per_mode=2, h_opt=-7.0, target=(0, 1, 0), n_depth=4,
        max_fock=(64, 64),
    ),
    "knapsack4a": dict(
        n_items=4, values=[2, 3, 1, 5], weights=[1, 1, 4, 1], max_weight=5, l_val=2,
        n_bits_per_mode=3, h_opt=-10.0, target=(1, 3, 2), n_depth=7,
        max_fock=(64, 64),
    ),
    "knapsack4b": dict(
        n_items=4, values=[4, 1, 2, 5], weights=[1, 3, 4, 2], max_weight=4, l_val=4,
        n_bits_per_mode=3, h_opt=-9.0, target=(1, 1, 1), n_depth=7,
        max_fock=(64, 64),
    ),
    # ---- the scaling series -------------------------------------------
    # Four instances at 4/8/12/16 binary variables, for
    # cvdv_vs_dv/vqe_resource_comparison.ipynb's scaling study. Derived by
    # cvdv_vs_dv/knapsack_scaling.py (deterministic, seed 0) rather than
    # hand-picked, so that size is the only thing that varies across them:
    # each has a unique optimum, a near-full knapsack, and an optimum whose
    # Fock numbers stay small (m0 <= 7, slack <= 3) so the drive amplitude
    # does not grow with the register. ``n_depth`` is the knee of each one's
    # own depth sweep, not a shared default -- see the notebook's section 2.
    "knapsack_n4": dict(
        n_items=2, values=[8, 9], weights=[1, 1],
        max_weight=3, l_val=10,
        n_bits_m0=1, n_bits_m1=2,
        h_opt=-17.0, target=(1, 1, 1), n_depth=7,
        max_fock=(16, 32),
    ),
    # gap=8.0  packed=2/3  target=(1, 1, 1)  n_vars=4
    "knapsack_n8": dict(
        n_items=5, values=[9, 9, 1, 1, 9], weights=[1, 3, 2, 3, 2],
        max_weight=7, l_val=10,
        n_bits_m0=4, n_bits_m1=3,
        h_opt=-27.0, target=(1, 3, 1), n_depth=7,
        max_fock=(128, 64),
    ),
    # gap=8.0  packed=6/7  target=(1, 3, 1)  n_vars=8
    "knapsack_n12": dict(
        n_items=7, values=[9, 9, 3, 2, 7, 2, 9], weights=[10, 9, 8, 5, 13, 13, 11],
        max_weight=31, l_val=10,
        n_bits_m0=6, n_bits_m1=5,
        h_opt=-27.0, target=(1, 3, 1), n_depth=7,
        max_fock=(128, 64),
    ),
    # gap=8.0  packed=28/31  target=(1, 7, 3)  n_vars=12
    # Alternative 12-variable layout: the same register size split 8 item bits
    # + 1 qubit + 3 slack bits, instead of 6 + 1 + 5. Same low target Fock state,
    # but the slack window is 8 levels rather than 32. ``knapsack_n12``'s random
    # starts fail by parking m1 at <n> ~ 5.6 against a target of 1, so this
    # tests whether the difficulty at 12 variables is the register size or the
    # width of the slack mode specifically. It is also the better-shaped
    # knapsack: 9 items against a capacity of 7, rather than 7 against 31.
    "knapsack_n12b": dict(
        n_items=9, values=[9, 9, 5, 3, 2, 1, 1, 2, 8],
        weights=[2, 3, 3, 3, 2, 2, 3, 3, 1],
        max_weight=7, l_val=10,
        n_bits_m0=8, n_bits_m1=3,
        h_opt=-26.0, target=(1, 3, 1), n_depth=16,
        max_fock=(512, 32),
    ),
    "knapsack_n16": dict(
        n_items=9, values=[8, 7, 1, 2, 1, 7, 4, 2, 8], weights=[41, 45, 46, 62, 13, 54, 54, 61, 40],
        max_weight=127, l_val=9,
        n_bits_m0=8, n_bits_m1=7,
        h_opt=-23.0, target=(1, 3, 1), n_depth=7,
        max_fock=(512, 256), opt_max_fock=(320, 160),
    ),
    # gap=7.0  packed=126/127  target=(1, 7, 1)  n_vars=16
}

WIRES = ("q", "m0", "m1")          # (qubit, primary qumode, auxiliary qumode)
DEFAULT_HEADROOM = 8               # simulation cutoff / encoding window, per mode


def set_problem(name):
    """Switch the active problem at runtime and rebind every derived constant.

    Everything downstream (:func:`knapsack_cost`, :func:`decode_fock`,
    :func:`build_cost_diagonal`, :data:`OPTIMAL_PATH`) reads these globals, so
    this is all that is needed to re-run the whole pipeline against a different
    instance -- including regenerating its optimal parameters::

        for name in ecd_vqe_sandia.problem_names():
            ecd_vqe_sandia.set_problem(name)
            ...  # optimize and save; each writes its own OPTIMAL_PATH

    Note that ``ecd_vqe_sandia_jax`` reads these attributes off this module at
    call time, so it follows the switch automatically.
    """
    global PROBLEM, N_ITEMS, VALUES, WEIGHTS, MAX_WEIGHT, L_VAL, H_OPT, TARGET
    global N_DEPTH, N_BITS_M0, N_BITS_M1, PRIMARY_LEVELS, AUX_LEVELS, OPTIMAL_PATH
    global N_BITS_PER_MODE, MAX_FOCK_M0, MAX_FOCK_M1, MAX_FOCK, N_VARS
    if name not in _PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; choose from {sorted(_PROBLEMS)}")
    p = _PROBLEMS[name]
    PROBLEM = name
    N_ITEMS = p["n_items"]
    VALUES = p["values"]
    WEIGHTS = p["weights"]
    MAX_WEIGHT = p["max_weight"]
    L_VAL = p["l_val"]
    H_OPT = p["h_opt"]
    TARGET = p["target"]           # (qubit, m0 Fock, m1 Fock) at the optimum
    N_DEPTH = p["n_depth"]         # ansatz depth
    # Bit layout. The first N_BITS_M0 items live on m0, the last item lives on
    # the qubit, and the slack is binary-expanded over N_BITS_M1 bits on m1.
    # The two counts are independent: n_vars = N_BITS_M0 + 1 + N_BITS_M1 reaches
    # every register size, where the symmetric 2b+1 layout only reaches odd ones.
    if "n_bits_per_mode" in p:
        N_BITS_M0 = N_BITS_M1 = p["n_bits_per_mode"]
    else:
        N_BITS_M0 = p["n_bits_m0"]
        N_BITS_M1 = p["n_bits_m1"]
    if N_ITEMS != N_BITS_M0 + 1:
        raise ValueError(
            f"{name}: n_items={N_ITEMS} but the layout carries {N_BITS_M0} item "
            f"bits on m0 plus 1 on the qubit"
        )
    PRIMARY_LEVELS = 2**N_BITS_M0         # window on m0: the m0 item bits
    AUX_LEVELS = 2**N_BITS_M1             # window on m1: the slack bits
    N_VARS = N_BITS_M0 + 1 + N_BITS_M1
    # Only meaningful when the two modes carry the same number of bits. Left
    # undefined otherwise, so that code still assuming the symmetric layout
    # fails loudly instead of silently reading m0's width as m1's.
    if N_BITS_M0 == N_BITS_M1:
        N_BITS_PER_MODE = N_BITS_M0
    else:
        globals().pop("N_BITS_PER_MODE", None)
    # Simulation cutoffs, per mode. Defaulting to DEFAULT_HEADROOM x the window
    # leaves room above it for the confinement penalty to be measurable.
    mf = p.get("max_fock")
    if mf is None:
        MAX_FOCK_M0 = DEFAULT_HEADROOM * PRIMARY_LEVELS
        MAX_FOCK_M1 = DEFAULT_HEADROOM * AUX_LEVELS
    elif isinstance(mf, int):
        MAX_FOCK_M0 = MAX_FOCK_M1 = mf
    else:
        MAX_FOCK_M0, MAX_FOCK_M1 = mf
    # Back-compat scalar: the larger of the two, i.e. a cutoff safe for both.
    MAX_FOCK = max(MAX_FOCK_M0, MAX_FOCK_M1)
    # Cheaper cutoffs for the optimization stage; the result is re-evaluated at
    # the verification cutoffs above. 4x the window, floored at 32 so the small
    # instances keep the cutoff their shipped parameters were optimized at, and
    # capped at the verification cutoff so it is never the looser of the two.
    global OPT_MAX_FOCK_M0, OPT_MAX_FOCK_M1
    omf = p.get("opt_max_fock")
    if omf is None:
        OPT_MAX_FOCK_M0 = min(MAX_FOCK_M0, max(32, 4 * PRIMARY_LEVELS))
        OPT_MAX_FOCK_M1 = min(MAX_FOCK_M1, max(32, 4 * AUX_LEVELS))
    elif isinstance(omf, int):
        OPT_MAX_FOCK_M0 = OPT_MAX_FOCK_M1 = omf
    else:
        OPT_MAX_FOCK_M0, OPT_MAX_FOCK_M1 = omf
    if OPT_MAX_FOCK_M0 < PRIMARY_LEVELS or OPT_MAX_FOCK_M1 < AUX_LEVELS:
        raise ValueError(
            f"{name}: optimization cutoff ({OPT_MAX_FOCK_M0}, {OPT_MAX_FOCK_M1}) is "
            f"below the encoding window ({PRIMARY_LEVELS}, {AUX_LEVELS}); the cost "
            "diagonal could not represent every assignment"
        )
    # Per-problem handoff bundle, so switching never overwrites another
    # problem's optimized parameters.
    OPTIMAL_PATH = f"ecd_vqe_sandia_optimal_{PROBLEM}.npz"
    return PROBLEM


def problem_names():
    """Names of every instance defined in this module."""
    return list(_PROBLEMS)


set_problem(PROBLEM)

ACCEPTED_GATES = (
    set(bq_gates.cv_gate_map)
    | set(bq_gates.dv_gate_map)
    | set(bq_gates.hybrid_gate_map)
    | set(bq_gates.misc_gate_map)
)


def decode_fock(n_primary, n_aux, x_qubit):
    """Decode (m0 Fock, m1 Fock, qubit) into knapsack variables (x, y).

    m0 carries the first ``N_BITS_M0`` item variables in binary, the qubit
    carries the last item variable, and m1 carries the slack variable's
    ``N_BITS_M1``-bit binary expansion.
    """
    x = [(n_primary >> b) & 1 for b in range(N_BITS_M0)] + [x_qubit]
    y = [(n_aux >> b) & 1 for b in range(N_BITS_M1)]
    return x, y


def knapsack_cost(x, y):
    """Classical QUBO cost: -value + lambda * (W_max - weight - slack)^2."""
    value = sum(VALUES[i] * x[i] for i in range(N_ITEMS))
    weight = sum(WEIGHTS[i] * x[i] for i in range(N_ITEMS))
    slack = sum(2**b * y[b] for b in range(N_BITS_M1))
    return -value + L_VAL * (MAX_WEIGHT - weight - slack) ** 2


# ---------------------------------------------------------------------------
# Simulation plumbing
# ---------------------------------------------------------------------------


def _circuit_fn(params, ndepth, wires):
    p_r = np.reshape(params, (ndepth, 8))
    for d in range(ndepth):
        ECDLayer(*p_r[d], wires=list(wires))
    return hqml.expval(hqml.N(wires[1]))  # placeholder measurement, unused


def get_statevector(params, ndepth, dim_sizes, wires=WIRES):
    """Run the ECD ansatz and return (statevector, wire_order).

    ``dim_sizes``: dict {wire_name: Fock_dimension}, e.g. ``{'q': 2, 'm0': 64,
    'm1': 64}`` -- allows a large simulation cutoff independent of how many
    of those levels actually encode problem variables.
    """
    tape = qml.tape.make_qscript(_circuit_fn)(params, ndepth, wires)
    [decomposed], _ = qml.transforms.decompose(
        tape, gate_set=ACCEPTED_GATES, max_expansion=10
    )
    sa_res = sa.type_check(decomposed)
    truncation = FockTruncation.all_fock_space(sa_res.wire_order, dim_sizes)
    qc, _ = make_cv_circuit(decomposed, truncation)
    state, _, _ = bq.util.simulate(qc, shots=None, return_fockcounts=False)
    return np.array(state), sa_res.wire_order


def _wire_strides(wire_order, dim_sizes):
    """Mixed-radix strides for decoding a flat statevector index per wire."""
    strides = {}
    stride = 1
    for w in wire_order:
        strides[w] = stride
        stride *= dim_sizes[w]
    return strides


def build_cost_diagonal(
    wire_order,
    dim_sizes,
    wires=WIRES,
    penalty_coeff=5.0,
):
    """Build the diagonal cost vector over the full (large-cutoff) Hilbert space.

    Inside the window of interest (Fock 0..PRIMARY_LEVELS-1 on m0, 0..AUX_LEVELS-1
    on m1), the cost is the physical knapsack QUBO cost. Outside it, the raw
    QUBO cost is dropped (a Fock number past the window would alias to some
    in-window bit pattern via ``n & (levels-1)``, which would silently credit
    leaked probability as if it were a real solution) and replaced by a
    quadratic confinement penalty that grows with distance past the window
    edge -- see the module docstring and PR discussion for the reasoning.

    Args:
        wire_order: Wire ordering of the statevector (from
            ``get_statevector``'s second return value).
        dim_sizes: dict {wire_name: Fock_dimension} used for the simulation.
        penalty_coeff: Quadratic penalty strength for occupation past the
            window edge.

    Returns:
        (h_diag, in_window): the cost array and a boolean mask over the same
        flat index space marking which basis states fall inside the window
        of interest (used to compute confinement/leakage diagnostics).
    """
    q_wire, m0_wire, m1_wire = wires
    dim = 1
    for w in wire_order:
        dim *= dim_sizes[w]
    strides = _wire_strides(wire_order, dim_sizes)

    idx = np.arange(dim)
    q = (idx // strides[q_wire]) % dim_sizes[q_wire]
    n0 = (idx // strides[m0_wire]) % dim_sizes[m0_wire]
    n1 = (idx // strides[m1_wire]) % dim_sizes[m1_wire]

    # Same decoding as decode_fock, vectorized over the flat index space.
    x = [(n0 >> b) & 1 for b in range(N_BITS_M0)] + [q]
    y = [(n1 >> b) & 1 for b in range(N_BITS_M1)]

    value = sum(VALUES[i] * x[i] for i in range(N_ITEMS))
    weight = sum(WEIGHTS[i] * x[i] for i in range(N_ITEMS))
    slack = sum(2**b * y[b] for b in range(N_BITS_M1))
    cost = -value + L_VAL * (MAX_WEIGHT - weight - slack) ** 2

    in_window = (n0 < PRIMARY_LEVELS) & (n1 < AUX_LEVELS)
    h_diag = np.where(in_window, cost, 0.0).astype(float)

    excess0 = np.clip(n0 - (PRIMARY_LEVELS - 1), 0, None)
    excess1 = np.clip(n1 - (AUX_LEVELS - 1), 0, None)
    h_diag = h_diag + penalty_coeff * (excess0**2 + excess1**2)

    return h_diag, in_window


def get_dim_sizes(max_fock=None, wires=WIRES):
    """Per-wire simulation dimensions.

    ``max_fock`` may be ``None`` (use the active problem's per-mode cutoffs),
    an int (the same cutoff on both modes), or a ``(m0, m1)`` pair.
    """
    if max_fock is None:
        f0, f1 = MAX_FOCK_M0, MAX_FOCK_M1
    elif isinstance(max_fock, int):
        f0 = f1 = max_fock
    else:
        f0, f1 = max_fock
    return {wires[0]: 2, wires[1]: f0, wires[2]: f1}


def get_wire_order(dim_sizes, wires=WIRES):
    """Wire ordering used by the statevector (structural -- independent of ndepth)."""
    _, wire_order = get_statevector(np.zeros(8), 1, dim_sizes, wires)
    return wire_order


# ---------------------------------------------------------------------------
# VQE runner
# ---------------------------------------------------------------------------


def evaluate(params, ndepth, h_diag, in_window, dim_sizes, wires=WIRES):
    """Return (energy, confinement) for a single parameter vector.

    ``confinement`` is the probability mass inside the window of interest --
    the diagnostic that tells you how much amplitude leaked past it.
    """
    sv, _ = get_statevector(params, ndepth, dim_sizes, wires)
    probs = np.abs(sv) ** 2
    return float(probs @ h_diag), float(probs[in_window].sum())


def run_sandia_vqe(
    h_diag,
    in_window,
    dim_sizes,
    ndepth=None,
    wires=WIRES,
    n_restarts=5,
    maxiter=300,
    seed=42,
    eps_grad=1e-2,
):
    """BFGS VQE (numerical gradient) against a windowed cost diagonal.

    Returns dict with keys: ``params``, ``energy``, ``confinement``,
    ``energy_history``, ``confinement_history`` (both from the winning
    restart), ``result`` (raw scipy OptimizeResult).
    """
    ndepth = N_DEPTH if ndepth is None else ndepth

    def energy_only(params):
        return evaluate(params, ndepth, h_diag, in_window, dim_sizes, wires)[0]

    def grad_fn(params):
        return approx_fprime(params, energy_only, eps_grad)

    best_result, best_energy_hist, best_confinement_hist = None, [], []

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        p0 = random_ecd_params(ndepth, rng=rng)
        e_hist, c_hist = [], []

        def cost(p, _e=e_hist, _c=c_hist):
            e, conf = evaluate(p, ndepth, h_diag, in_window, dim_sizes, wires)
            _e.append(e)
            _c.append(conf)
            return e

        res = minimize(
            cost,
            p0,
            method="BFGS",
            jac=grad_fn,
            options={"maxiter": maxiter, "gtol": 1e-3},
        )
        print(
            f"  restart {restart + 1}/{n_restarts}  energy={res.fun:.4f}  "
            f"evals={len(e_hist)}  msg={res.message}"
        )

        if best_result is None or res.fun < best_result.fun:
            best_result = res
            best_energy_hist = e_hist
            best_confinement_hist = c_hist

    final_energy, final_confinement = evaluate(
        best_result.x, ndepth, h_diag, in_window, dim_sizes, wires
    )

    return {
        "params": best_result.x,
        "energy": final_energy,
        "confinement": final_confinement,
        "energy_history": best_energy_hist,
        "confinement_history": best_confinement_hist,
        "result": best_result,
    }


def run_sandia_vqe_adam(
    h_diag,
    in_window,
    dim_sizes,
    ndepth=N_DEPTH,
    wires=WIRES,
    n_restarts=5,
    maxiter=300,
    seed=42,
    learning_rate=1e-2,
    eps_grad=1e-2,
):
    """Same bosonic-qiskit forward simulator + finite-difference gradient as
    ``run_sandia_vqe``, but optimized with ``optax.adam`` instead of scipy's
    BFGS -- holds the optimizer fixed so it can be compared directly against
    ``ecd_vqe_sandia_jax.run_sandia_vqe_jax`` (also ``optax.adam``) without
    the optimizer algorithm itself being a confound in that comparison.

    bosonic-qiskit still can't provide autodiff gradients (it's not a
    jax-traceable simulator), so this still pays the ``8*N_DEPTH + 1``
    finite-difference cost per gradient -- only the optimizer changes, not
    the gradient method.

    Same return shape as ``run_sandia_vqe`` (minus ``result``, since there's
    no scipy OptimizeResult here).
    """
    import optax

    def energy_only(params):
        return evaluate(params, ndepth, h_diag, in_window, dim_sizes, wires)[0]

    def grad_fn(params):
        return approx_fprime(params, energy_only, eps_grad)

    optimizer = optax.adam(learning_rate)

    best_params, best_energy = None, np.inf
    best_energy_hist, best_confinement_hist = [], []

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        params = random_ecd_params(ndepth, rng=rng)
        opt_state = optimizer.init(params)

        e_hist, c_hist = [], []
        for _ in range(maxiter):
            grad = grad_fn(params)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = np.asarray(optax.apply_updates(params, updates))
            e, conf = evaluate(params, ndepth, h_diag, in_window, dim_sizes, wires)
            e_hist.append(e)
            c_hist.append(conf)

        final_energy, final_confinement = evaluate(
            params, ndepth, h_diag, in_window, dim_sizes, wires
        )
        print(
            f"  restart {restart + 1}/{n_restarts}  energy={final_energy:.4f}  "
            f"confinement={final_confinement:.4f}"
        )

        if final_energy < best_energy:
            best_energy = final_energy
            best_params = params
            best_energy_hist = e_hist
            best_confinement_hist = c_hist

    final_energy, final_confinement = evaluate(
        best_params, ndepth, h_diag, in_window, dim_sizes, wires
    )

    return {
        "params": best_params,
        "energy": final_energy,
        "confinement": final_confinement,
        "energy_history": best_energy_hist,
        "confinement_history": best_confinement_hist,
    }


# ---------------------------------------------------------------------------
# Displacement budget -- what the hardware actually has to drive
# ---------------------------------------------------------------------------


def displacement_amplitudes(params, ndepth=None):
    """Per-ECD-gate physical displacement amplitudes |alpha|, split by qumode.

    **Convention (this is the part worth getting right).** ``ECDLayer`` passes
    its ``beta*_mag`` parameter straight into :class:`~hybridlane.ECD`, and
    ``ECD(a) = X . CD(a/2)`` with ``CD(alpha) = exp[sigma_z (alpha a^dag -
    alpha^* a)]``. So conditioned on the qubit the mode is displaced by
    ``+-a/2``, i.e. **the physical displacement amplitude is |beta_mag| / 2**,
    not ``beta_mag``. Verified numerically against the Fock matrix: for
    ``ECD(a)`` the vacuum-to-one element is ``sin(a/2)``.

    The absolute value matters too: nothing constrains ``beta_mag`` to be
    positive during optimization, so it comes back signed. A negative magnitude
    is just a pi phase shift (``ECD.adjoint`` negates it), and the drive
    amplitude the experiment must produce is the modulus.

    Args:
        params: Flat optimal parameter vector, shape ``(8 * ndepth,)``.
        ndepth: Number of layers; inferred from ``params`` if omitted.

    Returns:
        ``(alpha_primary, alpha_aux)`` -- two arrays of length ``ndepth``
        holding ``|alpha|`` for the m0 and m1 ECD gate of each layer.
    """
    params = np.asarray(params).flatten()
    ndepth = params.size // 8 if ndepth is None else ndepth
    layers = params.reshape(ndepth, 8)
    # layer layout: beta1_mag, beta1_arg, theta1, phi1, beta2_mag, beta2_arg, theta2, phi2
    return np.abs(layers[:, 0]) / 2.0, np.abs(layers[:, 4]) / 2.0


def displacement_stats(params, ndepth=None):
    """Summary statistics of the ECD displacement amplitudes.

    Intended as the handoff number for an experimental group: ``max`` is the
    largest single displacement the hardware must produce, and ``total`` (the
    sum of |alpha| over all gates) scales the total drive time. ``max_n_bar``
    is ``|alpha|^2``, the mean photon number of a coherent state at that
    amplitude -- a convenient sanity check against a mode's usable range.

    Returns:
        dict of dicts keyed ``"m0"``, ``"m1"``, ``"all"``.
    """
    a0, a1 = displacement_amplitudes(params, ndepth)

    def summarize(a):
        return {
            "n_gates": int(a.size),
            "max": float(np.max(a)),
            "min": float(np.min(a)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "total": float(np.sum(a)),
            "max_n_bar": float(np.max(a) ** 2),
        }

    return {"m0": summarize(a0), "m1": summarize(a1),
            "all": summarize(np.concatenate([a0, a1]))}


def format_displacement_stats(params, ndepth=None):
    """Human-readable table of :func:`displacement_stats`."""
    stats = displacement_stats(params, ndepth)
    hdr = (f"{'register':<22}{'gates':>6}{'max':>9}{'min':>9}{'mean':>9}"
           f"{'median':>9}{'std':>9}{'total':>9}")
    lines = ["ECD displacement amplitudes |alpha|  (physical, = |beta_mag|/2)",
             hdr, "-" * len(hdr)]
    labels = {"m0": "m0 (primary)", "m1": "m1 (auxiliary)", "all": "both modes"}
    for key in ("m0", "m1", "all"):
        st = stats[key]
        lines.append(
            f"{labels[key]:<22}{st['n_gates']:>6}{st['max']:>9.4f}{st['min']:>9.4f}"
            f"{st['mean']:>9.4f}{st['median']:>9.4f}{st['std']:>9.4f}{st['total']:>9.4f}"
        )
    lines.append("")
    lines.append(f"Largest single displacement: |alpha|_max = {stats['all']['max']:.4f}  "
                 f"(coherent-state n_bar = {stats['all']['max_n_bar']:.3f})")
    lines.append(f"Summed displacement over all {stats['all']['n_gates']} ECD gates: "
                 f"{stats['all']['total']:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handoff artifact: optimal parameters + everything needed to reproduce/decode
# the circuit, so an experimental group can run it once with no VQE loop.
# ---------------------------------------------------------------------------


def save_optimal_result(
    result,
    path=None,
    ndepth=None,
    wires=WIRES,
    max_fock=None,
):
    """Bundle the optimal params with everything needed to rebuild and decode
    the ansatz, so an experimental group can load this one file and run the
    circuit directly -- no VQE optimization loop required.

    ``path`` defaults to :data:`OPTIMAL_PATH`, which is per-problem, so saving
    one problem's result never clobbers another's.
    """
    path = OPTIMAL_PATH if path is None else path
    ndepth = N_DEPTH if ndepth is None else ndepth
    max_fock = (MAX_FOCK_M0, MAX_FOCK_M1) if max_fock is None else max_fock
    # Convergence traces are optional -- only present when the result came from
    # an optimizer that records them. Saving them means the notebook's
    # convergence plots still work on a *loaded* bundle, without re-running VQE.
    history = {}
    for key in ("energy_history", "confinement_history"):
        values = result.get(key)
        if values is not None and len(values) > 0:
            history[key] = np.asarray(values, dtype=float)
    np.savez(
        path,
        **history,
        params=result["params"],
        ndepth=ndepth,
        max_fock=np.atleast_1d(np.asarray(max_fock)),
        wires=np.array(wires),
        energy=result["energy"],
        confinement=result["confinement"],
        values=np.array(VALUES, dtype=float),
        weights=np.array(WEIGHTS, dtype=float),
        max_weight=float(MAX_WEIGHT),
        l_val=float(L_VAL),
        primary_levels=PRIMARY_LEVELS,
        aux_levels=AUX_LEVELS,
        problem=PROBLEM,
        target=np.array(TARGET),
        n_items=N_ITEMS,
        n_bits_m0=N_BITS_M0,
        n_bits_m1=N_BITS_M1,
        n_vars=N_VARS,
        h_opt=H_OPT,
    )
    return path


def load_optimal_result(path=None):
    """Load a bundle saved by :func:`save_optimal_result` (defaults to
    :data:`OPTIMAL_PATH` for the currently-selected ``PROBLEM``)."""
    data = np.load(OPTIMAL_PATH if path is None else path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Hardware-realistic readout: AJC (Blue-sideband) probe + NNLS reconstruction
# ---------------------------------------------------------------------------


def ajc_probe_sweep(
    optimal_params,
    ndepth,
    max_fock=None,
    n_reconstruct=12,
    n_theta=14,
    n_shots=300,
    wires=WIRES,
):
    """Read out low-lying Fock occupation of m0/m1 via AJC (Blue) probes on
    ancilla qubits, reconstructed via NNLS over the lowest ``n_reconstruct``
    Fock levels -- not the full ``max_fock`` cutoff. Reconstructing a few
    levels beyond the window of interest (default window is 4, reconstruct
    12) is the diagnostic that shows how much probability leaked past it.
    """
    q_wire, m0_wire, m1_wire = wires
    max_fock = MAX_FOCK if max_fock is None else max_fock
    dev = qml.device("bosonicqiskit.hybrid", max_fock_level=max_fock)

    def probe_circuit(theta_probe):
        p = np.reshape(optimal_params, (ndepth, 8))
        for d in range(ndepth):
            ECDLayer(*p[d], wires=list(wires))
        hqml.Blue(theta_probe, 0.0, ["anc0", m0_wire])
        hqml.Blue(theta_probe, 0.0, ["anc1", m1_wire])
        return (
            qml.expval(qml.Z("anc0")),
            qml.expval(qml.Z("anc1")),
            qml.expval(qml.Z(q_wire)),
        )

    qnode = qml.set_shots(qml.QNode(probe_circuit, dev), n_shots)

    thetas = np.linspace(0.02, 3 * np.pi, n_theta)
    pe_m0 = np.zeros(n_theta)
    pe_m1 = np.zeros(n_theta)
    pq1 = np.zeros(n_theta)

    for k, theta in enumerate(thetas):
        z0, z1, zq = qnode(theta)
        pe_m0[k] = (1 - float(z0)) / 2
        pe_m1[k] = (1 - float(z1)) / 2
        pq1[k] = (1 - float(zq)) / 2

    def build_a(th, n_max):
        n = np.arange(n_max + 1)
        return np.sin(th[:, None] * np.sqrt(n + 1)) ** 2

    a_mat = build_a(thetas, n_reconstruct - 1)
    pn_m0, _ = nnls(a_mat, pe_m0)
    pn_m0 /= pn_m0.sum()
    pn_m1, _ = nnls(a_mat, pe_m1)
    pn_m1 /= pn_m1.sum()

    return {
        "thetas": thetas,
        "pe_m0": pe_m0,
        "pe_m1": pe_m1,
        "pn_m0": pn_m0,
        "pn_m1": pn_m1,
        "p_x2_1": float(pq1.mean()),
    }


# ---------------------------------------------------------------------------
# QSCOUT native-gate export
# ---------------------------------------------------------------------------


def export_qscout_jaqal(
    optimal_params,
    ndepth,
    thetas,
    outdir="jaqal_probe_circuits_sandia",
    wires=WIRES,
    precision=5,
):
    """Compile the ECD-VQE + AJC-probe circuit to Jaqal for QSCOUT, one file
    per probe angle in ``thetas``."""
    from hybridlane.devices.sandia_qscout import to_jaqal

    # QSCOUT's fixed_decomps for the ECD->XCD mapping requires the new
    # graph-based decomposition system.
    qml.decomposition.enable_graph()

    q_wire, m0_wire, m1_wire = wires
    dev = qml.device("sandiaqscout.hybrid", optimize=True, n_qubits=3)

    def probe_circuit_qscout(theta_probe):
        p = np.reshape(optimal_params, (ndepth, 8))
        for d in range(ndepth):
            ECDLayer(*p[d], wires=list(wires))
        hqml.Blue(theta_probe, 0.0, ["anc0", m0_wire])
        hqml.Blue(theta_probe, 0.0, ["anc1", m1_wire])
        return (
            qml.sample(qml.Z("anc0")),
            qml.sample(qml.Z("anc1")),
            qml.sample(qml.Z(q_wire)),
        )

    qnode = qml.set_shots(qml.QNode(probe_circuit_qscout, dev), 1024)

    os.makedirs(outdir, exist_ok=True)
    for k, theta in enumerate(thetas):
        jaqal_str = to_jaqal(qnode, level="device", precision=precision)(theta)
        fname = os.path.join(outdir, f"probe_{k:02d}_theta{theta:.4f}.jaqal")
        with open(fname, "w") as f:
            f.write(jaqal_str)

    return qnode


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_ansatz(ndepth=N_DEPTH, wires=WIRES):
    import matplotlib.pyplot as plt

    params = np.zeros(8 * ndepth)
    hqml.draw_mpl(_circuit_fn, style="sketch")(params, ndepth, wires)
    plt.show()
