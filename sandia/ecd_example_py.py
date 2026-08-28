# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "hybridlane[qscout]",
#     "jax[cuda13]",
#     "matplotlib",
#     "optax",
# ]
#
# [tool.uv.sources]
# hybridlane = { git = "https://github.com/pnnl/hybridlane", rev = "main" }
# ///
import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: This notebook uses JAX with `jax[cuda13]` and `optax`
    """)
    return


@app.cell
def _():
    import os
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    print("CPUs:", jax.devices("cpu"))
    print("GPUs:", jax.devices("gpu"))
    return jax, jnp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ECD VQE Ansatz

    Let's start by defining the VQE circuit structure
    """)
    return


@app.cell
def _():
    import numpy as np

    import pennylane as qp
    from pennylane.typing import TensorLike
    from pennylane.wires import WiresLike, Wires
    qp.decomposition.enable_graph()

    import hybridlane as hl
    from hybridlane.devices.sandia_qscout.ops import R as RXY
    from hybridlane.devices.sandia_qscout import get_default_style

    def vqe_ansatz(params: TensorLike, wires: WiresLike):
        qubit, qumode, aux_qumode = Wires(wires)

        # params should have shape (..., n_layers, 8)
        for layer_params in hl.math.unstack(params, axis=-2):
            phi1, theta1 = layer_params[..., 0], layer_params[..., 1]
            qp.RZ(-phi1, qubit)
            qp.RX(theta1, qubit)
            qp.RZ(phi1, qubit)
            hl.ECD(layer_params[..., 2], layer_params[..., 3], wires=(qubit, qumode))

            phi2, theta2 = layer_params[..., 4], layer_params[..., 5]
            qp.RZ(-phi2, qubit)
            qp.RX(theta2, qubit)
            qp.RZ(phi2, qubit)
            hl.ECD(layer_params[..., 6], layer_params[..., 7], wires=(qubit, aux_qumode))

    def random_ecd_params(n_layers: int, seed: int | None = None, like: str | None = None):
        target_shape = (n_layers, 8)

        if like == "jax":
            import jax
            key = jax.random.key(seed)
            return jax.random.normal(key, shape=target_shape)

        rng = np.random.default_rng(seed)
        return rng.normal(size=target_shape)

    return TensorLike, hl, np, qp, random_ecd_params, vqe_ansatz


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's visualize how to instantiate the circuit
    """)
    return


@app.cell
def _(hl, random_ecd_params, vqe_ansatz):
    import matplotlib.pyplot as plt

    def _show_ecd_ansatz(n_layers):
        params = random_ecd_params(n_layers, 42)
        fig = plt.figure()
        hl.draw_mpl(vqe_ansatz, style="sketch", fig=fig)(params, wires=("q", "m1i2", "m1i3"))
        return fig

    _show_ecd_ansatz(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knapsack cost definitions

    Next we replicate the knapsack optimization functions of [https://github.com/shubdeepmohapatra01/hybridlane-benchmarking/blob/main/examples/templates/ecd_vqe_test.ipynb]().
    """)
    return


@app.cell
def _(TensorLike, hl, jax, jnp):
    # Problem parameters 
    VALUES     = jnp.array([2, 5, 7, 3])
    WEIGHTS    = jnp.array([2.5, 3, 4, 3.5])
    MAX_WEIGHT = 7
    L_VAL      = 3

    FOCK_LEVEL = 32

    @jax.jit
    def knapsack_cost(x: TensorLike, y: TensorLike):
        value = hl.math.dot(x, VALUES)
        weight = hl.math.dot(x, WEIGHTS)
        slack = hl.math.dot(y, 2 ** jnp.arange(3))
        return -value + L_VAL * (MAX_WEIGHT - weight - slack) ** 2

    @jax.jit
    def idx_to_vars(idx: int):
        shape = (2, FOCK_LEVEL, FOCK_LEVEL)
        q, m0, m1 = jnp.unravel_index(idx, shape)

        x0 = m0 & 1
        x1 = (m0 >> 1) & 1
        x2 = (m0 >> 2) & 1
        x3 = q

        y0 = m1 & 1
        y1 = (m1 >> 1) & 1
        y2 = (m1 >> 2) & 1

        return jnp.array([x0, x1, x2, x3]), jnp.array([y0, y1, y2])

    @jax.jit
    def idx_to_cost(idx: int):
        return knapsack_cost(*idx_to_vars(idx))

    idx_to_cost(10)
    return FOCK_LEVEL, idx_to_cost


@app.cell
def _(FOCK_LEVEL, TensorLike, hl, idx_to_cost, jax, jnp, qp, vqe_ansatz):
    dev = qp.device("default.hybrid", fock_level=FOCK_LEVEL)

    @jax.jit
    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def get_state(params: TensorLike):
        vqe_ansatz(params, wires=("q", "m0", "m1"))
        return hl.state()

    @jax.jit
    def energy(params: TensorLike):
        state = get_state(params)
        probs = hl.math.real(hl.math.abs(state) ** 2)
        basis_states = jnp.arange(probs.size)
        h_diag = jax.vmap(idx_to_cost)(basis_states)
        return hl.math.dot(h_diag, probs)

    # Get the vacuum energy
    energy(jnp.zeros((1, 8)))
    return dev, energy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimize

    In this cell we optimize the parameters using `optax`
    """)
    return


@app.cell
def _(energy, jax, np, random_ecd_params):
    from time import time

    import optax

    MAX_ITERATIONS = 300
    SEED = 42
    TRIALS = 5

    def optimize_params(n_layers: int):
        min_energy = np.inf
        optimal_params = None

        # adam is a lot faster for testing purposes
        optimizer = optax.adam(1e-2)
        # optimizer = optax.lbfgs(learning_rate=0.01)

        @jax.jit
        def train_step(params, opt_state):
            value, grad = jax.value_and_grad(energy)(params)
            updates, opt_state = optimizer.update(grad, opt_state)
            # updates, opt_state = optimizer.update(
            #      grad, opt_state, params, value=value, grad=grad, value_fn=energy
            #   )
            params = optax.apply_updates(params, updates)
            return params, opt_state

        @jax.jit
        def train_loop(params, opt_state, iterations: int):
            val = (params, opt_state)
            return jax.lax.fori_loop(
                0,
                iterations,
                lambda _, val: train_step(*val),
                val
            )

        print("Optimizing params...")
        for i in range(TRIALS):
            start = time()
            params = random_ecd_params(n_layers, like="jax", seed=SEED+i)
            opt_state = optimizer.init(params)

            params, opt_state = train_loop(params, opt_state, MAX_ITERATIONS)

            params = params.block_until_ready()
            elapsed = time() - start
            final_energy = energy(params)
            print(f"Trial {i} obtained energy {final_energy:.4f} in {elapsed:.3}s")

            if final_energy < min_energy:
                min_energy = final_energy
                optimal_params = params

        print(f"Best energy: {min_energy:.4f}")
        return params

    params = optimize_params(5)
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probe sweep
    """)
    return


@app.cell
def _(dev, hl, jax, jnp, params, qp, vqe_ansatz):
    import tqdm

    SHOTS = 300
    THETA_MAX = 3 * jnp.pi
    N_THETA = 50

    def probe_circuit(params, theta, phi):
        vqe_ansatz(params, wires=("q", "m0", "m1"))
        hl.Blue(theta, phi, ['anc0', 'm0'])
        hl.Blue(theta, phi, ['anc1', 'm1'])

        return tuple(hl.expval(qp.Projector([1], w)) for w in ("q", "anc0", "anc1"))

    probe_sim_qnode = qp.set_shots(qp.QNode(probe_circuit, dev, interface="jax"), SHOTS)

    @jax.jit
    def f(theta):
        probs = probe_sim_qnode(params, theta, 0)
        return jnp.array(probs)

    theta_vals = jnp.linspace(0.02, THETA_MAX, N_THETA)

    def _():
        prob_vals = []
        for t in tqdm.tqdm(theta_vals):
            probs = f(t).block_until_ready()
            prob_vals.append(probs)

        return jnp.array(prob_vals)

    probs = _()
    return N_THETA, SHOTS, probe_circuit, theta_vals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    todo: do some plotting/validation here

    ## Exporting to QSCOUT

    Now assuming our circuit params are good, let's show how to export batched circuits to QSCOUT
    """)
    return


@app.cell
def _(N_THETA, SHOTS, hl, jnp, params, probe_circuit, qp, theta_vals):
    from pennylane.transforms import convert_to_numpy_parameters

    from hybridlane.devices.sandia_qscout import to_jaqal

    qscout_dev = qp.device("sandiaqscout.hybrid", optimize=True, n_qubits=4)

    wire_map = {
        "q": 1,
        "anc0": 0,
        "anc1": 2,
        "m0": "m1i2",
        "m1": "m1i3"
    }

    # Hack because qscout isn't programmed to accept qp.Projector right now
    @qp.transform
    def erase_measurements(tape: qp.tape.QuantumScript):
        new_measurements = [hl.expval(qp.Z(mp.wires)) for mp in tape.measurements]
        new_tape = tape.copy(measurements=new_measurements)
        return (new_tape,), lambda x: x[0]

    phi_vals = jnp.zeros_like(theta_vals)
    batched_params = jnp.stack([params] * N_THETA, axis=0)

    # Here the batch_barams transform broadcasts all the inputs over a common batch dimension of N_THETA
    pipeline = (
        convert_to_numpy_parameters
        + qp.map_wires(wire_map=wire_map)
        + erase_measurements
        + qp.batch_params(all_operations=True)
    )

    qscout_qnode = pipeline(
        qp.set_shots(
            qp.QNode(probe_circuit, qscout_dev),
            SHOTS,
        )
    )
    return batched_params, phi_vals, qscout_qnode, to_jaqal


@app.cell
def _(batched_params, phi_vals, qscout_qnode, theta_vals, to_jaqal):
    # The jaqal exporting automatically synthesizes all tapes in a batch into a single program
    # with multiple subcircuit blocks

    jaqal_str = to_jaqal(qscout_qnode, level="device", precision=5)(batched_params, theta_vals, phi_vals)
    print(jaqal_str[:1500])
    print("...")
    return


if __name__ == "__main__":
    app.run()
