"""Exercise the vendored solver on independent bounded integrators."""
import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))


def problem(batch, horizon):
    shape = (batch, 1, horizon)
    return {
        "A": np.eye(1),
        "B_s": np.ones((batch, horizon, 1, 1)),
        "e": np.zeros((batch, 1)),
        "Wy": np.eye(1),
        "Wu": np.eye(1) * 0.1,
        "Wdu": np.eye(1) * 0.01,
        "Wf": np.eye(1),
        "x0": np.linspace(-0.1, 0.1, batch).reshape(-1, 1),
        "u0": np.zeros((batch, 1)),
        "yref": np.broadcast_to(np.linspace(1.0, -1.0, batch)[:, None, None], shape).copy(),
        "uref": np.zeros(shape),
        "umin_steps": np.full(shape, -0.2),
        "umax_steps": np.full(shape, 0.2),
    }


def check(data, x, u):
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise AssertionError("Solver returned nonfinite values.")
    if x.shape != (u.shape[0], u.shape[1], u.shape[2] + 1) or u.shape != data["uref"].shape:
        raise AssertionError((x.shape, u.shape))
    expected = (
        data["A"] @ x[:, :, :-1]
        + np.einsum("bkij,bjk->bik", data["B_s"], u)
        + data["e"][:, :, None]
    )
    initial_error = float(np.max(np.abs(x[:, :, 0] - data["x0"])))
    dynamics_error = float(np.max(np.abs(x[:, :, 1:] - expected)))
    bound_violation = float(np.maximum(0, np.maximum(u - data["umax_steps"], data["umin_steps"] - u)).max())
    errors = initial_error, dynamics_error, bound_violation

    delta = np.diff(np.concatenate([data["u0"][:, :, None], u], axis=-1), axis=-1)
    cost = float(np.sum((x[:, :, 1:] - data["yref"]) ** 2 + 0.1 * u ** 2 + 0.01 * delta ** 2))
    frozen_cost = float(np.sum((data["x0"][:, :, None] - data["yref"]) ** 2))
    if max(errors) > 2e-3 or cost >= frozen_cost:
        raise AssertionError(f"errors={errors}, cost={cost}, frozen_cost={frozen_cost}")
    print(f"Initial error {initial_error:.3g}, dynamics residual {dynamics_error:.3g}, "
          f"bound violation {bound_violation:.3g}, cost {cost:.3g} vs zero-control {frozen_cost:.3g}")


def torch_call(data, device, horizon):
    import torch
    from pimpc import Model

    model = Model()
    model.setup(data["A"], data["B_s"], horizon, Wy=data["Wy"], Wu=data["Wu"],
                Wdu=data["Wdu"], Wf=data["Wf"], maxiter=500, tol=1e-10,
                device=device, dtype=torch.float64)
    args = {name: torch.as_tensor(data[name], device=device, dtype=torch.float64)
            for name in ("x0", "u0", "yref", "uref", "umin_steps", "umax_steps")}
    args["w"] = torch.as_tensor(data["e"], device=device, dtype=torch.float64)

    def solve():
        with torch.inference_mode():
            result = model.solve_batch(**args)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        return result.x, result.u

    return solve, lambda value: value.cpu().numpy()


def jax_call(data, device, horizon):
    import jax
    import jax.numpy as jnp
    from jax_pimpc import PiMPCSolver

    devices = jax.devices("gpu" if device.startswith("cuda") else "cpu")
    device_id = int(device.split(":", 1)[1]) if ":" in device else 0
    chosen = devices[device_id]
    with jax.default_device(chosen):
        model = PiMPCSolver(B=data["x0"].shape[0], N=horizon, nx=1, nu=1,
                            maxiter=500, accel=True, dtype=jnp.float32)
        args = {name: jax.device_put(value.astype(np.float32), chosen) for name, value in data.items()}

    def solve():
        x, u, _ = model.solve(args)
        x.block_until_ready()
        return x, u

    return solve, np.asarray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jax"), default="torch")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:INDEX")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch, args.horizon, args.repeats) < 1:
        parser.error("batch, horizon, and repeats must be positive.")
    if args.device != "cpu" and not (args.device == "cuda" or (
            args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("device must be cpu, cuda, or cuda:INDEX.")
    data = problem(args.batch, args.horizon)
    build = torch_call if args.backend == "torch" else jax_call
    start = perf_counter()
    solve, to_numpy = build(data, args.device, args.horizon)
    x, u = solve()
    first_call = perf_counter() - start
    check(data, to_numpy(x), to_numpy(u))
    times = []
    for _ in range(args.repeats):
        start = perf_counter()
        solve()
        times.append((perf_counter() - start) * 1e3)
    print(f"Setup and first call {first_call:.3f} s; warm median {np.median(times):.3f} ms/batch.")


if __name__ == "__main__":
    main()
