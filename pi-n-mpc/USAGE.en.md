# πⁿMPC batched solver

[简体中文](使用说明.md) | English

## Using the package in a new GPT or Gemini chat

Give the directory to a coding assistant that can read a workspace. In a regular chat, upload this guide, SKILL.md, and the source files you need. If the upload interface accepts archives, you can zip the directory. The files and commands are self-contained, so they can be used from another account.

You can use this prompt.

> Read SKILL.md and the usage guide, then inspect the target project I provide. Determine whether training repeatedly solves an MPC with a matching structure. Identify the integration files, tensor dimensions, and constraints. Run smoke_mpc.py and use the vendored backend that matches the training stack. Compare solvers on the same problem at similar residual levels, then measure the effect on the full training loop.

## Requirements

- Each environment solves an independent finite-horizon MPC with the same structure. A promising case has many GPU environments, a longer horizon, and a substantial share of training time spent on MPC.
- The public implementation shares the state-transition matrix A. The input matrix B may vary by environment and prediction step. Strong constraints across steps, general polytopes, coupled friction cones, discrete choices, nonlinear dynamics, and step-varying A need separate adaptation or extensions.
- The public JAX interface applies per-step box bounds to the controls and uses large values for state bounds. PyTorch also accepts shared state bounds. Disable its built-in precond when the batched PyTorch path uses step-varying B. Check which setup parameters each backend actually uses.
- The solves are independent across the batch. GPU launch overhead, synchronization, copies, and memory contention can affect even large batches with short horizons. JAX compilation reuse depends on tensor shapes.
- Training-reference use does not require backpropagation through the solver. Differentiable MPC, online hardware safety control, or mixed-size batches need additional numerical requirements and validation.

## Limitations

- A fixed linearization is reliable near its operating point. Measure model error when the reference covers a wider range of states or configurations. Varying A within a horizon, or adding coupled constraints such as friction cones and obstacle avoidance, requires a different formulation or solver.
- The PyTorch interface accepts dumin and dumax, while the current projection enforces componentwise bounds on states and inputs. Wdu penalizes control increments. Hard increment bounds need a separate implementation. JAX uses fixed large internal state bounds, so task-specific state limits need separate handling.
- The solver returns the result of a finite number of iterations. JAX reports a batch residual, and PyTorch also checks convergence across the batch. PyTorch's obj_val field contains the iteration residual. Check input limits, state bounds, and dynamics residuals on each returned trajectory as required by the task. Define how the controller responds when a solve fails.

## Running locally

Use a separate Python 3.11 or 3.12 environment. Install the backend you need. The framework versions in the two requirements files come from the authors' repository. Choose GPU packages that match the CUDA version, driver, and platform.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.\.venv\Scripts\python.exe scripts/smoke_mpc.py --backend torch --device cpu
~~~

On Linux or WSL, use .venv/bin/python. For JAX on CPU, install requirements-jax-cpu.txt and run python scripts/smoke_mpc.py --backend jax --device cpu. Once CUDA is configured, select --device cuda. Follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and the [PyTorch installation page](https://pytorch.org/get-started/locally/) for GPU support.

The example solves a bounded one-dimensional integrator. It checks initial-state error, dynamics residual, input-bound violation, and objective improvement over zero control. Warm timing waits for GPU completion. Use it to check the interface, then evaluate robot performance with the target task's data and hardware.

## Tensor interface

E is the number of environments, N the prediction horizon, and nx and nu the state and control dimensions. In PyTorch, ny is the tracked output dimension defined by C.

| Quantity | Shape and meaning |
|---|---|
| A | (nx, nx). State transition shared across a solve |
| B_s | (E, N, nx, nu). Input matrices by environment and prediction step |
| x0 / u0 | (E, nx) / (E, nu). Current state and previous control |
| e | JAX takes (E, nx) and broadcasts across the horizon. PyTorch uses shared e in setup and adds per-environment offsets through (E, nx) w in solve_batch |
| yref / uref | (E, ny, N) / (E, nu, N). References over the horizon. JAX tracks the state directly, so ny equals nx |
| Wy / Wu / Wdu / Wf | Quadratic weights for state or output, control, control increments, and terminal state |
| umin_steps / umax_steps | (E, nu, N). Componentwise control bounds for each environment and step |
| Returned x / u | (E, nx, N+1) / (E, nu, N). x includes the initial state. The first receding-horizon control is u[:, :, 0] |

For PyTorch, call Model.setup() and Model.solve_batch() from vendor/pimpc.py. For JAX, use PiMPCSolver(B, N, nx, nu) and solve(prob) from vendor/jax_pimpc.py. Follow the example's parameter mapping and keep persistent tensors on the training device. Put robot-specific joint ordering and contact geometry in the task adapter.

## Integrating and evaluating a new task

1. Define the state, control input, and reference trajectory. Fix the sampling interval and write the discrete dynamics. Start with a small batch to check indexing and units for the state, control, and previous control input.
2. Map costs and constraints to the tensor interface. Check whether componentwise bounds express the required limits. Problems in the same batch share A, while B may vary by environment and prediction step.
3. Choose how to use the output. In a closed loop, pass the first control input to the controller and replan from the next observation. In training, use some or all of the predicted trajectory in rewards, the critic, or distillation.

When replacing an existing MPC solver, use the same batch of real inputs. Match objectives, feasibility, and constraints before comparing warm batch time, peak VRAM, throughput, and tail latency. Check residuals and constraint violations per environment. The JAX batch residual alone cannot establish feasibility for every environment. Include dynamics construction, cross-framework tensor conversion, solves, and reward updates when timing the full training loop.

## Example tasks

- Wheeled-base speed tracking. Use speed as the state and acceleration as the input. Track a speed command at a fixed sampling interval and apply componentwise acceleration bounds.
- Robot-arm joint tracking. Linearize around an operating point with joint positions and velocities as the state. Select the control input to match the actual interface. Check input limits and model error over the intended motion range.

### Walk the PLANC

PLANC uses LIP-based analytical step timing, CoM splines, and CLF rewards. Adding MPC therefore changes how its training reference is generated. To study reference quality, start with the original next stone, foothold pose, and step duration, then add centroidal MPC under a prescribed contact schedule. Use its CoM prediction as the CLF reference. Compare foot-placement error, disturbed traversal success, and wall-clock training time against the original splines. Discrete stone selection and joint optimization of contact timing require extensions beyond the current kernel. [PLANC Section III](https://arxiv.org/html/2601.06286v1#S3)

## Source

vendor/pimpc.py, vendor/jax_pimpc.py, and vendor/LICENSE come from the authors' repository at commit 97e37648d19b635f9b233c7146c28e68a8d672eb. The extracted files were checked line by line against the public source, and the MIT license is retained. scripts/smoke_mpc.py is the independent example supplied with this package. [Pinned source](https://github.com/junhengl/mpc-rl/tree/97e37648d19b635f9b233c7146c28e68a8d672eb/src/themis_mpc) · [MPC-RL paper](https://arxiv.org/html/2606.05687v1#S3)
