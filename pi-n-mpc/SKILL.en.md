# πⁿMPC batched solver

[简体中文](SKILL.md) | English

Use this package to integrate, inspect, or benchmark πⁿMPC in parallel robot training when the MPC has linear-affine dynamics, quadratic costs, and supported componentwise constraints. It can also help determine whether an existing training-time MPC is a bottleneck.

Read the [usage guide](使用说明.md) ([English](USAGE.en.md)). The vendor/ directory contains the authors' public PyTorch and JAX kernels. The example checks independent dynamics and constraint errors on a bounded tracking problem with targets in both directions.

1. Establish whether the goal is to accelerate an existing MPC or add MPC references to training. For a solver replacement, keep the optimization problem fixed and measure the effect. For new references, define how they enter rewards, the critic, or distillation.
2. Check state, control, reference, coordinate, and unit conventions before integration, along with prediction step length and the interval between solves. Use the source code's tensor interface. The public version shares A, while B may vary by environment and prediction step. Check whether the constraints fit the existing bound projection.
3. Use the backend already present in the target training stack. Run the small example first and inspect dynamics residuals, bound violations, objective values, devices, and tensor shapes. Keep the new environment model separate from the solver kernel.
4. Compare the same model, contact constraints, and solution accuracy. Measure setup and first compilation, warm solves, and the full training loop separately. Report speed or policy-quality gains after measuring them in the target environment.
5. For Walk the PLANC, the original CoM reference comes from analytical splines. πⁿMPC can be tested as an alternative training reference. Begin with the original foothold targets and step timing, and use the original splines as the baseline.
