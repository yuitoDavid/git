# Robotics training toolkit

[简体中文](README.md) | English

A collection of reusable code, runnable examples, and integration notes for humanoid robot training. Each tool has its own directory.

## About me

I am a student at Shanghai University majoring in Electronic Information Engineering. I enrolled in 2025.

## Tools

| Directory | Purpose | Implementation source | Guides |
|---|---|---|---|
| [pi-n-mpc](pi-n-mpc/) | Solve batches of finite-horizon MPC problems to generate training references for center of mass, momentum, or control inputs, and measure existing MPC bottlenecks | The authors' public PyTorch and JAX kernels, with a local example | Usage ([English](pi-n-mpc/USAGE.en.md), [中文](pi-n-mpc/使用说明.md)) · Instructions ([English](pi-n-mpc/SKILL.en.md), [SKILL.md](pi-n-mpc/SKILL.md)) |
| [body-part-motion-completion](body-part-motion-completion/) | Reconstruct one missing group of joint references from the remaining body parts for trajectory processing and training-reference experiments | An independently written NumPy temporal ridge-regression baseline based on body-part masking in Humanoid-VLA | Usage ([English](body-part-motion-completion/USAGE.en.md), [中文](body-part-motion-completion/使用说明.md)) · Instructions ([English](body-part-motion-completion/SKILL.en.md), [SKILL.md](body-part-motion-completion/SKILL.md)) |

## πⁿMPC batched solver

### Purpose and requirements

The solver suits repeated linear-affine MPC problems with quadratic costs inside a training loop. Larger environment batches and longer prediction horizons provide more GPU work. The benefit depends on how much training time MPC consumes, the problem size, constraints, and hardware.

The public kernels share one state-transition matrix A per solve. The input matrix B may vary by environment and prediction step. The existing projections mainly enforce componentwise bounds. Time-varying A, coupled friction constraints, nonlinear dynamics, and discrete foothold selection require additional modeling or solver changes.

For Walk the PLANC, one experiment is to replace the training-time CoM spline reference with MPC predictions. The original method uses analytical step timing and splines. Evaluate reference quality, stepping performance, and the extra computation.

### Code and sources

- [vendor/pimpc.py](pi-n-mpc/vendor/pimpc.py) is the authors' PyTorch implementation, and [vendor/jax_pimpc.py](pi-n-mpc/vendor/jax_pimpc.py) is their JAX implementation. Both were extracted from commit 97e37648d19b635f9b233c7146c28e68a8d672eb. The original [MIT license](pi-n-mpc/vendor/LICENSE) is included. [Pinned upstream source](https://github.com/junhengl/mpc-rl/tree/97e37648d19b635f9b233c7146c28e68a8d672eb/src/themis_mpc)
- The method comes from *Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation*, Junheng Li et al., 2026. Sections II and III describe training-time MPC guidance and the solver. [Paper](https://arxiv.org/html/2606.05687v1)
- [scripts/smoke_mpc.py](pi-n-mpc/scripts/smoke_mpc.py) is an example written for this toolkit. It checks the initial state, dynamics residual, input-bound violations, and objective value, then reports warm batch timing.

### Running and validation

Run these commands from pi-n-mpc in a suitable Python environment.

~~~powershell
python -m pip install -r requirements-torch.txt
python scripts/smoke_mpc.py --backend torch --device cpu
~~~

For JAX, install requirements-jax-cpu.txt and select --backend jax. The usage guide covers CUDA, tensor interfaces, and training integration.

On November 17, 2026, the extracted code was checked against the pinned upstream version and the Python files passed syntax checks. The packaging environment lacked PyTorch and JAX. The solver example and GPU benchmarks still need to run. Speed and policy-quality claims for a new task require measurements on that task.

## Body-part masking and motion completion

### Purpose and requirements

Given complete robot trajectories, hide one body group, such as a leg, the torso, or an arm, and predict it from the remaining groups and a temporal window. This can support reference-data completion or experiments with training-time arm references in PLANC.

Input uses .npz files. The motion array has shape “clips × frames × joints.” Build the groups from the target robot's joint order, and keep units, sampling rate, and coordinate conventions consistent. Split training and validation by trajectory or terrain instance.

The current NumPy ridge-regression implementation fills one body group at a time. By default it uses the current frame and two past frames. Simulation evaluation should also check joint limits, motion smoothness, foot-placement accuracy, and disturbance recovery.

### Code and sources

- The idea comes from *Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration*, Pengxiang Ding et al., 2025. Section 3.2 describes the five-part motion representation and masked completion tasks. [Paper](https://arxiv.org/html/2502.14795v1#S3.SS2)
- [scripts/body_mask.py](body-part-motion-completion/scripts/body_mask.py) and [scripts/complete_motion.py](body-part-motion-completion/scripts/complete_motion.py) were written independently for this toolkit as a continuous joint-angle baseline. The paper and author pages checked during packaging on November 17, 2026 did not provide the corresponding training code. This implementation uses temporal ridge regression. The paper's VQ codebooks and language model would require separate work.
- The PLANC integration notes follow its CoM, swing-foot, and phase-synchronized arm references in Section III. [PLANC paper](https://arxiv.org/html/2601.06286v1#S3)

### Running and validation

Run these commands from body-part-motion-completion.

~~~powershell
python -m pip install -r requirements.txt
python scripts/complete_motion.py demo
~~~

The usage guide covers real trajectory data, joint-group files, and the train and complete commands.

Synthetic-data checks on November 17, 2026 covered training, saving, loading, NaN placeholders, and held-out clip error. They also checked that visible groups stay intact and placeholder values in the missing group do not affect predictions. Evaluation on G1 or PLANC robot data is still pending.

## Technical history

This history records changes to tool functionality, algorithms, interfaces, dependencies, supported conditions, and experimental validation.

| Date | Maintainer | Change | Source and validation |
|---|---|---|---|
| 2026-07-28 | yuitoDavid | Created the repository | Creation date retained from the original README |
| 2026-11-18 | yuitoDavid+dsh+codex | Added the πⁿMPC and body-part motion-completion packages | [Commit 1e5b920](https://github.com/yuitoDavid/git/commit/1e5b920). Validation status appears in each tool's guide |
