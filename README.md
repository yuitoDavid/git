# 机器人训练工具库 · Robotics training toolkit

<details name="readme-language">
<summary><strong>English</strong></summary>

Reusable tools for robot control and training. Each tool has its own directory with source code, examples, and integration notes.

## About the author

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

Possible applications include speed tracking for a wheeled base, joint tracking for a locally linearized arm, and training references for legged robots with a prescribed contact schedule. For each task, supply the dynamics, costs, bounds, and reference trajectories. The usage guide sketches how to integrate the solver into Walk the PLANC.

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

Fit a model on complete trajectories, then reconstruct one missing joint group from the remaining groups and their recent motion. The group definitions come from the data. They can describe body parts on a humanoid or sets of joints on a manipulator. The method can fill gaps in reference trajectories or provide training targets when the visible joints carry enough information about the missing group.

Input uses .npz files. The motion array has shape “clips × frames × joints.” Build the groups from the target robot's joint order, and keep units, sampling rate, and coordinate conventions consistent. Split training and validation by trajectory or terrain instance.

The current NumPy ridge-regression implementation fills one body group at a time. By default it uses the current frame and two past frames. Simulation evaluation should also check joint limits, motion smoothness, foot-placement accuracy, and disturbance recovery.

### Code and sources

- The idea comes from *Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration*, Pengxiang Ding et al., 2025. Section 3.2 describes the five-part motion representation and masked completion tasks. [Paper](https://arxiv.org/html/2502.14795v1#S3.SS2)
- [scripts/body_mask.py](body-part-motion-completion/scripts/body_mask.py) and [scripts/complete_motion.py](body-part-motion-completion/scripts/complete_motion.py) were written independently for this toolkit as a continuous joint-angle baseline. The paper and author pages checked during packaging on November 17, 2026 did not provide the corresponding training code. This implementation uses temporal ridge regression. The paper's VQ codebooks and language model would require separate work.
- Walk the PLANC is one example of using completed arm references in a training reward. Its CoM, swing-foot, and phase-synchronized arm references are described in Section III. [PLANC paper](https://arxiv.org/html/2601.06286v1#S3)

### Running and validation

Run these commands from body-part-motion-completion.

~~~powershell
python -m pip install -r requirements.txt
python scripts/complete_motion.py demo
~~~

The usage guide covers real trajectory data, joint-group files, and the train and complete commands.

Synthetic-data checks on November 17, 2026 covered training, saving, loading, NaN placeholders, and held-out clip error. They also checked that visible groups stay intact and placeholder values in the missing group do not affect predictions. Evaluation on real robot trajectories and downstream control tasks is still pending.

## Technical history

This history records changes to tool functionality, algorithms, interfaces, dependencies, supported conditions, and experimental validation.

| Date | Maintainer | Change | Source and validation |
|---|---|---|---|
| 2026-07-28 | yuitoDavid | Created the repository | Creation date retained from the original README |
| 2026-11-18 | yuitoDavid+dsh+codex | Added the πⁿMPC and body-part motion-completion packages | [Commit 1e5b920](https://github.com/yuitoDavid/git/commit/1e5b920). Validation status appears in each tool's guide |

</details>

<details open name="readme-language">
<summary><strong>简体中文</strong></summary>

收录可用于机器人控制与训练的工具。每个工具都有独立目录，包含源码、运行示例和接入说明。

## About Me

I am a student at Shanghai University majoring in Electronic Information Engineering. I enrolled in 2025.

## 工具目录

| 目录 | 用途 | 实现来源 | 使用入口 |
|---|---|---|---|
| [pi-n-mpc](pi-n-mpc/) | 批量求解有限时域 MPC，供并行仿真训练生成质心、动量或控制参考，也可测试已有 MPC 求解瓶颈 | 作者公开的 PyTorch、JAX 内核，加上本库运行示例 | [使用说明](pi-n-mpc/使用说明.md) · [SKILL.md](pi-n-mpc/SKILL.md) |
| [body-part-motion-completion](body-part-motion-completion/) | 根据其余身体部位的动作，补全缺失的一组关节参考，用于轨迹处理和训练参考实验 | 根据 Humanoid-VLA 身体分组遮挡思路编写的 NumPy 时序岭回归基线 | [使用说明](body-part-motion-completion/使用说明.md) · [SKILL.md](body-part-motion-completion/SKILL.md) |

## πⁿMPC 批量求解器

### 用途与适用条件

适用于训练回路中反复求解的线性仿射、二次代价 MPC。较多并行环境和较长预测时域能够提供更多 GPU 并行工作量，实际收益取决于求解时间占比、问题尺寸、约束和硬件。

公开内核要求单次求解共享状态转移矩阵 A，输入矩阵 B 可以随环境和预测步变化。现有约束以逐维上下界投影为主。目标任务中的时变 A、耦合摩擦约束、非线性动力学和离散落脚选择需要进一步建模或扩展。

可用于轮式底盘速度跟踪、局部线性化的机械臂关节跟踪，也可为接触时序已知的腿式机器人生成训练参考。接入时按任务提供动力学、代价、约束和参考轨迹。使用说明中以 Walk the PLANC 为例演示接入。

### 源码与来源

- [vendor/pimpc.py](pi-n-mpc/vendor/pimpc.py) 为作者的 PyTorch 实现，[vendor/jax_pimpc.py](pi-n-mpc/vendor/jax_pimpc.py) 为 JAX 实现。提取自固定提交 97e37648d19b635f9b233c7146c28e68a8d672eb，保留 [MIT 许可证](pi-n-mpc/vendor/LICENSE)。[上游固定版本](https://github.com/junhengl/mpc-rl/tree/97e37648d19b635f9b233c7146c28e68a8d672eb/src/themis_mpc)
- 方法来自 *Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation*，Junheng Li 等，2026。训练期 MPC 指导与求解器设计分别见 §II、§III。[论文全文](https://arxiv.org/html/2606.05687v1)
- [scripts/smoke_mpc.py](pi-n-mpc/scripts/smoke_mpc.py) 是本库编写的小算例，检查初态、动力学残差、输入越界量和目标函数，并输出预热后批量耗时。

### 运行与验证情况

进入 pi-n-mpc 目录，在有合适 Python 环境的机器上运行。

~~~powershell
python -m pip install -r requirements-torch.txt
python scripts/smoke_mpc.py --backend torch --device cpu
~~~

JAX 路径安装 requirements-jax-cpu.txt，运行时选择 --backend jax。CUDA、张量接口和训练接法见该目录的使用说明。

2026-11-17 整理时已核对固定版本源码并检查 Python 语法。整理环境缺少 PyTorch、JAX，求解算例与 GPU 基准待运行。迁移到新任务后的速度和策略收益以该任务的对照实测为依据。

## 身体分组遮挡与动作补全

### 用途与适用条件

用完整轨迹训练，再根据其他组的关节运动补全缺失的一组。分组由数据决定，可以按人形机器人的身体部位划分，也可以按机械臂的关节划分。当可见关节的运动对缺失组有预测价值时，可用于补全参考轨迹或生成训练目标。

数据采用 .npz 格式，motion 的维度为“片段数 × 帧数 × 关节数”。需要按目标机器人的真实关节顺序建立分组，并统一单位、采样率和坐标定义。训练与验证按轨迹或地形实例划分。

当前实现一次补全一个身体组，使用 NumPy 与岭回归，默认读取当前帧和过去两帧。仿真评估还需检查关节限位、动作平滑、落脚精度和抗扰表现。

### 源码与来源

- 思路来自 *Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration*，Pengxiang Ding 等，2025。§3.2 描述五个身体部位的动作表示和遮挡补全任务。[论文全文](https://arxiv.org/html/2502.14795v1#S3.SS2)
- [scripts/body_mask.py](body-part-motion-completion/scripts/body_mask.py) 与 [scripts/complete_motion.py](body-part-motion-completion/scripts/complete_motion.py) 是本库独立编写的连续关节角基线。2026-11-17 整理时，所检查的论文和作者页面未定位到对应训练源码。本实现的模型范围为时序岭回归，原论文的 VQ 码本与语言模型可另行研究。
- Walk the PLANC 是将补全的手臂参考用于训练奖励的一个示例。其质心、摆动脚和相位同步摆臂参考见 §III。[PLANC 论文](https://arxiv.org/html/2601.06286v1#S3)

### 运行与验证情况

进入 body-part-motion-completion 目录运行。

~~~powershell
python -m pip install -r requirements.txt
python scripts/complete_motion.py demo
~~~

真实数据训练、关节分组文件以及 train、complete 命令见该目录的使用说明。

2026-11-17 已用合成数据跑通训练、保存、加载、带 NaN 占位的补全，以及独立片段上的误差评估。还检查了可见组数据保持一致、待补组占位值不影响预测。真实机器人轨迹与下游控制任务的评估待完成。

## 技术变更记录

记录工具功能、算法实现、接口、依赖、适用条件与实验验证的变化。

| 日期 | 维护者 | 改动 | 来源与验证记录 |
|---|---|---|---|
| 2026-07-28 | yuitoDavid | 创建仓库 | 保留原 README 的创建日期 |
| 2026-11-18 | yuitoDavid+dsh+codex | 收录 πⁿMPC 与身体分组动作补全两个资料包 | [提交 1e5b920](https://github.com/yuitoDavid/git/commit/1e5b920)，验证状态见各工具说明 |

</details>
