# 机器人训练工具库

整理人形机器人训练中可迁移的算法源码、运行示例与接入说明。每个子文件夹是一个独立资料包，可交给新的 GPT、Gemini 或本地编程智能体继续使用。

后续智能体开始改动前，请先阅读 [AGENTS.md](AGENTS.md)。新增工具、修改接口或补充实测结果时，同步维护本页工具目录和末尾的变更记录。

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

在 Walk the PLANC 中，可研究用 MPC 预测替换训练期质心样条参考。原版采用解析步时与样条，新增 MPC 的实验应比较参考质量、踏石表现和额外计算开销。

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

输入完整机器人轨迹，训练时遮挡左腿、右腿、躯干、左臂或右臂中的一组，用其他组和时间窗口预测缺失组。可先用于补全参考数据，或在 PLANC 中研究训练期的手臂参考。

数据采用 .npz 格式，motion 的维度为“片段数 × 帧数 × 关节数”。需要按目标机器人的真实关节顺序建立分组，并统一单位、采样率和坐标定义。训练与验证按轨迹或地形实例划分。

当前实现一次补全一个身体组，使用 NumPy 与岭回归，默认读取当前帧和过去两帧。仿真评估还需检查关节限位、动作平滑、落脚精度和抗扰表现。

### 源码与来源

- 思路来自 *Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration*，Pengxiang Ding 等，2025。§3.2 描述五个身体部位的动作表示和遮挡补全任务。[论文全文](https://arxiv.org/html/2502.14795v1#S3.SS2)
- [scripts/body_mask.py](body-part-motion-completion/scripts/body_mask.py) 与 [scripts/complete_motion.py](body-part-motion-completion/scripts/complete_motion.py) 是本库独立编写的连续关节角基线。2026-11-17 整理时，所检查的论文和作者页面未定位到对应训练源码。本实现的模型范围为时序岭回归，原论文的 VQ 码本与语言模型可另行研究。
- PLANC 的接入讨论依据其质心、摆动脚和相位同步摆臂参考，见 §III。[PLANC 论文](https://arxiv.org/html/2601.06286v1#S3)

### 运行与验证情况

进入 body-part-motion-completion 目录运行。

~~~powershell
python -m pip install -r requirements.txt
python scripts/complete_motion.py demo
~~~

真实数据训练、关节分组文件以及 train、complete 命令见该目录的使用说明。

2026-11-17 已用合成数据跑通训练、保存、加载、带 NaN 占位的补全，以及独立片段上的误差评估。还检查了可见组数据保持一致、待补组占位值不影响预测。G1 与 PLANC 的机器人数据评估待完成。

## 交给下一位智能体

可将仓库链接连同下面的文字发送给新账号中的智能体。

> 请先阅读仓库根目录的 AGENTS.md、README.md，以及相关工具目录中的 SKILL.md 和使用说明。根据我的目标项目判断适用条件，复用对应源码。完成新增或修改后，同步更新 README 的用途、来源和验证情况，并在变更记录中写入本次工作。请分别说明已经运行的验证与仍待验证的结果。

普通网页对话可上传相关资料包。能够访问工作区的编程智能体可克隆仓库后直接读取这些文件。

## 变更记录

| 日期 | 维护者 | 改动 | 来源与验证记录 |
|---|---|---|---|
| 2026-07-28 | yuitoDavid | 创建仓库 | 保留原 README 的创建日期 |
| 2026-11-18 | yuitoDavid+dsh+codex | 收录 πⁿMPC 与身体分组动作补全两个资料包 | [提交 1e5b920](https://github.com/yuitoDavid/git/commit/1e5b920)，验证状态见各工具说明 |
| 2026-11-18 | yuitoDavid+dsh+codex | 补充根目录工具说明、来源和验证状态，新增 AGENTS.md 维护约定，并同步仓库最新地址 | 检查仓库内链接目标与文档差异，沿用已有代码验证结果；当前远程为 yuitoDavid/git |
| 2026-11-18 | Codex | Added an English introduction with the university, major, and year of enrollment | Owner-provided details. README diff and local links checked. |
| 2026-11-18 | yuitoDavid+dsh+codex | 根据仓库所有者要求修正截图中三条记录的维护者 | 创建记录为 yuitoDavid，两条工具资料记录为 yuitoDavid+dsh+codex。已核对文档差异。 |
