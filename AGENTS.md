# 仓库维护说明

本仓库收集可迁移的机器人训练工具。每个工具使用独立子文件夹，包含源码、适用条件、运行方法与来源信息。

## 开始工作

阅读根目录 README.md，以及当前任务相关工具中的 SKILL.md 和使用说明.md。根据实际代码核对输入输出、依赖、关节或坐标约定，再修改接入点。

README 的“工具目录”和“技术变更记录”是项目索引。新增工具或改变算法、功能、接口、依赖、上游版本、适用条件、实验验证结果时，同步更新受影响的说明和技术变更记录。

## 更新文档

- 新增工具时，在 README 工具目录添加相对链接，说明用途、实现来源和使用入口。把详细命令、接口及条件写进工具目录的使用说明。
- 修改现有工具时，同步更新受影响的用途、条件、命令或验证状态。
- README 末尾的技术变更记录收录新增工具、算法或功能变化、接口与依赖调整、实质性适用条件或上游版本变更，以及实验验证结果。填写实际日期、维护者、技术修改摘要及来源或验证信息。保留已有技术记录，用新记录说明技术修正。
- 个人介绍、维护者署名修正、排版、措辞、首页结构、协作说明、仓库地址及记录清理等非技术编辑，以 Git 提交历史留痕。只有实质技术变化进入 README 的技术变更记录。
- 来源写清论文标题、作者或上游仓库。复制上游代码时记录固定提交，保留版权和许可证。独立编写或改写的部分注明实现者与实现范围。
- 验证写清实际运行的命令、数据类型和结果。依赖或设备不足时记录具体原因。只有在目标任务实测之后，才将速度、精度或策略收益记为该任务的结果。
- 同步维护 README.md 和本文件各自的中英文部分。README.md 采用同页语言切换，各工具的使用说明.md 对应 USAGE.en.md，SKILL.md 对应 SKILL.en.md。保持代码块、命令、路径、来源链接和技能元数据一致。翻译与措辞调整通过 Git 提交历史留痕。

## 当前工具的接入要点

- pi-n-mpc 的公开内核共享 A，B 可按环境和预测步变化，现成约束以逐维上下界为主。vendor 文件对应上游固定提交，修改 vendor 时同时记录修改内容和版本依据。新的机器人模型、坐标转换和接触定义放在任务适配代码中。
- body-part-motion-completion 使用独立编写的 NumPy 时序岭回归，一次补全一组连续关节角。按目标机器人实际关节名建立分组，按轨迹或地形拆训练与验证数据。实时输入采用因果窗口时，将 future 设为 0。
- PLANC 原版通过解析模型和样条生成训练参考。加入 MPC 或学习的摆臂参考时，需要相应的原版对照与任务指标。

## 提交前

文档修改检查相对链接与命令路径。代码修改运行与本次改动相关的现有示例或验证，记录结果与未完成项。检查 git diff，确保提交内容与当前任务相符。是否推送遵循用户在当前任务中的授权。

## 文档分工与复用

README 面向工具使用者，介绍用途、来源、适用条件、运行方法和技术变更记录。智能体接续工作流程、提示词及维护约定集中在 AGENTS.md。

可将仓库链接连同下面的文字发送给新账号中的智能体。

> 请先阅读仓库根目录的 AGENTS.md、README.md，以及相关工具目录中的 SKILL.md 和使用说明。根据我的目标项目判断适用条件，复用对应源码。改动影响工具用途、来源、接口或验证情况时，同步更新相关说明。仅在出现实质技术变化时追加 README 技术变更记录，非技术编辑通过 Git 提交历史留痕。请分别说明已经运行的验证与仍待验证的结果。

普通网页对话可上传相关资料包。能够访问工作区的编程智能体可克隆仓库后直接读取这些文件。

# Repository maintenance

This repository collects reusable tools for robot training. Each tool has its own directory with source code, requirements for use, running instructions, and provenance.

## Before making changes

Read README.md and the relevant tool's SKILL.md and 使用说明.md. Check the actual input and output interfaces, dependencies, joint mapping, and coordinate conventions before changing an integration.

The README tool index and technical history are the project index. When adding a tool or changing an algorithm, functionality, interface, dependency, upstream version, supported conditions, or experimental result, update the affected documentation and technical history.

## Documentation updates

- When adding a tool, add relative links, its purpose, implementation source, and entry points to the README index. Keep detailed commands, interfaces, and requirements in that tool's usage guide.
- When changing a tool, update the affected purposes, requirements, commands, and validation status.
- The README technical history records new tools, algorithm or functionality changes, interface and dependency updates, substantive changes to supported conditions or upstream versions, and experimental validation. Include the actual date, maintainer, technical change, and source or validation information. Preserve technical history and explain later technical corrections in new entries.
- Personal introductions, maintainer-name corrections, formatting, wording, homepage structure, collaboration notes, repository addresses, and history cleanup are recorded through Git commits. Add entries to the README technical history only for substantive technical changes.
- Credit the paper, authors, and upstream repository. When copying upstream code, pin its commit and retain its copyright and license. Identify independently written or adapted parts and their implementation scope.
- Record the commands actually run, data type, and results. State which dependencies or devices prevented a check. Record task-specific speed, accuracy, or policy gains after measuring them in that task.
- Update both language sections of README.md and AGENTS.md together. README.md switches languages in place. Each tool's 使用说明.md pairs with USAGE.en.md, and SKILL.md pairs with SKILL.en.md. Keep code blocks, commands, paths, source links, and skill frontmatter intact. Translation and wording changes are recorded in Git history.

## Current integration constraints

- The public pi-n-mpc kernels share A. B can vary by environment and prediction step, and the supplied projections mainly support componentwise bounds. The vendor files correspond to a pinned upstream commit. Record changes to those files and their version basis. Keep robot models, coordinate conversions, and contact definitions in task adapters.
- body-part-motion-completion is an independently written NumPy temporal ridge-regression implementation that fills one group of continuous joint angles. Build groups from the target robot's actual joint names and split training and validation by trajectory or terrain. Set future to 0 when using causal windows online.
- PLANC generates its original training references through analytical models and splines. Experiments with MPC or learned arm references need the original baseline and task metrics.

## Before committing

For documentation changes, check relative links and command paths. For code changes, run existing examples or checks relevant to the change and report results and pending work. Review git diff so the commit matches the task. Follow the user's authorization for pushing.

## Document roles and reuse

README explains the tools, sources, supported conditions, running instructions, and technical history for tool users. Agent handoff prompts and maintenance instructions belong in AGENTS.md.

The following prompt can be supplied with the repository link in a new account.

> Read AGENTS.md, README.md, and the relevant tool's SKILL.md and usage guide. Check whether the tool fits my target project and reuse its source. Update documentation when a change affects purpose, provenance, interfaces, or validation. Add a README technical-history entry for substantive technical changes. Record nontechnical edits in Git history. State which checks ran and which results remain unverified.

In a regular web chat, upload the relevant package. Coding assistants with workspace access can clone the repository and read these files.
