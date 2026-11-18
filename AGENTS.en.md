# Repository maintenance

[简体中文](AGENTS.md) | English

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
- Keep the English and Chinese documents in sync. README.md pairs with README.en.md, AGENTS.md with AGENTS.en.md, each 使用说明.md with USAGE.en.md, and each SKILL.md with SKILL.en.md. Keep code blocks, commands, paths, source links, and skill frontmatter intact. Translation and wording changes are recorded in Git history.

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
