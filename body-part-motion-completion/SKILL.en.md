# Body-part motion completion

[简体中文](SKILL.md) | English

Use body-part masking to reconstruct missing joints from partial motions or reference trajectories. This package supports experiments that adapt the Humanoid-VLA idea to simulation data, teacher trajectories, or PLANC.

Read the [usage guide](使用说明.md) ([English](USAGE.en.md)). The package follows the masked body-part reconstruction idea in Humanoid-VLA Section 3.2. Its code is an independently written NumPy temporal ridge-regression baseline.

1. Establish whether the task is to complete training data, reference trajectories, or observations affected by dropouts. These uses require different inputs and evaluations. This implementation handles continuous joint-angle references with one missing body group at a time.
2. Build the groups from the target project's joint names and export order. Use the same mapping, units, coordinates, and sampling rate for training and testing. Split by complete trajectory or terrain instance before making fixed-length windows.
3. Run the synthetic checks, fit on real training trajectories, and evaluate the hidden group on independent trajectories. Complete trajectories provide supervision, and completion preserves visible groups. Set the number of future frames to 0 for online use.
4. In PLANC, a first experiment can complete one arm's reference while using the existing footholds, step timing, CoM references, and student input interface. To learn more adaptable arm motion, use successful rollouts containing those adaptations as labels. Evaluate task metrics, normal-operation performance, and failures alongside reconstruction error.
