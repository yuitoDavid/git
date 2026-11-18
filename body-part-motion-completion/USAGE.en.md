# Body-part masking and motion completion

[简体中文](使用说明.md) | English

## Using the package in a new GPT or Gemini chat

Give the directory to a coding assistant that can read files. In a regular chat, upload this guide, SKILL.md, the code under scripts/, and the target project's joint definitions. Zip the directory if needed.

You can use this prompt.

> Read SKILL.md and the usage guide. I want to create body-part masking samples from complete robot trajectories and reconstruct missing motions. Check the target project's joint order and distinguish control actions from reference trajectories. Provide a joint-group JSON file. Run the synthetic checks first, then measure completion on independent validation trajectories. For PLANC, begin by replacing the training-time arm reference and record control metrics.

## Method and requirements

Humanoid-VLA Section 3.2 divides motion into the left leg, right leg, torso, left arm, and right arm, then pairs masked inputs with complete target motions. This package is an independently written minimal implementation of that idea. It uses continuous joint angles, temporal windows, and ridge regression to fill one group. The checked paper and author pages did not provide the corresponding training code. The package contains the continuous baseline implementation. VQ codebooks, the language model, and the robot controller would require separate implementations. [Humanoid-VLA Section 3.2](https://arxiv.org/html/2502.14795v1#S3.SS2) · [Author's publications](https://dingpx.github.io/#publications)

- Input is a NumPy .npz file with a motion array shaped (clips, frames, joints). Clips in a file have the same length. Joint angles typically use radians. Keep units, axis order, and sampling rate consistent between training and inference.
- The group JSON assigns every joint index to exactly one group. examples/toy-joints.json is a seven-dimensional example. Build real G1 indices from the simulator's joint names and order.
- Complete trajectories provide supervision. Split training and validation by trajectory, terrain, or collection batch before making windows, so overlapping adjacent windows stay within one split. Include motions that cover the target loads, gaits, and disturbances. Fixed arm-swing labels train the model to reproduce that type of motion.
- The model fills one group at a time and excludes that group's original values throughout each clip. Visible groups must contain valid values. At inference, the missing group can contain placeholders or NaN. Completion preserves visible groups. Simultaneously missing groups, joint mappings across robots, and quaternion poses need additional adaptation.
- By default the model uses the current frame and two past frames. Offline reconstruction can add future frames with --future. Causal windows repeat the first frame at the start of a clip. Outputs are kinematic references, so physical feasibility and control gains need simulation tests.

## Quick start

Run from this directory with Python 3.11 or 3.12. NumPy is the only dependency. Training and evaluation can run on CPU.

~~~powershell
python -m pip install -r requirements.txt
python scripts/complete_motion.py demo
~~~

The synthetic example uses held-out clips to check missing-group error, preservation of visible values, independence from masked placeholders, and model saving and loading. It tests the implementation. Its error does not estimate performance on G1.

For your own data, supply complete training and validation trajectories and the matching joint-group file.

~~~powershell
python scripts/complete_motion.py train --motion train.npz --groups g1-joint-groups.json --model arm-completer.npz
python scripts/complete_motion.py complete --motion validation.npz --model arm-completer.npz --part left_arm --out completed.npz --score
~~~

--score requires complete validation ground truth and reports MAE only for the reconstructed joints. Omit it when inferring from incomplete data. The output .npz has the same motion shape as the input. For Python integration, use BodyPartCompleter.fit(), complete(), save(), load(), and mask_body_part() from scripts/body_mask.py.

## PLANC integration

Export the robot's shoulder and elbow motions, both legs, torso, and references already available during training from complete simulated G1 rollouts. Make fixed-length samples at the original gait timing. Use the original footholds, step timing, and CoM supervision while generating arm targets from the available motion inputs. In a first experiment, connect the model to the training-time arm tracking reward and compare against the phase-based arm swing using the same student checkpoint and curriculum. Record foot-placement error, success on height-varying stones, and trunk angular velocity under disturbances beyond the training conditions. Observation-dropout experiments need their own dropout patterns and evaluation. [PLANC paper](https://arxiv.org/html/2601.06286v1#S3) · [Training entry point](https://github.com/Zolkin1/robot_rl/blob/mlip/scripts/rsl_rl/train_policy.py#L8-L24)

After establishing the lightweight baseline, a GRU or another temporal model can be tested under the same masking conditions, supervision, and validation tasks. Existing work on completion in physics simulation is relevant when studying coordination across groups. [MaskedMimic](https://arxiv.org/abs/2409.14393)
