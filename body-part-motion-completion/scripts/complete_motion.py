"""Demo, training, and completion commands for the body-part masking baseline."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from body_mask import BodyPartCompleter, mask_body_part


def motion_file(path):
    with np.load(path, allow_pickle=False) as archive:
        return archive["motion"]


def demo():
    generator = np.random.default_rng(11)
    phases = generator.uniform(0, 2 * np.pi, (38, 1))
    time = np.arange(64)[None, :] * 0.13 + phases
    hip = np.sin(time)
    knee = np.cos(time)
    motion = np.stack(
        [hip, 0.9 * knee, -hip, -0.9 * knee, 0.25 * knee,
         -0.6 * hip + 0.2 * knee, 0.6 * hip + 0.2 * knee],
        axis=-1,
    )
    motion += generator.normal(0, 0.009, motion.shape)
    groups = {
        "left_leg": [0, 1], "right_leg": [2, 3], "torso": [4],
        "left_arm": [5], "right_arm": [6],
    }
    train, test = motion[:30], motion[30:]
    part = "left_arm"
    masked, missing = mask_body_part(test, groups, part)
    model = BodyPartCompleter(groups).fit(train)
    completed = model.complete(masked, part)
    mean_target = train[..., groups[part]].mean(axis=(0, 1))
    baseline_mae = float(np.abs(test[..., groups[part]] - mean_target).mean())
    mae = float(np.abs(test[missing] - completed[missing]).mean())
    assert np.array_equal(completed[~missing], test[~missing])
    assert np.allclose(completed, model.complete(test, part))
    assert mae < baseline_mae * 0.25, (mae, baseline_mae)
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "completer.npz"
        model.save(saved)
        loaded = BodyPartCompleter.load(saved)
        assert np.allclose(loaded.complete(masked, part), completed)
    print(f"Synthetic held-out MAE: {mae:.4f} rad. Mean baseline: {baseline_mae:.4f} rad.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    train = commands.add_parser("train")
    train.add_argument("--motion", type=Path, required=True)
    train.add_argument("--groups", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--past", type=int, default=2)
    train.add_argument("--future", type=int, default=0)
    train.add_argument("--ridge", type=float, default=1e-3)
    complete = commands.add_parser("complete")
    complete.add_argument("--motion", type=Path, required=True)
    complete.add_argument("--model", type=Path, required=True)
    complete.add_argument("--part", required=True)
    complete.add_argument("--out", type=Path, required=True)
    complete.add_argument("--score", action="store_true", help="Input contains intact ground truth.")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "train":
        groups = json.loads(args.groups.read_text(encoding="utf-8-sig"))
        model = BodyPartCompleter(groups, past=args.past, future=args.future, ridge=args.ridge)
        model.fit(motion_file(args.motion)).save(args.model)
        print(f"Saved {args.model}")
    else:
        model = BodyPartCompleter.load(args.model)
        original = motion_file(args.motion)
        result = model.complete(original, args.part)
        if args.score:
            indices = list(model.groups[args.part])
            reference = original[..., indices]
            if not np.isfinite(reference).all():
                parser.error("--score requires finite ground truth in the masked part.")
            mae = np.abs(result[..., indices] - reference).mean()
            print(f"Held-out reconstruction MAE: {mae:.6g} input units.")
        with args.out.open("wb") as stream:
            np.savez_compressed(stream, motion=result)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
