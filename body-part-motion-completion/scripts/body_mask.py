"""CPU baseline for reconstructing one masked body part from a motion clip."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _clips(value, *, allow_missing=False):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("Expected nonempty motion clips shaped (clips, frames, joints).")
    if not allow_missing and not np.isfinite(x).all():
        raise ValueError("Motion clips must contain finite values.")
    return x


def validate_groups(groups: Mapping[str, list[int]], joint_count: int):
    """Check that each joint index belongs to one named group."""
    if not isinstance(groups, Mapping) or len(groups) < 2:
        raise ValueError("Provide at least two groups of joint indices.")
    result = {}
    for name, indices in groups.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(indices, (list, tuple)):
            raise ValueError("Each group must have a nonempty name and a sequence of indices.")
        if not indices or any(type(index) is not int for index in indices):
            raise ValueError(f"Group {name} must contain integer joint indices.")
        result[name] = tuple(indices)
    actual = sorted(index for indices in result.values() for index in indices)
    if actual != list(range(joint_count)):
        raise ValueError("Groups must partition indices 0 through joint_count - 1.")
    return result


def mask_body_part(clips, groups, part):
    """Return a zero-filled copy and a (clips, frames, joints) missing indicator."""
    x = _clips(clips)
    groups = validate_groups(groups, x.shape[-1])
    masked = x.copy()
    missing = np.zeros(x.shape, dtype=bool)
    missing[..., list(groups[part])] = True
    masked[missing] = 0
    return masked, missing


class BodyPartCompleter:
    """One temporally windowed ridge model per body part."""

    def __init__(self, groups, *, past=2, future=0, ridge=1e-3):
        if type(past) is not int or type(future) is not int or min(past, future) < 0:
            raise ValueError("past and future must be nonnegative integers.")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive.")
        self.groups = dict(groups)
        self.past = past
        self.future = future
        self.ridge = float(ridge)

    def _design(self, clips, group):
        seen = np.ones(clips.shape[-1], dtype=bool)
        seen[list(group)] = False
        observed = clips[..., seen]
        if not np.isfinite(observed).all():
            raise ValueError("Visible joints must be finite.")
        frames = observed.shape[1]
        padded = np.pad(observed, ((0, 0), (self.past, self.future), (0, 0)), mode="edge")
        windows = [padded[:, step : step + frames] for step in range(self.past + self.future + 1)]
        features = np.concatenate(windows, axis=-1)
        return features.reshape(-1, features.shape[-1])

    def fit(self, clips):
        x = _clips(clips)
        groups = validate_groups(self.groups, x.shape[-1])
        weights = {}
        for name, indices in groups.items():
            design = self._design(x, indices)
            labels = x[..., list(indices)].reshape(-1, len(indices))
            mean = design.mean(axis=0)
            scale = design.std(axis=0)
            scale = np.where(scale > 1e-9, scale, 1.0)
            centered = (design - mean) / scale
            offset = labels.mean(axis=0)
            n = len(design)
            cov = centered.T @ centered / n
            cross = centered.T @ (labels - offset) / n
            coef = np.linalg.solve(cov + self.ridge * np.eye(cov.shape[0]), cross)
            weights[name] = (mean, scale, offset, coef)
        self.groups = groups
        self.joint_count = x.shape[-1]
        self.weights = weights
        return self

    def complete(self, clips, part):
        if not hasattr(self, "weights"):
            raise RuntimeError("Train or load the model before completion.")
        x = _clips(clips, allow_missing=True)
        if x.shape[-1] != self.joint_count:
            raise ValueError("Joint count differs from the training data.")
        indices = self.groups[part]
        mean, scale, offset, coef = self.weights[part]
        features = (self._design(x, indices) - mean) / scale
        estimate = features @ coef + offset
        output = x.copy()
        output[..., list(indices)] = estimate.reshape(*x.shape[:2], len(indices))
        return output

    def save(self, path):
        if not hasattr(self, "weights"):
            raise RuntimeError("Train the model before saving.")
        metadata = json.dumps(
            dict(groups=self.groups, joint_count=self.joint_count, past=self.past,
                 future=self.future, ridge=self.ridge),
            ensure_ascii=False,
        )
        arrays = {"metadata": np.asarray(metadata)}
        for i, name in enumerate(self.groups):
            for key, value in zip(("mean", "scale", "offset", "coef"), self.weights[name]):
                arrays[f"{i}_{key}"] = value
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            config = json.loads(arrays["metadata"].item())
            joint_count = config.pop("joint_count")
            model = cls(**config)
            model.groups = validate_groups(model.groups, joint_count)
            model.joint_count = joint_count
            model.weights = {
                name: tuple(arrays[f"{i}_{key}"] for key in ("mean", "scale", "offset", "coef"))
                for i, name in enumerate(model.groups)
            }
        return model
