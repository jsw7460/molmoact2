"""Per-row action loss weights for human-in-the-loop deployment data.

Rollouts recorded with operator takeover carry a per-frame ``control_source``
column: 0 policy, 1 human, 2 transition (align / blend / handback), 3 human arm
with the gripper not yet caught. Teleop demonstrations either carry 4 or have
no column at all (then every frame is a demonstration).

Training weights follow Sirius (Liu et al., RSS 2023), weighted behavior
cloning with importance weights ``w(c) = P*(c) / P(c)``:

* ``P(c)`` is each class's share of the frames, transitions excluded;
* ``P*(intv) = intv_share`` (Sirius: 0.5), ``P*(demo) = P(demo)``,
  ``P*(preintv) = 0``, and robot frames take the remainder;
* pre-intervention frames are the last ``preintv_frames`` policy frames before
  each takeover (the operator judged them bad);
* transitions weigh 0; grip-pending frames take the intervention weight on the
  arm dims and 0 on the gripper dims (their gripper action is the held value).

The weights average to 1 over the dataset, so the loss keeps its usual
mean-over-rows reduction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

CONTROL_SOURCE_KEY = "control_source"

ROBOT = 0
INTV = 1
TRANSITION = 2
INTV_GRIP_PENDING = 3
DEMO = 4
PREINTV = 5

CLASS_NAMES = {
    ROBOT: "robot",
    INTV: "intv",
    TRANSITION: "transition",
    INTV_GRIP_PENDING: "intv_grip_pending",
    DEMO: "demo",
    PREINTV: "preintv",
}
_NAME_TO_CLASS = {name: cls for cls, name in CLASS_NAMES.items()}
_RECORDED = (ROBOT, INTV, TRANSITION, INTV_GRIP_PENDING, DEMO)


def load_frame_labels(dataset_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(index, episode_index, control_source)`` per frame of a LeRobot v3
    dataset, sorted by global index. A dataset without the column is all DEMO."""
    files = sorted(Path(dataset_root).glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset_root}/data")
    schema_names = pq.read_schema(files[0]).names
    has_source = CONTROL_SOURCE_KEY in schema_names
    columns = ["index", "episode_index"] + ([CONTROL_SOURCE_KEY] if has_source else [])
    parts = [pq.read_table(f, columns=columns) for f in files]
    index = np.concatenate([p.column("index").to_numpy() for p in parts]).astype(np.int64)
    episode = np.concatenate([p.column("episode_index").to_numpy() for p in parts]).astype(np.int64)
    if has_source:
        source = np.concatenate(
            [np.asarray(p.column(CONTROL_SOURCE_KEY).to_pylist()).reshape(-1) for p in parts]
        ).astype(np.int64)
        bad = sorted(set(np.unique(source).tolist()) - set(_RECORDED))
        if bad:
            raise ValueError(f"{dataset_root}: unknown control_source values {bad}")
    else:
        source = np.full(index.shape, DEMO, dtype=np.int64)
    order = np.argsort(index, kind="stable")
    return index[order], episode[order], source[order]


def derive_classes(source: np.ndarray, episode: np.ndarray, preintv_frames: int) -> np.ndarray:
    """Recorded sources -> training classes: policy frames within
    ``preintv_frames`` before a takeover (policy -> transition in the same
    episode) become PREINTV. Frames must be in temporal order."""
    if preintv_frames < 0:
        raise ValueError(f"preintv_frames must be >= 0, got {preintv_frames}")
    classes = source.astype(np.int64).copy()
    if preintv_frames == 0 or len(classes) < 2:
        return classes
    takeover = np.flatnonzero(
        (source[:-1] == ROBOT) & (source[1:] == TRANSITION) & (episode[:-1] == episode[1:])
    ) + 1
    for start in takeover:
        lo = start - 1
        while lo >= 0 and start - lo <= preintv_frames and episode[lo] == episode[start] and source[lo] == ROBOT:
            classes[lo] = PREINTV
            lo -= 1
    return classes


def class_counts(classes: np.ndarray) -> Dict[int, int]:
    return {cls: int((classes == cls).sum()) for cls in CLASS_NAMES}


def sirius_class_weights(counts: Mapping[int, int], intv_share: float = 0.5) -> Dict[int, float]:
    """Importance weights ``P*(c) / P(c)`` with transitions excluded from P."""
    if not 0.0 < intv_share < 1.0:
        raise ValueError(f"intv_share must be in (0, 1), got {intv_share}")
    n_intv = counts[INTV] + counts[INTV_GRIP_PENDING]
    total = counts[DEMO] + counts[ROBOT] + counts[PREINTV] + n_intv
    if total == 0:
        raise ValueError("no weighted frames (everything is a transition)")
    p_demo = counts[DEMO] / total
    target_intv = intv_share if n_intv > 0 else 0.0
    target_robot = 1.0 - p_demo - target_intv
    if counts[ROBOT] > 0 and target_robot < 0.0:
        raise ValueError(
            f"demonstrations are {p_demo:.2f} of the frames, so policy frames cannot share the "
            f"remaining {1.0 - target_intv:.2f} after intv_share={intv_share}; lower intv_share"
        )
    weights = {cls: 0.0 for cls in CLASS_NAMES}
    if counts[DEMO] > 0:
        weights[DEMO] = 1.0
    if n_intv > 0:
        weights[INTV] = weights[INTV_GRIP_PENDING] = target_intv / (n_intv / total)
    if counts[ROBOT] > 0:
        weights[ROBOT] = target_robot / (counts[ROBOT] / total)
    return weights


def format_class_weights(weights: Mapping[int, float]) -> str:
    return ",".join(f"{CLASS_NAMES[cls]}={weights[cls]:.6g}" for cls in sorted(weights))


def parse_class_weights(spec: str) -> Dict[int, float]:
    """``"demo=1,robot=0.7,intv=3.2,..."`` -> {class: weight}. Every class must
    be listed, so a typo or a missing class fails loudly."""
    weights: Dict[int, float] = {}
    for part in spec.split(","):
        name, _, value = part.strip().partition("=")
        if name not in _NAME_TO_CLASS or not value:
            raise ValueError(f"bad class weight entry {part!r}; expected <class>=<weight> with class in {sorted(_NAME_TO_CLASS)}")
        weights[_NAME_TO_CLASS[name]] = float(value)
    missing = sorted(set(CLASS_NAMES) - set(weights))
    if missing:
        raise ValueError(f"class weights missing {[CLASS_NAMES[c] for c in missing]}")
    return weights


def parse_gripper_dims(spec: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in spec.split(",") if x.strip())


def row_weights(
    classes: np.ndarray,
    weights: Mapping[int, float],
    action_dim: int,
    gripper_dims: Sequence[int],
) -> np.ndarray:
    """``(H,)`` row classes -> ``(H, action_dim)`` loss weights."""
    out = np.asarray([weights[int(c)] for c in classes], dtype=np.float32)[:, None].repeat(action_dim, axis=1)
    pending = classes == INTV_GRIP_PENDING
    if pending.any():
        if not gripper_dims:
            raise ValueError("intv_grip_pending rows need the gripper action dims (hitl_gripper_dims)")
        for dim in gripper_dims:
            out[pending, dim] = 0.0
    return out


class FrameClasses:
    """Training class of every frame of one dataset, looked up by global index."""

    def __init__(self, dataset_root: Path, preintv_frames: int):
        index, episode, source = load_frame_labels(dataset_root)
        self.index = index
        self.episode = episode
        self.classes = derive_classes(source, episode, preintv_frames)

    def chunk(self, start_index: int, offsets: Sequence[int]) -> np.ndarray:
        """Classes of the rows ``start_index + offsets`` in the start frame's
        episode; rows past the episode get TRANSITION (weight 0; they are also
        masked as horizon padding)."""
        pos = int(np.searchsorted(self.index, start_index))
        if pos >= len(self.index) or self.index[pos] != start_index:
            raise KeyError(f"frame index {start_index} not in the dataset labels")
        ep = self.episode[pos]
        out = np.full(len(offsets), TRANSITION, dtype=np.int64)
        for k, off in enumerate(offsets):
            p = pos + int(off)
            if 0 <= p < len(self.index) and self.episode[p] == ep:
                out[k] = self.classes[p]
        return out


def dataset_class_counts(dataset_roots: Sequence[Path], preintv_frames: int) -> Dict[int, int]:
    """Class counts summed over the datasets of one mixture."""
    total = {cls: 0 for cls in CLASS_NAMES}
    for root in dataset_roots:
        _, episode, source = load_frame_labels(root)
        for cls, n in class_counts(derive_classes(source, episode, preintv_frames)).items():
            total[cls] += n
    return total


def describe(counts: Mapping[int, int], weights: Optional[Mapping[int, float]] = None) -> str:
    lines = []
    for cls in sorted(CLASS_NAMES):
        w = "" if weights is None else f"  weight {weights[cls]:.4g}"
        lines.append(f"  {CLASS_NAMES[cls]:>18}: {counts[cls]:>8} frames{w}")
    return "\n".join(lines)
