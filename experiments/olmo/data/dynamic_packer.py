import dataclasses
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from olmo.config import BaseConfig
from olmo.preprocessing.preprocessor_utils import TOKEN_POOLING_KEYS
from olmo.preprocessing.multimodal_preprocessor import MultimodalTypes

ACTION_HORIZON_IS_PAD_KEY = "action_horizon_is_pad"
LEGACY_ACTION_IS_PAD_KEY = "action_is_pad"
PACKED_ACTION_HORIZON_IS_PAD_KEY = "packed_action_horizon_is_pad"
LEGACY_PACKED_ACTION_IS_PAD_KEY = "packed_action_is_pad"

ACTION_LOSS_WEIGHT_KEY = "action_loss_weight"
PACKED_ACTION_LOSS_WEIGHT_KEY = "packed_action_loss_weight"
ACTION_DATA_KEYS = {
    "states",
    "actions",
    ACTION_HORIZON_IS_PAD_KEY,
    LEGACY_ACTION_IS_PAD_KEY,
    "action_dim_is_pad",
    ACTION_LOSS_WEIGHT_KEY,
}
PACKED_ACTION_META_KEYS = {
    "packed_states",
    "packed_actions",
    PACKED_ACTION_HORIZON_IS_PAD_KEY,
    LEGACY_PACKED_ACTION_IS_PAD_KEY,
    "packed_action_dim_is_pad",
    PACKED_ACTION_LOSS_WEIGHT_KEY,
    "packed_example_ids",
    "packed_num_chunks",
}
DEPTH_SIDE_CHANNEL_KEYS = {"depth_updated_mask", "depth_buffer_codes"}
PACKED_DEPTH_META_KEYS = {
    "packed_depth_updated_mask",
    "packed_depth_buffer_codes",
    "packed_depth_example_ids",
    "packed_num_depth_examples",
}
ALL_ACTION_KEYS = ACTION_DATA_KEYS | PACKED_ACTION_META_KEYS
ALL_DEPTH_KEYS = DEPTH_SIDE_CHANNEL_KEYS | PACKED_DEPTH_META_KEYS


EXAMPLE_SUBSEGMENT_INCREMENT = 100000


def _extract_action_chunk(example: Dict[str, Any], example_ix: int) -> Optional[Dict[str, Any]]:
    chunk: Dict[str, Any] = {}
    for key in ACTION_DATA_KEYS:
        if key in example:
            chunk[key] = example.pop(key)
    if not chunk:
        return None
    chunk["example_ix"] = example_ix
    return chunk


def _extract_depth_side_channel(example: Dict[str, Any], example_ix: int) -> Optional[Dict[str, Any]]:
    side_channel: Dict[str, Any] = {}
    for key in DEPTH_SIDE_CHANNEL_KEYS:
        if key in example:
            side_channel[key] = example.pop(key)
    if not side_channel:
        return None
    side_channel["example_ix"] = example_ix
    return side_channel


def _pack_action_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks or len(chunks) == 0:
        return {}

    def _as_array(value, dtype=None):
        if value is None:
            return None
        arr = np.asarray(value)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    states_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    pad_list: List[np.ndarray] = []
    dim_pad_list: List[np.ndarray] = []
    weight_list: List[np.ndarray] = []
    example_ids: List[int] = []
    for chunk in chunks:
        example_ids.append(int(chunk["example_ix"]))
        states = _as_array(chunk.get("states"), np.float32)
        actions = _as_array(chunk.get("actions"), np.float32)
        pads = chunk.get(ACTION_HORIZON_IS_PAD_KEY)
        if pads is None:
            pads = chunk.get(LEGACY_ACTION_IS_PAD_KEY)
        dim_pads = chunk.get("action_dim_is_pad")
        if states is not None:
            states_list.append(states)
        if actions is not None:
            actions_list.append(actions)
        if pads is None and actions is not None:
            pads = np.zeros(actions.shape[:-1], dtype=np.bool_)
        if dim_pads is None and actions is not None:
            dim_pads = np.zeros(actions.shape[-1], dtype=np.bool_)
        if pads is not None:
            pad_list.append(_as_array(pads, np.bool_))
        if dim_pads is not None:
            dim_pad_list.append(_as_array(dim_pads, np.bool_))
        weight = chunk.get(ACTION_LOSS_WEIGHT_KEY)
        if weight is not None:
            weight_list.append(_as_array(weight, np.float32))

    out: Dict[str, Any] = {}
    if states_list:
        out["packed_states"] = np.stack(states_list, axis=0)
    if actions_list:
        out["packed_actions"] = np.stack(actions_list, axis=0)
    if pad_list:
        out[PACKED_ACTION_HORIZON_IS_PAD_KEY] = np.stack(pad_list, axis=0)
    if dim_pad_list:
        out["packed_action_dim_is_pad"] = np.stack(dim_pad_list, axis=0)
    if weight_list:
        if len(weight_list) != len(example_ids):
            raise ValueError("action_loss_weight must be set on every packed chunk or on none")
        out[PACKED_ACTION_LOSS_WEIGHT_KEY] = np.stack(weight_list, axis=0)
    out["packed_example_ids"] = np.asarray(example_ids, dtype=np.int64)
    out["packed_num_chunks"] = np.asarray([len(example_ids)], dtype=np.int32)
    return out


def _pack_depth_side_channels(side_channels: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not side_channels:
        return {}

    def _as_array(value, dtype=None):
        if value is None:
            return None
        arr = np.asarray(value)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    updated_masks: List[np.ndarray] = []
    buffer_codes: List[np.ndarray] = []
    example_ids: List[int] = []
    for side_channel in side_channels:
        example_ids.append(int(side_channel["example_ix"]))
        updated_mask = _as_array(side_channel.get("depth_updated_mask"), np.bool_)
        depth_buffer_codes = _as_array(side_channel.get("depth_buffer_codes"), np.int64)
        if updated_mask is not None:
            updated_masks.append(updated_mask)
        if depth_buffer_codes is not None:
            buffer_codes.append(depth_buffer_codes)

    out: Dict[str, Any] = {
        "packed_depth_example_ids": np.asarray(example_ids, dtype=np.int64),
        "packed_num_depth_examples": np.asarray([len(example_ids)], dtype=np.int32),
    }
    if updated_masks:
        out["packed_depth_updated_mask"] = np.stack(updated_masks, axis=0)
    if buffer_codes:
        out["packed_depth_buffer_codes"] = np.stack(buffer_codes, axis=0)
    return out


def pack(*examples: Dict) -> Dict:
    keys = set()
    for ex in examples:
        keys.update(ex)
    keys = {k for k in keys if k not in {"metadata"} | ALL_ACTION_KEYS | ALL_DEPTH_KEYS}
    if "subsegment_ids" not in keys:
        keys.add("subsegment_ids")
    patch_keys = [
        k for k in TOKEN_POOLING_KEYS if k in keys]
    if "images" in keys:
        assert len(patch_keys) > 0, "Example had images but no image->token mapping idx"
    image_offset = 0
    action_chunks: List[Dict[str, Any]] = []
    depth_side_channels: List[Dict[str, Any]] = []
    for example_ix, example in enumerate(examples):
        chunk = _extract_action_chunk(example, example_ix)
        if chunk is not None:
            action_chunks.append(chunk)
        depth_side_channel = _extract_depth_side_channel(example, example_ix)
        if depth_side_channel is not None:
            depth_side_channels.append(depth_side_channel)
        # Patch indices need to be offset by total number of images patches
        if "images" in example:
            n_patches = np.prod(example["images"].shape[:2])
            for k in patch_keys:
                if k in example:
                    assert np.all(example[k] < n_patches)
                    example[k] = np.where(example[k] >= 0, example[k] + image_offset, example[k])
            image_offset += n_patches
            assert "position_ids" in example
        n_tokens = len(example["input_tokens"])

        # Modify or add subsegment ids to prevent intra-example attention
        # Tokens can only attend to subsegments >= then their subsegment, so
        # we give example increasing subsegments to prevent cross-example attention
        example_subsegemnt_id = example_ix*EXAMPLE_SUBSEGMENT_INCREMENT
        if "subsegment_ids" not in example:
            example["subsegment_ids"] = np.full([n_tokens], example_subsegemnt_id)
        else:
            example["subsegment_ids"] += example_subsegemnt_id

    if "images" in keys:
        img = next(iter(ex for ex in examples if "images" in ex))["images"]
        for ex in examples:
            if "images" not in ex:
                ex["images"] = np.zeros([0]+list(img.shape[1:]), dtype=img.dtype)

    offset = 0
    if "point_target_ids" in keys:
        dim = [example["point_target_ids"] for example in examples
               if "point_target_ids" in example][0].shape[1]
        for ex in examples:
            if "point_target_ids" in ex:
                target_ids = ex["point_target_ids"]
                target_ids[:, 0] = np.where(
                    target_ids[:, 0] >= 0, target_ids[:, 0] + offset, target_ids[:, 0])
            else:
                ex["point_target_ids"] = np.zeros([0, dim], dtype=np.int64)
            if "token_pooling" in ex:
                n_image_tokens = np.any(ex["token_pooling"] >= 0, -1).sum()
                offset += n_image_tokens

    for ex in examples:
        for key in patch_keys:
            max_pooling_shape = max(ex[key].shape[1] for ex in examples if key in ex)
            if key not in ex:
                ex[key] = np.full([0, max_pooling_shape], -1)
            elif ex[key].shape[1] < max_pooling_shape:
                delta = max_pooling_shape - ex[key].shape[1]
                ex[key] = np.pad(ex[key], [[0, 0], [0, delta]], constant_values=-1)
        for key in ["num_images", "num_image_starts"]:
            if key in patch_keys and key not in ex:
                ex[key] = np.array([0], dtype=np.int64)
        if "multimodal_type" in patch_keys and "multimodal_type" not in ex:
            ex["multimodal_type"] = np.array([MultimodalTypes.TEXT_ONLY], dtype=np.int64)
    packed = {k: np.concatenate([ex[k] for ex in examples], axis=0) for k in keys if k != "metadata"}
    packed.update(_pack_action_chunks(action_chunks))
    packed.update(_pack_depth_side_channels(depth_side_channels))
    if any("metadata" in ex for ex in examples):
        # Preserve per-example metadata order so trainer-side online teacher models
        # can recover raw inputs even after dynamic packing.
        packed["metadata"] = [ex.get("metadata", {}) for ex in examples]
    return packed


def packed_iterator(it, packer):
    for i, ex in enumerate(it):
        out = packer(i, ex)
        if out is not None:
            yield out


@dataclasses.dataclass
class PackingConfig(BaseConfig):
    buffer_size: int = 32
    mode: str = "dynamic_solver"

    text_weight: float = 1.0
    """Text token weight for the dynamic solver"""

    image_weight: float = 10.0
    """Image token weight for the dynamic solver"""

    shortcut_max_len_images: bool = False
    """Don't buffer examples that have the max number of images"""

    pad_action_chunks: bool = False
    """If True, pad each packed sample to a fixed number of action chunks in the collator."""

    action_chunk_cap: Optional[int] = None
    """Maximum number of action chunks selected for each packed sample."""

    track_packing_state: bool = True

    def bulid(self, text_max_len, image_max_len):
        if self.mode == "dynamic_solver":
            text_c = Constraint("input_tokens", text_max_len, True, self.text_weight, max(1, text_max_len//512))
            image_c = Constraint("images", image_max_len, self.shortcut_max_len_images, self.image_weight, 1)
            return DynamicSolver(
                self.buffer_size,
                [text_c, image_c],
                max_action_chunks=self.action_chunk_cap,
            )
        else:
            raise ValueError(self.mode)


def select_subset_2d_knapsack(t_values, i_values, max_t, max_i, obj_vals):
    """Vectorized 2D knapsack dynamic program solver"""
    M = len(t_values)

    # DP table with quantized dimensions
    dp = np.zeros((M + 1, max_t + 1, max_i + 1), dtype=np.float32)

    # Vectorized DP fill
    for item in range(1, M + 1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]
        obj_val = obj_vals[item - 1]

        # Copy previous layer
        dp[item] = dp[item - 1]

        # Vectorized update where item can fit
        if t_val_q <= max_t and i_val_q <= max_i:
            # Create shifted view for the "take item" case
            take_val = dp[item - 1, :max_t + 1 - t_val_q, :max_i + 1 - i_val_q] + obj_val

            # Update positions where taking item is better
            dp[item, t_val_q:, i_val_q:] = np.maximum(
                dp[item, t_val_q:, i_val_q:],
                take_val
            )

    # Backtrack to find solution
    selected_indices = []
    t_rem_q, i_rem_q = max_t, max_i

    for item in range(M, 0, -1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]

        if (t_val_q <= t_rem_q and i_val_q <= i_rem_q and
            dp[item, t_rem_q, i_rem_q] != dp[item - 1, t_rem_q, i_rem_q]):
            selected_indices.append(item - 1)
            t_rem_q -= t_val_q
            i_rem_q -= i_val_q
    return selected_indices


def select_subset_3d_knapsack(t_values, i_values, c_values, max_t, max_i, max_c, obj_vals):
    """Vectorized 3D knapsack dynamic program solver."""
    M = len(t_values)

    dp = np.zeros((M + 1, max_t + 1, max_i + 1, max_c + 1), dtype=np.float32)

    for item in range(1, M + 1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]
        c_val_q = c_values[item - 1]
        obj_val = obj_vals[item - 1]

        dp[item] = dp[item - 1]

        if t_val_q <= max_t and i_val_q <= max_i and c_val_q <= max_c:
            take_val = (
                dp[
                    item - 1,
                    :max_t + 1 - t_val_q,
                    :max_i + 1 - i_val_q,
                    :max_c + 1 - c_val_q,
                ]
                + obj_val
            )
            dp[item, t_val_q:, i_val_q:, c_val_q:] = np.maximum(
                dp[item, t_val_q:, i_val_q:, c_val_q:],
                take_val,
            )

    selected_indices = []
    t_rem_q, i_rem_q, c_rem_q = max_t, max_i, max_c

    for item in range(M, 0, -1):
        t_val_q = t_values[item - 1]
        i_val_q = i_values[item - 1]
        c_val_q = c_values[item - 1]

        if (
            t_val_q <= t_rem_q
            and i_val_q <= i_rem_q
            and c_val_q <= c_rem_q
            and dp[item, t_rem_q, i_rem_q, c_rem_q] != dp[item - 1, t_rem_q, i_rem_q, c_rem_q]
        ):
            selected_indices.append(item - 1)
            t_rem_q -= t_val_q
            i_rem_q -= i_val_q
            c_rem_q -= c_val_q
    return selected_indices


@dataclasses.dataclass
class Constraint:
    key: str
    """Key in example dictionaries to contain"""

    max_len: int
    """Max total length of `key` tensors in the packed examples"""

    allow_shortcut: bool
    """Don't buffer examples that are at `max_len` on their own"""

    weight: float
    """Value to put on this example in the solver"""

    granularity: int
    """Granularity to run the solver at"""

    def get_quantized_value(self, val: int):
        return (val + self.granularity - 1) // self.granularity

    def get_quantized_max_len(self):
        return self.get_quantized_value(self.max_len)


@dataclasses.dataclass
class BufferedExample:
    example_id: int
    value: float
    quantized_lens: Dict[str, int]
    example: Dict
    action_chunk_count: int = 0


class DynamicSolver:
    """Pack examples by running a dynamic program to optimize what examples to pack"""

    def __init__(
        self,
        max_buffer_size: int,
        constraints: List[Constraint],
        verbosity=0,
        max_action_chunks: Optional[int] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self._buffer: List[BufferedExample] = []
        self.verbosity = verbosity
        self.constraints = constraints
        self.max_action_chunks = None if max_action_chunks is None else int(max_action_chunks)

    @staticmethod
    def _action_chunk_count(example: Dict) -> int:
        return 1 if any(key in example for key in ACTION_DATA_KEYS) else 0

    def _example_str(self, example):
        return ' '.join(f"{c.key}={len(example.get(c.key, []))}" for c in self.constraints)

    def get_buffered_example_ids(self):
        return [x.example_id for x in self._buffer]

    def __call__(self, example_id: int, example: Dict) -> List:
        # Maybe short-cut the example
        for constraint in self.constraints:
            if constraint.allow_shortcut:
                m = len(example[constraint.key])
                if constraint.get_quantized_value(m) >= constraint.get_quantized_max_len():
                    if self.verbosity > 1:
                        print(f"Example already at constraints: {self._example_str(example)}")
                    return pack(example)

        # Build the `BufferedExample` with cached values/lengths
        quantized_lens = {}
        value = 0
        for c in self.constraints:
            if c.key in example:
                quantized_lens[c.key] = c.get_quantized_value(len(example[c.key]))
                value += len(example[c.key])*c.weight
            else:
                quantized_lens[c.key] = 0
        buffered_example = BufferedExample(
            example_id,
            quantized_lens=quantized_lens,
            value=value,
            example=example,
            action_chunk_count=self._action_chunk_count(example),
        )

        if len(self._buffer) < self.max_buffer_size:
            self._buffer.append(buffered_example)
            if self.verbosity > 1:
                print(f"Add to pool: {self._example_str(example)}, buffer_sz{len(self._buffer)}")
            return None
        # else Buffer is full so we need to run the solver to make room

        if self.verbosity > 1:
            for c in self.constraints:
                print(f"{c.key}: {[len(x.example.get(c.key, [])) for x in self._buffer]}")

        # Run the solver on the full buffer
        if len(self.constraints) != 2:
            raise NotImplementedError("Solver currently only supports exactly 2 constraints")

        c1, c2 = list(self.constraints)
        if self.max_action_chunks is None:
            indices = select_subset_2d_knapsack(
                [ex.quantized_lens[c1.key] for ex in self._buffer],
                [ex.quantized_lens[c2.key] for ex in self._buffer],
                c1.get_quantized_max_len(),
                c2.get_quantized_max_len(),
                [ex.value for ex in self._buffer]
            )
        else:
            indices = select_subset_3d_knapsack(
                [ex.quantized_lens[c1.key] for ex in self._buffer],
                [ex.quantized_lens[c2.key] for ex in self._buffer],
                [ex.action_chunk_count for ex in self._buffer],
                c1.get_quantized_max_len(),
                c2.get_quantized_max_len(),
                self.max_action_chunks,
                [ex.value for ex in self._buffer]
            )
        if len(indices) == 0:
            raise RuntimeError("No indices returned by dynamic packing solver")

        if self.verbosity > 0:
            print(f"Yield {indices}")
            for c in self.constraints:
                vals = [len(self._buffer[i].example.get(c.key, ())) for i in indices]
                print(f"{c.key}: {sum(vals)} {vals}")

        # Pack and remove the selected example
        out = pack(*(self._buffer[i].example for i in indices))
        for ix in sorted(indices, reverse=True):
            self._buffer.pop(ix)
        self._buffer.append(buffered_example)
        return out
