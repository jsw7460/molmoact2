"""Normalization bounds come from the pooled frames, not averaged episode quantiles."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from launch_scripts.lerobot_utils.stats import _exact_quantiles


def _write(root, action):
    (root / "data" / "chunk-000").mkdir(parents=True)
    table = pa.table({"action": pa.array(action.tolist(), type=pa.list_(pa.float32(), action.shape[1]))})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_pooled_quantiles_cover_episodes_with_different_ranges(tmp_path):
    rng = np.random.default_rng(0)
    # Two datasets whose episodes reach different elbow ranges (part placed near / far).
    near = rng.uniform(0.0, 1.0, size=(3000, 2))
    far = rng.uniform(0.4, 1.4, size=(3000, 2))
    _write(tmp_path / "a", near)
    _write(tmp_path / "b", far)
    exact = _exact_quantiles([("a", str(tmp_path / "a")), ("b", str(tmp_path / "b"))], "action", ["q01", "q99"])
    pooled = np.concatenate([near, far])
    assert np.allclose(exact["q99"], np.quantile(pooled, 0.99, axis=0))
    assert np.allclose(exact["q01"], np.quantile(pooled, 0.01, axis=0))
    averaged = (np.quantile(near, 0.99, axis=0) + np.quantile(far, 0.99, axis=0)) / 2
    assert np.all(exact["q99"] > averaged + 0.15)  # what the old merge stored
