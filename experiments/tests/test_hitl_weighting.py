"""Human-in-the-loop row weights: labels -> classes -> weights -> flow loss."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from olmo.data.hitl_weighting import (
    DEMO,
    INTV,
    INTV_GRIP_PENDING,
    PREINTV,
    ROBOT,
    TRANSITION,
    FrameClasses,
    class_counts,
    derive_classes,
    format_class_weights,
    parse_class_weights,
    row_weights,
    sirius_class_weights,
)
from olmo.models.molmoact2.molmoact2 import MolmoAct2
from olmo.nn.action_expert import ActionExpertConfig

R, I, T, G = ROBOT, INTV, TRANSITION, INTV_GRIP_PENDING


def test_preintv_marks_only_policy_frames_right_before_a_takeover():
    source = np.array([R, R, R, R, T, T, G, I, I, T, R, R, R, R, T, I])
    episode = np.zeros_like(source)
    classes = derive_classes(source, episode, preintv_frames=2)
    expected = np.array([R, R, PREINTV, PREINTV, T, T, G, I, I, T, R, R, PREINTV, PREINTV, T, I])
    assert classes.tolist() == expected.tolist()
    # The handback (intv -> transition -> robot) leaves the next policy frames alone.
    assert classes[10] == R


def test_preintv_does_not_cross_episodes():
    source = np.array([R, R, R, R, T])
    episode = np.array([0, 0, 1, 1, 1])
    classes = derive_classes(source, episode, preintv_frames=5)
    assert classes.tolist() == [R, R, PREINTV, PREINTV, T]


def test_sirius_weights_hit_the_target_shares_and_average_to_one():
    counts = {cls: 0 for cls in (ROBOT, INTV, TRANSITION, INTV_GRIP_PENDING, DEMO, PREINTV)}
    counts.update({DEMO: 300, ROBOT: 500, INTV: 120, INTV_GRIP_PENDING: 30, PREINTV: 50, TRANSITION: 999})
    w = sirius_class_weights(counts, intv_share=0.5)
    weighted = {c: w[c] * n for c, n in counts.items() if c != TRANSITION}
    total = sum(n for c, n in counts.items() if c != TRANSITION)  # transitions excluded from P(c)
    assert sum(weighted.values()) / total == pytest.approx(1.0)
    assert (weighted[INTV] + weighted[INTV_GRIP_PENDING]) / total == pytest.approx(0.5)
    assert weighted[DEMO] / total == pytest.approx(300 / total)  # demos keep their share
    assert w[PREINTV] == 0.0 and w[TRANSITION] == 0.0 and w[DEMO] == 1.0
    assert w[INTV_GRIP_PENDING] == w[INTV]


def test_demo_only_data_keeps_unit_weights():
    counts = {cls: 0 for cls in (ROBOT, INTV, TRANSITION, INTV_GRIP_PENDING, DEMO, PREINTV)}
    counts[DEMO] = 1000
    w = sirius_class_weights(counts)
    assert w[DEMO] == 1.0


def test_too_many_demos_for_the_intervention_share_is_refused():
    counts = {cls: 0 for cls in (ROBOT, INTV, TRANSITION, INTV_GRIP_PENDING, DEMO, PREINTV)}
    counts.update({DEMO: 900, ROBOT: 50, INTV: 50})
    with pytest.raises(ValueError, match="lower intv_share"):
        sirius_class_weights(counts, intv_share=0.5)


def test_class_weight_spec_round_trips_and_rejects_gaps():
    counts = class_counts(np.array([DEMO, ROBOT, INTV, TRANSITION]))
    w = sirius_class_weights(counts)
    assert parse_class_weights(format_class_weights(w)) == pytest.approx(w)
    with pytest.raises(ValueError, match="missing"):
        parse_class_weights("demo=1,robot=1")
    with pytest.raises(ValueError, match="bad class weight"):
        parse_class_weights("demo=1,robto=1")


def test_grip_pending_rows_zero_only_the_gripper_dims():
    weights = {ROBOT: 0.5, INTV: 3.0, TRANSITION: 0.0, INTV_GRIP_PENDING: 3.0, DEMO: 1.0, PREINTV: 0.0}
    out = row_weights(np.array([R, G, I, T]), weights, action_dim=7, gripper_dims=(6,))
    assert out.shape == (4, 7)
    assert out[0].tolist() == [0.5] * 7
    assert out[1, :6].tolist() == [3.0] * 6 and out[1, 6] == 0.0
    assert out[2].tolist() == [3.0] * 7
    assert out[3].tolist() == [0.0] * 7
    with pytest.raises(ValueError, match="gripper"):
        row_weights(np.array([G]), weights, action_dim=7, gripper_dims=())


def _write_dataset(root, source, episode):
    (root / "data" / "chunk-000").mkdir(parents=True)
    cols = {"index": np.arange(len(episode)), "episode_index": np.asarray(episode)}
    if source is not None:
        cols["control_source"] = np.asarray(source)
    pq.write_table(pa.table(cols), root / "data" / "chunk-000" / "file-000.parquet")


def test_frame_classes_read_a_chunk_and_stop_at_the_episode_end(tmp_path):
    source = [R, R, R, T, I, I, R, R]
    episode = [0, 0, 0, 0, 0, 0, 1, 1]
    _write_dataset(tmp_path, source, episode)
    frames = FrameClasses(tmp_path, preintv_frames=1)
    assert frames.chunk(1, [0, 1, 2, 3]).tolist() == [R, PREINTV, T, I]
    # Rows past episode 0's end are TRANSITION (weight 0), never episode 1's frames.
    assert frames.chunk(4, [0, 1, 2, 3]).tolist() == [I, I, T, T]


def test_dataset_without_the_column_is_all_demo(tmp_path):
    _write_dataset(tmp_path, None, [0, 0, 1])
    frames = FrameClasses(tmp_path, preintv_frames=30)
    assert frames.classes.tolist() == [DEMO, DEMO, DEMO]


# --- flow loss -------------------------------------------------------------------

B, H, D, S, KV, LAYERS = 2, 30, 8, 7, 48, 2


def _loss_model(rtc_max_delay=0):
    torch.manual_seed(0)
    model = object.__new__(MolmoAct2)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        flow_matching_cutoff=1.0,
        flow_matching_time_offset=0.001,
        flow_matching_time_scale=1.0,
        flow_matching_beta_alpha=1.0,
        flow_matching_beta_beta=1.5,
        mask_action_dim_padding=True,
        rtc_max_delay=rtc_max_delay,
        action_expert_detach_vlm=False,
    )
    cfg = ActionExpertConfig(hidden_size=64, num_layers=LAYERS, num_heads=4, max_action_dim=D, max_horizon=H, compile=None)
    expert = cfg.build(llm_dim=KV, llm_kv_dim=KV, llm_num_layers=LAYERS).cuda().float()
    with torch.no_grad():  # adaLN-zero init would make every prediction 0
        for p in expert.parameters():
            p.normal_(0.0, 0.2)
    model.action_expert = expert
    captured = {}
    forward = expert.forward

    def capture(*args, **kwargs):
        out = forward(*args, **kwargs)
        out.retain_grad()
        captured["pred"] = out
        return out

    expert.forward = capture
    return model, captured


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(1)
    actions = torch.randn(B, H, D, device="cuda", generator=g)
    kv = [(torch.randn(B, S, KV, device="cuda", generator=g), torch.randn(B, S, KV, device="cuda", generator=g))
          for _ in range(LAYERS)]
    dim_pad = torch.zeros(B, D, dtype=torch.bool, device="cuda")
    dim_pad[:, -1] = True
    return dict(
        actions=actions,
        layer_states=None,
        layer_kv_states=kv,
        encoder_attention_mask=None,
        action_horizon_is_pad=torch.zeros(B, H, dtype=torch.bool, device="cuda"),
        action_dim_is_pad=dim_pad,
        states=None,
        packed_batch_idx=None,
        packed_example_ids=None,
        packed_action_chunk_is_valid=None,
        subsegment_ids=None,
    )


def _loss(model, weight, seed=5):
    torch.manual_seed(seed)
    loss, _ = model._compute_flow_matching_loss(**_inputs(), action_loss_weight=weight)
    return loss


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="action expert attention needs CUDA")


@needs_cuda
def test_unit_weights_reproduce_the_unweighted_loss_exactly():
    model, _ = _loss_model()
    ones = torch.ones(B, H, D, device="cuda")
    assert torch.equal(_loss(model, None), _loss(model, ones))
    assert torch.allclose(_loss(model, 2.0 * ones), 2.0 * _loss(model, None))


@needs_cuda
@pytest.mark.parametrize("rtc_max_delay", [0, 5])
def test_zero_weight_rows_and_dims_get_no_gradient(rtc_max_delay):
    model, captured = _loss_model(rtc_max_delay)
    w = torch.ones(B, H, D, device="cuda")
    w[0, 10:20] = 0.0  # a preintv / transition stretch
    w[1, :, 6] = 0.0  # a grip-pending gripper dim
    _loss(model, w).backward()
    grad = captured["pred"].grad.reshape(B, H, D)
    assert torch.all(grad[0, 10:20] == 0)
    assert torch.all(grad[1, :, 6] == 0)
    assert grad[0, :10].abs().sum() > 0 and grad[1, :, :6].abs().sum() > 0
    with pytest.raises(ValueError, match="shape"):
        _loss(model, torch.ones(B, H, D - 1, device="cuda"))
