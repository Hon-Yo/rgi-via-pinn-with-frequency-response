from dataclasses import replace
from io import BytesIO
from math import tau

import numpy as np
import torch
from torch.nn import functional as F

from rgi.geometry import Shell
from rgi.model import LossConfig, ModelConfig, Prediction, RoomCNN, match_planes, supervised_loss

_RECTANGLE = [[0, 0], [4, 0], [4, 3], [0, 3]]
_L_SHAPE = [[0, 0], [5, 0], [5, 2], [2, 2], [2, 4], [0, 4]]


def _prediction_for(shells, slots, *, perturb=False):
    rng = np.random.default_rng(29)
    count = len(shells)
    n_slots = 6
    angles = rng.normal(size=(count, n_slots))
    offsets = np.full((count, n_slots), 30.0)
    for i, (shell, order) in enumerate(zip(shells, slots, strict=True)):
        angles[i, order] = np.arctan2(shell.planes[:, 1], shell.planes[:, 0])
        offsets[i, order] = shell.planes[:, 3]
        if perturb:
            angles[i, order] += rng.uniform(-0.08, 0.08, len(order))
            offsets[i, order] += rng.uniform(-0.15, 0.15, len(order))
    return Prediction(
        angles=torch.from_numpy(angles),
        offsets=torch.from_numpy(offsets),
        presence_logits=torch.from_numpy(rng.normal(size=(count, n_slots))),
        floor=torch.tensor([shell.floor for shell in shells], dtype=torch.float64),
        ceiling=torch.tensor([shell.ceiling for shell in shells], dtype=torch.float64),
        successor_logits=torch.from_numpy(rng.normal(size=(count, n_slots, n_slots))),
    )


def test_cnn_shapes_planes_and_state_reload():
    config = ModelConfig()
    assert config.max_walls == 12
    model = RoomCNN(81, config).eval()
    response = torch.linspace(-2, 2, steps=162).reshape(2, 81)
    with torch.no_grad():
        original = model(response)
    assert original.angles.shape == original.offsets.shape == original.presence_logits.shape == (2, 12)
    assert original.planes.shape == (2, 12, 4)
    assert original.successor_logits.shape == (2, 12, 12)
    assert original.floor.shape == original.ceiling.shape == (2,)
    assert torch.all(original.ceiling > original.floor)
    torch.testing.assert_close(original.planes[:, :, :2].square().sum(dim=-1), torch.ones(2, 12))
    torch.testing.assert_close(original.planes[:, :, 2], torch.zeros(2, 12))
    torch.testing.assert_close(original.planes[:, :, 3], original.offsets)

    saved = BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = RoomCNN(81, config).eval()
    restored.load_state_dict(torch.load(saved, weights_only=True))
    with torch.no_grad():
        reloaded = restored(response[:, None, :])
    for field in Prediction.__dataclass_fields__:
        torch.testing.assert_close(getattr(original, field), getattr(reloaded, field))


def test_loss_is_invariant_to_slot_permutation_and_ccw_start():
    shells = [Shell(_RECTANGLE, -0.4, 2.6), Shell(_L_SHAPE, 0.5, 3.4)]
    expected_slots = [np.array([4, 2, 5, 1]), np.array([3, 0, 4, 2, 5, 1])]
    prediction = _prediction_for(shells, expected_slots, perturb=True)
    config = LossConfig(plane_weight=0.7, presence_weight=3, boundary_weight=0.8, height_weight=2)
    loss, parts, matches = supervised_loss(prediction, shells, config)
    torch.testing.assert_close(loss, sum(parts[name] * getattr(config, f"{name}_weight") for name in parts))
    for match, expected in zip(matches, expected_slots, strict=True):
        np.testing.assert_array_equal(match, expected)

    permutation = torch.tensor([3, 5, 0, 4, 1, 2])
    inverse = permutation.argsort().numpy()
    shuffled = replace(
        prediction,
        angles=prediction.angles[:, permutation],
        offsets=prediction.offsets[:, permutation],
        presence_logits=prediction.presence_logits[:, permutation],
        successor_logits=prediction.successor_logits[:, permutation][:, :, permutation],
    )
    rotated_shells = [Shell(np.roll(shell.footprint, -2, axis=0), shell.floor, shell.ceiling) for shell in shells]
    shuffled_loss, shuffled_parts, shuffled_matches = supervised_loss(shuffled, rotated_shells, config)
    torch.testing.assert_close(loss, shuffled_loss)
    for name in parts:
        torch.testing.assert_close(parts[name], shuffled_parts[name])
    for match, expected in zip(shuffled_matches, expected_slots, strict=True):
        np.testing.assert_array_equal(match, inverse[np.roll(expected, -2)])


def test_matching_is_signed_and_normalizes_positive_plane_scales():
    room = Shell(np.asarray(_RECTANGLE) - [2, 1.5], 0, 3)
    permutation = [2, 1, 0, 3]
    predicted = room.planes[permutation] * np.array([0.5, 2.0, 4.0, 1.5])[:, None]
    rows, cols = match_planes(predicted, room.planes)
    np.testing.assert_array_equal(rows, np.arange(4))
    np.testing.assert_array_equal(cols, permutation)
    rows, cols = match_planes([[-1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0]])
    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(cols, [0])


def test_presence_is_balanced_without_regressing_absent_geometry():
    shell = Shell(_RECTANGLE, -0.2, 2.1)
    slots = [0, 1, 2, 3]
    prediction = _prediction_for([shell], [slots])
    perturbation = torch.tensor([[tau, -tau, tau, -tau, 0.8, -1.2]], dtype=torch.float64)
    modified = replace(
        prediction,
        angles=prediction.angles + perturbation,
        offsets=prediction.offsets + torch.tensor([[0, 0, 0, 0, -60, 90]], dtype=torch.float64),
    )
    loss, parts, matches = supervised_loss(modified, [shell])
    unmodified, _, _ = supervised_loss(prediction, [shell])
    np.testing.assert_array_equal(matches[0], slots)
    torch.testing.assert_close(loss, unmodified)
    torch.testing.assert_close(parts["plane"], loss.new_zeros(()))
    present_logit, absent_logit = prediction.presence_logits[0].split((4, 2))
    expected = 0.5 * (
        F.binary_cross_entropy_with_logits(present_logit, torch.ones_like(present_logit))
        + F.binary_cross_entropy_with_logits(absent_logit, torch.zeros_like(absent_logit))
    )
    torch.testing.assert_close(parts["presence"], expected)


def test_boundary_regularizer_accepts_concavity():
    shell = Shell(_L_SHAPE, 0, 3)
    prediction = _prediction_for([shell], [[0, 1, 2, 3, 4, 5]])
    _, parts, _ = supervised_loss(prediction, [shell])
    expected = F.cross_entropy(prediction.successor_logits[0], torch.tensor([1, 2, 3, 4, 5, 0]))
    torch.testing.assert_close(parts["boundary"], expected)


def test_boundary_regularizer_penalizes_backwards_edges():
    shell = Shell(_RECTANGLE, 0, 3)
    slots = [0, 1, 2, 3]
    prediction = _prediction_for([shell], [slots])
    config = LossConfig(
        plane_weight=0, presence_weight=0, boundary_weight=1, height_weight=0, validity_weight=1
    )
    healthy, _, _ = supervised_loss(prediction, [shell], config)

    # Move the rightmost wall past the leftmost wall without changing its normal.
    invalid = prediction.offsets.clone()
    invalid[0, 1] = 0.25
    invalid.requires_grad_()
    unhealthy, _, matches = supervised_loss(replace(prediction, offsets=invalid), [shell], config)
    np.testing.assert_array_equal(matches[0], slots)
    assert unhealthy > healthy
    unhealthy.backward()
    assert invalid.grad[0, 1] > 0


def test_supervised_gradient_reaches_encoder_and_all_heads():
    model = RoomCNN(65, ModelConfig(max_walls=6))
    response = torch.linspace(-1, 1, steps=130).reshape(2, 65)
    shells = [Shell(_RECTANGLE, 0, 2.7), Shell(_L_SHAPE, -0.1, 3)]
    loss, components, matches = supervised_loss(model(response), shells)
    assert torch.isfinite(loss)
    assert set(components) == {"plane", "presence", "boundary", "height"}
    assert [len(match) for match in matches] == [4, 6]
    loss.backward()
    for layer in (model.encoder[0], model.plane_head, model.presence_head, model.height_head, model.successor_head):
        assert torch.isfinite(layer.weight.grad).all()
        assert torch.count_nonzero(layer.weight.grad) > 0
