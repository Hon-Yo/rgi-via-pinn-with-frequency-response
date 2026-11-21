import numpy as np
import pytest
import torch

from rgi.data import split_rooms, validate_split
from rgi.diagnostics import probe_case
from rgi.model import ModelConfig, RoomCNN
from rgi.physics import matched_geometry, physics_loss


def test_grouped_holdout_keeps_all_poses_and_takes_together():
    ids = ["a", "b", "a", "d", "d", "c", "e", "b", "c"]
    split = split_rooms(ids, 29, test_rooms=["b"], val_rooms=["a"])
    assert split == {"train": ["c", "d", "e"], "validation": ["a"], "test": ["b"]}
    assignments = {name: [i for i, room in enumerate(ids) if room in members] for name, members in split.items()}
    assert assignments == {"train": [3, 4, 5, 6, 8], "validation": [0, 2], "test": [1, 7]}
    with pytest.raises(ValueError, match="overlap"):
        validate_split({**split, "train": ["a", *split["train"]]}, ids)


def test_simulator_finite_difference_reaches_cnn_with_upstream_scale():
    vector, _, objective = probe_case()
    torch.manual_seed(4)
    model = RoomCNN(len(objective.audio.frequency_hz), ModelConfig(max_walls=4, conv_channels=(4, 6, 8, 8),
                                                                  hidden_features=8)).double()
    x = torch.tensor(objective.feature.magnitude_db[None], dtype=torch.float64)
    slots = np.arange(4)
    initial = matched_geometry(model(x), 0, slots).detach()
    target = torch.tensor(vector)
    local = matched_geometry(model(x), 0, slots) - initial + target
    loss = physics_loss(local, objective)
    assert objective.calls == 1 + 2 * len(vector)
    assert 0 < loss.item() == pytest.approx(objective(vector))
    (0.165 * loss).backward()
    grad = model.plane_head.bias.grad[5].item()  # second wall offset
    h = objective.simulator.distance_step
    dp = np.zeros_like(vector)
    dp[3] = h
    assert grad == pytest.approx(0.165 * (objective(vector + dp) - objective(vector - dp)) / (2 * h))
    assert abs(grad) > 0.01
    assert torch.linalg.vector_norm(model.encoder[0].weight.grad) > 0

    with torch.no_grad():
        model.plane_head.bias[5] -= np.sign(grad) * h
        moved = matched_geometry(model(x), 0, slots) - initial + target
    assert objective(moved.numpy()) < loss.item()
