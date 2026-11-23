from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from rgi.audio import save_feature
from rgi.config import Config, TrainingConfig, read_json, write_json
from rgi.data import DATASET_VERSION, split_rooms, validate_split
from rgi.diagnostics import probe_case
from rgi.model import ModelConfig, RoomCNN
from rgi.physics import matched_geometry, physics_loss
from rgi.pipeline import FeatureStats, train


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


def test_measured_training_uses_corrected_db_not_raw_transfer(tmp_path):
    vector, shell, objective = probe_case()
    # Covered features keep the raw system transfer for diagnostics. Give it a
    # different spectral shape while keeping the corrected target unchanged.
    raw_color = np.linspace(0.25, 2, len(objective.feature.frequency_hz))
    mask = objective.feature.valid_mask.copy()
    mask[0] = False
    corrected_db = objective.feature.magnitude_db.copy()
    corrected_db[0] += 200  # an invalid bin must not contribute to physics loss
    feature = replace(objective.feature, calibration="covered", magnitude_db=corrected_db, valid_mask=mask,
                      transfer_real=objective.feature.transfer_real * raw_color,
                      transfer_imag=objective.feature.transfer_imag * raw_color)
    config = Config(audio=objective.audio, simulator=objective.simulator, model=ModelConfig(max_walls=4),
                    training=TrainingConfig(epochs=1, physics_warmup_epochs=0, physics_ramp_epochs=1,
                                            physics_weight=0.005, threshold_count=1))
    model = RoomCNN(len(feature.frequency_hz), config.model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.plane_head.bias.copy_(torch.tensor(vector[:-2].reshape(-1, 2).T.ravel()))
        height = shell.ceiling - shell.floor - config.model.min_height_m
        model.height_head.bias.copy_(torch.tensor([shell.floor, np.log(np.expm1(height))]))

    init = tmp_path / "init.pt"
    torch.save({"version": 1, "config": config.to_dict(), "model": model.state_dict(),
                "statistics": FeatureStats.fit([feature]).to_dict(), "learned_rooms": []}, init)
    save_feature(tmp_path / "covered.npz", feature)
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    index = tmp_path / "dataset.json"
    write_json(index, {"version": DATASET_VERSION, "audio": asdict(config.audio), "measurements": [
        {"room_id": room, "measurement_id": room, "origin": "measured", "device_config_id": "laptop",
         "calibration": {"mode": "covered", "version": "cloth-v1"}, "feature": "covered.npz",
         "shell": shell.to_dict(), "pose": {**asdict(objective.pose), "physics_approved": True}}
        for room in ("a", "b", "c")
    ]})
    output = tmp_path / "model"
    checkpoint = train(index, config, split, output, init=init)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["physics_examples_seen"] == 1

    # Evaluate the initial prediction with the same mask but a clean synthetic
    # target: it must give the same loss used in training.
    objective.feature = replace(objective.feature, valid_mask=mask)
    with torch.no_grad():
        initial = matched_geometry(model(torch.zeros(1, len(feature.frequency_hz))), 0, np.arange(4))
    expected_db = objective(initial.double().numpy())
    epoch = read_json(output / "history.json")[0]
    assert epoch["physics"]["mean_loss_db"] == pytest.approx(expected_db, abs=1e-6)
    assert epoch["train_losses"]["total"] == pytest.approx(
        epoch["train_losses"]["supervised"] + config.training.physics_weight * expected_db
    )
