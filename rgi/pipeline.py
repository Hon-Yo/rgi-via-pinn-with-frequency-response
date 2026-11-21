"""Supervised training, optional acoustic refinement, evaluation and WAV inference."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .audio import process_wavs
from .config import Config, write_json
from .data import feature_signature, load_dataset, reference_feature, validate_split
from .model import RoomCNN, supervised_loss
from .physics import AcousticObjective, comparable_alignment, matched_geometry, physics_loss
from .report import decode_prediction, plot_geometry, plot_spectrum, predictions, score, summarize, tune_threshold


@dataclass
class FeatureStats:
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray

    @classmethod
    def fit(cls, features):
        magnitude = np.stack([feature.magnitude_db for feature in features], dtype=np.float64)
        valid = np.stack([feature.valid_mask for feature in features])
        count = valid.sum(axis=0)
        divisor = np.maximum(count, 1)
        mean = np.where(valid, magnitude, 0).sum(axis=0) / divisor
        scale = np.sqrt(np.where(valid, (magnitude - mean) ** 2, 0).sum(axis=0) / divisor)
        scale[scale < 1e-5] = 1.0
        return cls(mean, scale, count)

    def transform(self, features):
        values = np.stack([feature.magnitude_db for feature in features])
        valid = np.stack([feature.valid_mask for feature in features]) & (self.count > 0)
        if not np.all(valid.any(axis=1)):
            raise ValueError("example has no valid bins observed during training")
        return torch.from_numpy(np.where(valid, (values - self.mean) / self.scale, 0.0).astype(np.float32))

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in ("mean", "scale", "count")}

    @classmethod
    def from_dict(cls, obj):
        return cls(*(np.asarray(obj[name]) for name in ("mean", "scale", "count")))


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1:
        raise ValueError("unsupported checkpoint version")
    config = Config.from_dict(checkpoint["config"])
    torch.set_num_threads(config.training.cpu_threads)
    model = RoomCNN(len(config.audio.frequency_hz), config.model)
    model.load_state_dict(checkpoint["model"])
    return checkpoint, config, model, FeatureStats.from_dict(checkpoint["statistics"])


def _require_signature(examples, signature):
    for example in examples:
        if example.signature != signature:
            raise ValueError("dataset mixes calibration modes/versions or differs from the checkpoint; process comparisons separately")


def _eligible(example):
    return (example.pose is not None and example.record["pose"].get("physics_approved")
            and example.feature.calibration in ("none", "complex") and comparable_alignment(example.feature))


def _physics_batch(prediction, matches, examples, config, stats):
    values = []
    attempted = 0
    for row, example in enumerate(examples):
        if not _eligible(example):
            continue
        if attempted == config.training.physics_batch_size:
            break
        attempted += 1
        stats["attempted"] += 1
        vector = matched_geometry(prediction, row, matches[row])
        objective = AcousticObjective(len(matches[row]), example.pose, example.feature, config.audio, config.simulator)
        try:
            values.append(physics_loss(vector, objective))
        except ValueError as exc:
            reason = str(exc)
            stats["failures"][reason] = stats["failures"].get(reason, 0) + 1
        finally:
            stats["simulator_calls"] += objective.calls
            stats["simulator_seconds"] += objective.seconds
            stats["step_retries"] += objective.retries
    if values:
        result = torch.stack(values)
        stats["used"] += len(values)
        stats["db_sum"] += float(result.detach().sum())
        return result.mean()
    return None


def train(index, config, split, output, *, init=None):
    examples = load_dataset(index, config.audio)
    validate_split(split, (example.room_id for example in examples))
    if any(len(ex.shell.footprint) > config.model.max_walls for ex in examples):
        raise ValueError("dataset has a shell exceeding the candidate wall limit")
    signature = examples[0].signature
    _require_signature(examples, signature)
    train_rooms, val_rooms = set(split["train"]), set(split["validation"])
    training = [example for example in examples if example.room_id in train_rooms]
    validation = [example for example in examples if example.room_id in val_rooms]
    settings = config.training
    if settings.physics_weight and (settings.epochs <= settings.physics_warmup_epochs or not any(map(_eligible, training))):
        raise ValueError("nonzero physics weight requires a post-warmup epoch and an eligible training measurement")
    seed = settings.seed
    torch.manual_seed(seed)
    torch.set_num_threads(settings.cpu_threads)
    rng = np.random.default_rng(seed)
    stats = FeatureStats.fit([example.feature for example in training])
    train_input = stats.transform([example.feature for example in training])
    model = RoomCNN(len(config.audio.frequency_hz), config.model)
    learned_rooms = set(train_rooms) | val_rooms
    if init:
        prior, prior_config, prior_model, _ = _load_checkpoint(init)
        torch.set_num_threads(settings.cpu_threads)
        if prior_config.model != config.model or prior_config.audio != config.audio:
            raise ValueError("pretrained checkpoint must share CNN architecture and audio configuration")
        if set(prior["learned_rooms"]) & (val_rooms | set(split["test"])):
            raise ValueError("pretraining saw current held-out rooms")
        learned_rooms.update(prior["learned_rooms"])
        model.load_state_dict(prior_model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    if checkpoint_path.exists():
        raise ValueError(f"{checkpoint_path} already exists; choose a new training output directory")
    history, best = [], None
    global_step = 0
    physics_examples_seen = 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        loss_sums = {}
        physics = {"used": 0, "attempted": 0, "db_sum": 0.0, "simulator_calls": 0, "simulator_seconds": 0.0,
                   "step_retries": 0, "failures": {}}
        weight = settings.physics_weight * float(np.clip((epoch - settings.physics_warmup_epochs) /
                                                        settings.physics_ramp_epochs, 0, 1))
        shuffled = rng.permutation(len(training))
        steps = 0
        for start in range(0, len(training), settings.batch_size):
            rows = shuffled[start:start + settings.batch_size]
            batch = [training[i] for i in rows]
            prediction = model(train_input[rows])
            supervised, parts, matches = supervised_loss(prediction, [ex.shell for ex in batch], config.loss)
            total = supervised
            if weight and global_step % settings.physics_every_steps == 0:
                term = _physics_batch(prediction, matches, batch, config, physics)
                if term is not None:
                    total = total + weight * term
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for name, value in {**parts, "supervised": supervised, "total": total}.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach()) * len(rows)
            steps += 1
            global_step += 1
        physics_examples_seen += physics["used"]
        predicted = list(predictions(model, stats, validation, settings.batch_size))
        ranking, threshold, metrics = tune_threshold(validation, predicted, config)
        entry = {"epoch": epoch, "threshold": threshold, "validation": metrics,
                 "train_losses": {key: value / len(training) for key, value in loss_sums.items()},
                 "physics": {"lambda": weight, "training_examples": len(training), "eligible": sum(map(_eligible, training)),
                             "attempted": physics["attempted"], "used": physics["used"],
                             "coverage": physics["used"] / len(training),
                             "mean_loss_db": physics["db_sum"] / physics["used"] if physics["used"] else None,
                             "simulator_calls": physics["simulator_calls"], "simulator_calls_per_step": physics["simulator_calls"] / steps,
                             "simulator_seconds": physics["simulator_seconds"], "step_retries": physics["step_retries"],
                             "failure_reasons": physics["failures"]}}
        history.append(entry)
        print(json.dumps(entry, allow_nan=False), flush=True)
        if (not settings.physics_weight or physics_examples_seen) and (best is None or ranking > best):
            best = ranking
            torch.save({"version": 1, "config": config.to_dict(), "model": model.state_dict(), "statistics": stats.to_dict(),
                        "threshold": threshold, "split": split, "learned_rooms": sorted(learned_rooms), "epoch": epoch,
                        "physics_examples_seen": physics_examples_seen,
                        "signature": signature, "training_index": str(Path(index).resolve()),
                        "init_checkpoint": str(Path(init).resolve()) if init else None}, checkpoint_path)
    write_json(output / "history.json", history)
    write_json(output / "split.json", split)
    if best is None:
        raise ValueError("no examples received physics gradients; no combined checkpoint was saved (see history.json)")
    return checkpoint_path


def evaluate(index, checkpoint_path, output, *, split="test"):
    checkpoint, config, model, stats = _load_checkpoint(checkpoint_path)
    examples = load_dataset(index, config.audio)
    validate_split(checkpoint["split"], (example.room_id for example in examples))
    selected_rooms = checkpoint["split"][split]
    examples = [example for example in examples if example.room_id in selected_rooms]
    _require_signature(examples, checkpoint["signature"])
    output = Path(output)
    scored = [score(example, pred, checkpoint["threshold"], config.loss)
              for example, pred in zip(examples, predictions(model, stats, examples, config.training.batch_size), strict=True)]
    rows = [entry[0] for entry in scored]
    valid = [i for i, row in enumerate(rows) if row["valid"]]
    invalid = [i for i, row in enumerate(rows) if not row["valid"]]
    representative = {}
    if valid:
        best = max(valid, key=lambda i: rows[i]["footprint_iou_all"])
        worst = min(valid, key=lambda i: rows[i]["footprint_iou_all"])
        representative["best-valid"] = best
        if best != worst:
            representative["lowest-iou-valid"] = worst
    if invalid:
        representative["invalid"] = invalid[0]
    artifacts = {}
    for name, i in representative.items():
        _, result, shell = scored[i]
        geometry, spectrum = f"{name}-geometry.svg", f"{name}-spectrum.svg"
        plot_geometry(output / geometry, result, shell, examples[i].shell)
        failures = plot_spectrum(output / spectrum, examples[i], shell, config.audio, config.simulator)
        artifacts[name] = {"measurement_id": rows[i]["measurement_id"], "geometry_svg": geometry,
                           "spectrum_svg": spectrum, "spectrum_limitations": failures}
    report = {**summarize(rows), "split": split, "room_ids": selected_rooms, "threshold": checkpoint["threshold"],
              "checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"], "measurements": rows,
              "physics_examples_seen": checkpoint["physics_examples_seen"],
              "valid_only": summarize([rows[i] for i in valid]) if valid else None,
              "artifacts": artifacts, "device_config_ids": sorted({row["device_config_id"] for row in rows}),
              "calibration_signature": checkpoint["signature"]}
    write_json(output / "metrics.json", report)
    write_json(output / "predictions.json", [{"measurement_id": row["measurement_id"], **entry[1]}
                                             for row, entry in zip(rows, scored, strict=True)])
    return report


def infer(excitation, recording, checkpoint_path, output, *, excitation_channel, recording_channel,
          excitation_rate=None, recording_rate=None, sweep_start_seconds=None, calibration=None, calibration_root=None):
    checkpoint, config, model, stats = _load_checkpoint(checkpoint_path)
    calibration = calibration if calibration is not None else {"mode": "none"}
    if feature_signature(calibration) != checkpoint["signature"]:
        raise ValueError("inference calibration mode/version differs from training")
    reference = reference_feature(calibration, calibration_root or Path.cwd(), config.audio)
    feature = process_wavs(excitation, recording, config.audio, excitation_channel=excitation_channel,
                           recording_channel=recording_channel, excitation_rate=excitation_rate,
                           recording_rate=recording_rate, sweep_start_seconds=sweep_start_seconds,
                           calibration=calibration["mode"], reference=reference)
    model.eval()
    with torch.inference_mode():
        prediction = model(stats.transform([feature]))
    geometry, shell = decode_prediction({"planes": prediction.planes[0].numpy(),
                    "probabilities": prediction.presence_logits[0].sigmoid().numpy(), "floor": float(prediction.floor[0]),
                    "ceiling": float(prediction.ceiling[0]), "successors": prediction.successor_logits[0].numpy()},
                    checkpoint["threshold"])
    output = Path(output)
    plot_geometry(output / "geometry.svg", geometry, shell)
    write_json(output / "geometry.json", {**geometry, "calibration": calibration,
                                         "alignment_method": feature.alignment_method,
                                         "timing_reference_seconds": feature.timing_reference_seconds})
    return geometry
