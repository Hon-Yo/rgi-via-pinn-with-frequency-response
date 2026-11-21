"""Determinism, gradient-step and furnished-room pilot comparisons."""

from dataclasses import replace
from itertools import combinations

import numpy as np

from .audio import AudioConfig, Feature, spectral_magnitude, spectrum_from_rir
from .config import write_json
from .data import load_dataset
from .geometry import Shell, footprint_iou
from .physics import AcousticObjective, Pose, SimulatorConfig, centered_gradient, shell_vector, simulate_rir, simulated_magnitude, spectral_error
from .pipeline import _load_checkpoint, _require_signature
from .report import decode_prediction, predictions


def probe_case():
    audio = AudioConfig(sample_rate=8000, f_min=200, f_max=2000, frequency_step_hz=40, rir_seconds=0.08,
                        taper_seconds=0.01, remove_scalar_level=True)
    settings = SimulatorConfig(max_order=1)
    target = Shell([[0, 0], [4, 0], [4, 3], [0, 3]], 0, 2.7)
    pose = Pose((1.3, 1.4, 1.15), ((1.0, 1.50, 1.2), (1.56, 1.48, 1.1)), (1, 0.9))
    transfer = spectrum_from_rir(simulate_rir(target, pose, audio, settings), audio)
    mask = np.ones_like(transfer, dtype=bool)
    feature = Feature(audio.frequency_hz, spectral_magnitude(transfer, mask, audio), mask,
                      transfer.real, transfer.imag, alignment_method="marker")
    start = shell_vector(target)
    start[3] += 0.025
    return start, target, AcousticObjective(4, pose, feature, audio, settings)


def physics_check(output):
    vector, shell, objective = probe_case()
    first, second = objective(vector), objective(vector)
    steps = []
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        settings = replace(objective.simulator, angle_step=objective.simulator.angle_step * multiplier,
                           distance_step=objective.simulator.distance_step * multiplier)
        gradient = centered_gradient(vector, objective, settings)
        # Take a small step in dimensionless coordinates scaled by each local FD unit.
        units = np.full(len(vector), objective.simulator.distance_step)
        units[:-2:2] = objective.simulator.angle_step
        direction = gradient * units
        after = objective(vector - units * direction / np.linalg.norm(direction))
        steps.append({"angle_step_rad": settings.angle_step, "distance_step_m": settings.distance_step,
                      "gradient_l2": float(np.linalg.norm(gradient)), "loss_after_step_db": after,
                      "descent": after < first})
    report = {"repeat_exact": first == second, "starting_loss_db": first, "steps": steps,
              "simulator_calls": objective.calls, "simulator_seconds": objective.seconds,
              "pose_sensitivity": _pose_sensitivity(shell, objective.pose, objective.feature, objective.audio,
                                                    objective.simulator, 0.02)}
    write_json(output, report)
    if not report["repeat_exact"] or not steps[2]["descent"]:
        raise ValueError("determinism or default finite-difference descent check failed; see report")
    return report


def _pose_sensitivity(shell, pose, feature, audio, simulator, shift_m):
    reference = simulated_magnitude(shell, pose, feature, audio, simulator)
    delta_hz = audio.frequency_step_hz
    result = {"surveyed_sim_vs_observed_db": spectral_error(reference, feature.magnitude_db, feature.valid_mask, delta_hz),
              "translation_m": shift_m, "shifts": {}}
    for i, axis in enumerate("xyz"):
        for sign in (-1, 1):
            offset = np.eye(3)[i] * (sign * shift_m)
            moved = Pose(tuple(np.asarray(pose.receiver) + offset),
                         tuple(tuple(np.asarray(src) + offset) for src in pose.sources), pose.gains)
            key = f"{axis}{sign:+d}"
            try:
                shifted = simulated_magnitude(shell, moved, feature, audio, simulator)
                result["shifts"][key] = {"sim_vs_nominal_db": spectral_error(shifted, reference, feature.valid_mask, delta_hz)}
            except ValueError as exc:
                result["shifts"][key] = {"unavailable": str(exc)}
    return result


def pilot(index, output, config, measurement_ids, shift_m, checkpoint=None):
    if shift_m <= 0:
        raise ValueError("pose perturbation must be positive")
    requested = set(measurement_ids)
    examples = [ex for ex in load_dataset(index, config.audio) if ex.record["measurement_id"] in requested]
    if len(examples) < 2 or len(examples) != len(requested):
        raise ValueError("select at least two distinct existing measurements")
    first = examples[0]
    for ex in examples:
        if (ex.room_id != first.room_id or ex.record["device_config_id"] != first.record["device_config_id"]
                or ex.pose != first.pose or ex.signature != first.signature
                or ex.feature.alignment_method != first.feature.alignment_method):
            raise ValueError("pilot comparisons must share a room, declared pose, device config, alignment and calibration")
    if checkpoint:
        state, saved, model, stats = _load_checkpoint(checkpoint)
        if saved.audio != config.audio:
            raise ValueError("pilot and checkpoint audio configurations differ")
        _require_signature(examples, state["signature"])
        decoded = [decode_prediction(pred, state["threshold"]) for pred in predictions(model, stats, examples, 8)]
    else:
        decoded = None
    rows = []
    for i, j in combinations(range(len(examples)), 2):
        a, b = examples[i], examples[j]
        mask = a.feature.valid_mask & b.feature.valid_mask
        row = {"pair": [ex.record["measurement_id"] for ex in (a, b)],
               "same_furniture": a.record.get("conditions", {}).get("furniture") == b.record.get("conditions", {}).get("furniture"),
               "common_bins": int(mask.sum()), "spectral_change_db": spectral_error(a.feature.magnitude_db,
                        b.feature.magnitude_db, mask, config.audio.frequency_step_hz) if mask.any() else None}
        if decoded is not None:
            (p, left), (q, right) = decoded[i], decoded[j]
            row.update(predicted_footprint_iou=footprint_iou(left, right) if left is not None and right is not None else None,
                       invalid_reasons=[p["invalid_reason"], q["invalid_reason"]])
        rows.append(row)
    summary = {}
    for group, same in (("repeat", True), ("furniture_change", False)):
        changes = [row["spectral_change_db"] for row in rows if row["same_furniture"] == same
                   and row["spectral_change_db"] is not None]
        summary[group] = {"usable_pairs": len(changes), "median_change_db": float(np.median(changes)) if changes else None}
    sensitivity = {}
    for ex in examples:
        identifier = ex.record["measurement_id"]
        if ex.pose is None or not ex.record["pose"].get("physics_approved"):
            sensitivity[identifier] = {"unavailable": "no approved pose"}
            continue
        try:
            sensitivity[identifier] = _pose_sensitivity(ex.shell, ex.pose, ex.feature, config.audio, config.simulator, shift_m)
        except ValueError as exc:
            sensitivity[identifier] = {"unavailable": str(exc)}
    report = {"room_id": first.room_id, "device_config_id": first.record["device_config_id"],
              "calibration_signature": first.signature, "pair_summary": summary, "pairs": rows,
              "pose_sensitivity": sensitivity}
    write_json(output, report)
    return report
