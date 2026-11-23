"""Prediction-only decoding, room metrics and diagnostic SVGs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .geometry import decode, footprint_iou, select_order
from .model import match_planes
from .physics import simulated_magnitude


def predictions(model, stats, examples, batch_size):
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            features = [ex.feature for ex in examples[start:start + batch_size]]
            pred = model(stats.transform(features))
            for planes, logits, floor, ceiling, successors in zip(
                    pred.planes.numpy(), pred.presence_logits.numpy(), pred.floor.numpy(), pred.ceiling.numpy(),
                    pred.successor_logits.numpy(), strict=True):
                yield {"planes": planes, "probabilities": torch.from_numpy(logits).sigmoid().numpy(),
                       "floor": float(floor), "ceiling": float(ceiling), "successors": successors}


def decode_prediction(prediction, threshold):
    planes, probabilities = prediction["planes"], prediction["probabilities"]
    slots = np.flatnonzero(probabilities >= threshold).tolist()
    order, shell, reason = None, None, None
    try:
        order = select_order(probabilities, prediction["successors"], threshold)
        shell = decode(planes, order, prediction["floor"], prediction["ceiling"])
    except ValueError as exc:
        reason = str(exc)
    result = {"valid": shell is not None, "invalid_reason": reason, "threshold": float(threshold),
              "candidates": [{"slot": i, "plane": p.tolist(), "presence_probability": float(probabilities[i])}
                             for i, p in enumerate(planes)],
              "selected_slots": slots, "selected_planes": planes[slots].tolist(), "boundary_order": order,
              "predicted_floor": prediction["floor"], "predicted_ceiling": prediction["ceiling"],
              "geometry": shell.to_dict() if shell is not None else None}
    return result, shell


def score(example, prediction, threshold, loss):
    result, shell = decode_prediction(prediction, threshold)
    reference = example.shell
    planes = prediction["planes"]
    slots, refs = match_planes(planes, reference.planes, loss)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(planes[slots, :3] * reference.planes[refs, :3], axis=1), -1, 1)))
    offset = np.abs(planes[slots, 3] - reference.planes[refs, 3])
    selected = set(result["selected_slots"])
    true_positive = len(selected & set(slots))
    iou = footprint_iou(reference, shell) if shell is not None else None
    record = example.record
    return {"measurement_id": record["measurement_id"], "room_id": example.room_id, "origin": record["origin"],
            "device_config_id": record["device_config_id"], "calibration": record["calibration"],
            "pillar_present": bool(reference.pillars), "angle_mae_deg": float(angle.mean()),
            "plane_offset_mae_m": float(offset.mean()), "wall_count_error": abs(len(selected) - len(reference.footprint)),
            "height_error_m": abs(prediction["ceiling"] - prediction["floor"] - (reference.ceiling - reference.floor)),
            "presence_tp": true_positive, "presence_fp": len(selected) - true_positive,
            "presence_fn": len(reference.footprint) - true_positive, "valid": result["valid"],
            "invalid_reason": result["invalid_reason"], "footprint_iou": iou, "footprint_iou_all": iou or 0.0}, result, shell


def _group_metrics(rows):
    result = {key: float(np.mean([row[key] for row in rows])) for key in
              ("angle_mae_deg", "plane_offset_mae_m", "wall_count_error", "height_error_m", "footprint_iou_all")}
    valid = [row["footprint_iou"] for row in rows if row["valid"]]
    tp, fp, fn = (sum(row[f"presence_{key}"] for row in rows) for key in ("tp", "fp", "fn"))
    result.update(takes=len(rows), valid_takes=len(valid), validity_rate=len(valid) / len(rows),
                  footprint_iou_valid=float(np.mean(valid)) if valid else None,
                  presence_precision=tp / (tp + fp) if tp + fp else 0.0, presence_recall=tp / (tp + fn),
                  presence_f1=2 * tp / (2 * tp + fp + fn))
    return result


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["room_id"], []).append(row)
    by_room = {room: _group_metrics(takes) for room, takes in sorted(groups.items())}
    aggregate = {key: float(np.mean([row[key] for row in by_room.values() if row[key] is not None]))
                 if any(row[key] is not None for row in by_room.values()) else None
                 for key in next(iter(by_room.values())) if key not in ("takes", "valid_takes")}
    return {"room_count": len(by_room), "measurement_count": len(rows), "aggregate_room_macro": aggregate,
            "per_room": by_room}


def tune_threshold(examples, predicted, config):
    settings = config.training
    def rank(threshold):
        rows = [score(ex, pred, threshold, config.loss)[0] for ex, pred in zip(examples, predicted, strict=True)]
        metrics = summarize(rows)["aggregate_room_macro"]
        key = (metrics["footprint_iou_all"], metrics["validity_rate"], metrics["presence_f1"],
               -metrics["wall_count_error"], -metrics["angle_mae_deg"], -metrics["plane_offset_mae_m"],
               -metrics["height_error_m"], -abs(threshold - 0.5))
        return key, float(threshold), metrics

    return max((rank(t) for t in np.linspace(settings.threshold_min, settings.threshold_max, settings.threshold_count)),
               key=lambda result: result[0])


def plot_geometry(path, result, shell, reference=None):
    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    top = fig.add_subplot(121)
    view = fig.add_subplot(122, projection="3d")
    for room, label, style in ((reference, "Reference", "--"), (shell, "Prediction", "-")):
        if room is None:
            continue
        xy = np.vstack((room.footprint, room.footprint[0]))
        (line,) = top.plot(*xy.T, style, label=label)
        if label == "Reference":
            for pillar in room.pillars:
                points = pillar.footprint
                top.fill(*points.T, color=line.get_color(), alpha=0.25)
        color = line.get_color()
        for z in (room.floor, room.ceiling):
            view.plot(*xy.T, zs=z, linestyle=style, color=color, label=label if z == room.floor else None)
        for x, y in room.footprint:
            view.plot([x, x], [y, y], [room.floor, room.ceiling], linestyle=style, color=color)
    if shell is None:
        candidates = result["selected_planes"]
        if not candidates:
            candidates = [candidate["plane"] for candidate in sorted(result["candidates"],
                         key=lambda candidate: -candidate["presence_probability"])[:3]]
        radius = max([1.0, *(abs(p[3]) for p in candidates),
                      *(np.max(np.abs(reference.footprint), axis=0) if reference is not None else [])]) * 1.25
        for i, (nx, ny, _, d) in enumerate(candidates):
            normal = np.array([nx, ny])
            segment = -d * normal + np.array([-radius, radius])[:, None] * [-ny, nx]
            top.plot(*segment.T, ":", color="tab:red", label="Candidate plane traces" if i == 0 else None)
        top.set(xlim=(-radius, radius), ylim=(-radius, radius))
        view.text2D(0.0, 0.02, "No finite predicted shell", transform=view.transAxes)
    top.set(xlabel="x (m)", ylabel="y (m)", aspect="equal", title="Top-down")
    view.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="3D shell")
    if shell is not None or reference is not None:
        points = np.vstack([room.footprint for room in (shell, reference) if room is not None])
        z = [value for room in (shell, reference) if room is not None for value in (room.floor, room.ceiling)]
        view.set_box_aspect((*np.ptp(points, axis=0), np.ptp(z)))
    if top.get_legend_handles_labels()[0]:
        top.legend(loc="best")
    fig.suptitle("Valid decode" if shell is not None else f"Invalid decode: {result['invalid_reason']}", wrap=True)
    _save_figure(fig, path)


def plot_scan_shell(path, shell, traces, inside):
    fig, axes = plt.subplots(figsize=(8, 6), layout="constrained")
    labeled = set()
    for _, xy, agrees in traces:
        label = "Agreeing scan slices" if agrees else "Rejected closed slices"
        axes.plot(*xy.T, color="0.5" if agrees else "tab:red", alpha=0.45, linestyle="--",
                  label=label if label not in labeled else None)
        labeled.add(label)
    boundary = np.vstack((shell.footprint, shell.footprint[0]))
    axes.plot(*boundary.T, color="tab:blue", linewidth=2, label="Extracted wall footprint")
    for i, pillar in enumerate(shell.pillars):
        axes.fill(*pillar.footprint.T, color="tab:orange", alpha=0.35, label="Annotated pillar" if i == 0 else None)
    axes.plot(*inside, marker="x", color="tab:green", label="Interior seed")
    axes.set(xlabel="x (m)", ylabel="y (m)", aspect="equal", title="Review extracted room shell against scan slices")
    axes.legend()
    _save_figure(fig, path)


def plot_spectrum(path, example, shell, audio, simulator):
    fig, axes = plt.subplots(figsize=(9, 4), layout="constrained")
    feature = example.feature
    grid, mask = feature.frequency_hz, feature.valid_mask
    axes.plot(grid, np.where(mask, feature.magnitude_db, np.nan), label="Observed target")
    failures = []
    if example.pose is None or not example.record["pose"].get("physics_approved"):
        failures.append("No approved pose for simulation")
    else:
        for room, label in ((shell, "Predicted shell"), (example.shell, "Surveyed/reference shell")):
            if room is None:
                failures.append("No valid predicted shell to simulate")
                continue
            try:
                magnitude = simulated_magnitude(room, example.pose, feature, audio, simulator)
            except ValueError as exc:
                failures.append(f"{label}: {exc}")
            else:
                axes.plot(grid, np.where(mask, magnitude, np.nan), label=f"ISM: {label}", alpha=0.8)
    axes.set(xlabel="Frequency (Hz)", ylabel="Feature (dB)", title="\n".join(dict.fromkeys(failures))
             or "Observed and simulated transfer magnitudes")
    axes.legend()
    _save_figure(fig, path)
    return failures


def _save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
