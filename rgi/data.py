"""Measured and synthetic dataset construction, provenance, and room-level splits."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import convolve

from .audio import AudioConfig, Feature, load_feature, process_wavs, save_feature, spectral_magnitude, spectrum_from_rir
from .config import Config, read_json, write_json
from .geometry import Shell
from .physics import Pose, simulate_rir


IDENTITY = ("room_id", "measurement_id", "session_id", "device_config_id", "take_id")
DATASET_VERSION = 1


@dataclass
class Example:
    record: dict
    shell: Shell
    feature: Feature
    pose: Pose | None

    @property
    def room_id(self):
        return self.record["room_id"]

    @property
    def signature(self):
        return feature_signature(self.record["calibration"])


def feature_signature(calibration):
    return [calibration["mode"], calibration.get("version") if calibration["mode"] != "none" else None]


def _validate_record(record):
    for key in IDENTITY:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"missing nonempty {key}")
    required = {
        "audio": ("excitation", "recording", "excitation_channel", "recording_channel", "excitation_rate",
                  "recording_rate", "sweep_start_hz", "sweep_end_hz", "sweep_duration_s", "recording_duration_s"),
        "device": ("playback", "recording", "routing", "microphone", "output_volume", "microphone_gain", "enhancements"),
        "pose": ("registration", "receiver", "sources", "gains", "orientation", "lid", "physics_approved"),
        "geometry": ("path", "meters_per_scan_unit", "coordinate_frame", "winding"),
        "conditions": ("furniture", "doors_windows", "temperature_c", "recording_problems"),
    }
    for group, keys in required.items():
        missing = set(keys) - record.get(group, {}).keys()
        if missing:
            raise ValueError(f"missing {group} fields: {', '.join(sorted(missing))}")
    if record["geometry"]["winding"] != "CCW" or not record["geometry"]["meters_per_scan_unit"] > 0:
        raise ValueError("geometry must have known metric scale and CCW footprint order")
    if record["device"]["routing"] not in ("isolated", "coherent"):
        raise ValueError("routing must be isolated or coherent (all active speakers play the same waveform)")
    audio = record["audio"]
    if not (0 <= audio["sweep_start_hz"] < audio["sweep_end_hz"] <= audio["excitation_rate"] / 2
            and 0 < audio["sweep_duration_s"] < audio["recording_duration_s"]):
        raise ValueError("invalid sweep band or audio duration metadata")
    if not record["pose"]["registration"] or not record["geometry"]["coordinate_frame"]:
        raise ValueError("pose registration and geometry coordinate frame must be documented")
    pose = record["pose"]
    if not isinstance(pose["physics_approved"], bool):
        raise ValueError("physics_approved must be a boolean")
    if pose["receiver"] is not None:
        active = sum(gain != 0 for gain in Pose.from_dict(pose).gains)
        if (record["device"]["routing"] == "isolated") != (active == 1):
            raise ValueError("speaker routing conflicts with the number of active sources")
    elif pose["physics_approved"]:
        raise ValueError("physics-approved measurements need a registered pose")
    calibration = record["calibration"]
    if calibration["mode"] != "covered":
        raise ValueError("measured training data requires covered calibration")
    if not all(calibration.get(key) for key in ("id", "method", "settings", "propagation", "version")):
        raise ValueError("calibration requires id, method, settings, propagation assumptions, and version")


def _wav_feature(recording, root, audio, *, calibration="none", reference=None):
    return process_wavs(root / recording["excitation"], root / recording["recording"], audio,
                        excitation_channel=recording["excitation_channel"], recording_channel=recording["recording_channel"],
                        excitation_rate=recording["excitation_rate"], recording_rate=recording["recording_rate"],
                        sweep_start_seconds=recording.get("sweep_start_seconds"), calibration=calibration, reference=reference)


def reference_feature(calibration, root, audio):
    mode = calibration["mode"]
    if mode == "none":
        return None
    if "reference_feature" in calibration:
        reference = load_feature(root / calibration["reference_feature"], audio)
        if reference.calibration != "none":
            raise ValueError("reference must contain an uncorrected transfer or independently characterized device transfer")
        return reference
    if mode in ("magnitude", "covered") and "reference_audio" in calibration:
        return _wav_feature(calibration["reference_audio"], root, audio)
    raise ValueError("complex mode requires a known device reference_feature; magnitude/covered also accept reference_audio")


def _write_dataset(output, config, records, **details):
    write_json(output / "dataset.json", {"version": DATASET_VERSION, "audio": asdict(config.audio),
                                         "measurements": records, **details})


def preprocess(manifest, output, config):
    manifest, output = Path(manifest).resolve(), Path(output)
    contents = read_json(manifest)
    records = contents["measurements"]
    covered_references = contents.get("covered_references", {})
    if not records:
        raise ValueError("manifest contains no measurements")
    ids = set()
    stored = []
    for i, record in enumerate(records):
        try:
            calibration = record.get("calibration", covered_references.get(record.get("device_config_id")))
            if calibration is None:
                raise ValueError("missing covered laptop reference for this device_config_id")
            record = {**record, "calibration": {"mode": "covered", **calibration}}
            _validate_record(record)
            measurement_id = record["measurement_id"]
            if measurement_id in ids:
                raise ValueError("duplicate measurement ID")
            ids.add(measurement_id)
            shell = Shell.from_dict(read_json(manifest.parent / record["geometry"]["path"]))
            if len(shell.footprint) > config.model.max_walls:
                raise ValueError("geometry exceeds configured candidate wall limit")
            mode = record["calibration"]["mode"]
            if "feature" in record:
                feature = load_feature(manifest.parent / record["feature"], config.audio)
                if feature.calibration != mode:
                    raise ValueError("imported feature calibration differs from manifest")
                record = {**record, "source_feature": record["feature"]}
            else:
                reference = reference_feature(record["calibration"], manifest.parent, config.audio)
                feature = _wav_feature(record["audio"], manifest.parent, config.audio, calibration=mode, reference=reference)
                durations = feature.diagnostics
                metadata = record["audio"]
                if abs(durations["recording"]["duration_seconds"] - metadata["recording_duration_s"]) > 1 / metadata["recording_rate"]:
                    raise ValueError("declared recording duration differs from the WAV")
                if metadata["sweep_duration_s"] > durations["excitation"]["duration_seconds"] + 1 / metadata["excitation_rate"]:
                    raise ValueError("declared sweep duration exceeds the excitation WAV")
            path = _save_example(output, i, feature)
            stored.append({**record, "origin": "measured", "feature": path, "shell": shell.to_dict()})
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"measurement {record.get('measurement_id', i)!r}: {exc}") from exc
    _write_dataset(output, config, stored, source_manifest=str(manifest))
    return len(stored)


def _save_example(output, i, feature):
    path = Path("features") / f"{i:06d}.npz"
    (output / path).parent.mkdir(parents=True, exist_ok=True)
    save_feature(output / path, feature)
    return path.as_posix()


def generate_synthetic(output, config: Config):
    output = Path(output)
    settings, audio = config.synthetic, config.audio
    rng = np.random.default_rng(settings.seed)
    records = []
    for room in range(settings.rooms):
        width, depth = rng.uniform(3, 8, 2)
        floor, height = rng.uniform(-0.2, 0.2), rng.uniform(2.3, 3.8)
        jitter = rng.uniform(-settings.absorption_jitter, settings.absorption_jitter)
        absorption = float(np.clip(config.simulator.absorption + jitter, 0, np.nextafter(1.0, 0.0)))
        simulator = replace(config.simulator, absorption=absorption)
        if room % 2 and config.model.max_walls >= 6:
            notch_x, notch_y = rng.uniform(0.48, 0.75, 2) * [width, depth]
            footprint = [[0, 0], [width, 0], [width, notch_y], [notch_x, notch_y], [notch_x, depth], [0, depth]]
        else:
            footprint = [[0, 0], [width, 0], [width, depth], [0, depth]]
        room_id = f"synthetic-{settings.seed}-{room:04d}"
        for take in range(settings.poses):
            # Sample in the common lower-left region, safely clear of the notch and walls.
            mic_xy = rng.uniform(0.19, 0.40, 2) * [width, depth]
            heading = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(heading), np.sin(heading)
            xy = (np.asarray(footprint) - mic_xy) @ np.array([[c, -s], [s, c]])
            shell = Shell(xy, floor, floor + height)
            pose = Pose((0, 0, 1.05), ((0.12, 0.08, 1.08),), (1.0,))
            rir = simulate_rir(shell, pose, audio, simulator)
            gain_db = rng.uniform(-settings.gain_jitter_db, settings.gain_jitter_db)
            coloration = rng.uniform(-settings.coloration_jitter, settings.coloration_jitter)
            if coloration:
                rir = convolve(rir, [coloration, 1.0, coloration])[:audio.rir_samples]
            rir *= 10 ** (gain_db / 20)
            transfer = spectrum_from_rir(rir, audio)
            mask = np.ones(transfer.shape, dtype=bool)
            feature = Feature(audio.frequency_hz, spectral_magnitude(transfer, mask, audio), mask,
                              transfer.real, transfer.imag, impulse_response=rir, alignment_method="marker",
                              timing_reference_seconds=0.0, rir_start_seconds=0.0)
            path = _save_example(output, len(records), feature)
            records.append({"room_id": room_id, "measurement_id": f"{room_id}-{take}", "take_id": str(take),
                            "session_id": f"generation-{settings.seed}", "device_config_id": "synthetic-isolated",
                            "origin": "synthetic", "calibration": {"mode": "none"}, "feature": path,
                            "shell": shell.to_dict(), "pose": {**asdict(pose), "physics_approved": True},
                            "augmentation": {"absorption": absorption, "gain_db": gain_db, "coloration_tap": coloration}})
    _write_dataset(output, config, records, generator={"synthetic": asdict(settings), "simulator": asdict(config.simulator)})
    return len(records)


def load_dataset(index, audio: AudioConfig):
    index = Path(index)
    data = read_json(index)
    if data["version"] != DATASET_VERSION or data["audio"] != asdict(audio):
        raise ValueError("dataset and run must use identical audio configuration and supported dataset version")
    examples = []
    ids = set()
    for row in data["measurements"]:
        if row["measurement_id"] in ids:
            raise ValueError(f"duplicate measurement ID: {row['measurement_id']}")
        ids.add(row["measurement_id"])
        feature = load_feature(index.parent / row["feature"], audio)
        if feature.calibration != row["calibration"]["mode"]:
            raise ValueError(f"calibration mismatch: {row['measurement_id']}")
        pose = row.get("pose")
        examples.append(Example(row, Shell.from_dict(row["shell"]), feature,
                                Pose.from_dict(pose) if pose and pose.get("receiver") is not None else None))
    if not examples:
        raise ValueError("dataset contains no measurements")
    return examples


def split_rooms(room_ids, seed, *, test_rooms=(), val_rooms=()):
    rooms = sorted(set(room_ids))
    if len(rooms) < 3:
        raise ValueError("training requires at least three distinct physical rooms")
    if (set(test_rooms) | set(val_rooms)) - set(rooms) or set(test_rooms) & set(val_rooms):
        raise ValueError("requested validation/test rooms must exist and must not overlap")
    remaining = [room for room in rooms if room not in test_rooms and room not in val_rooms]
    np.random.default_rng(seed).shuffle(remaining)
    count = max(1, len(rooms) // 5)
    test = list(test_rooms) or remaining[:count]
    if not test_rooms:
        remaining = remaining[count:]
    validation = list(val_rooms) or remaining[:count]
    if not val_rooms:
        remaining = remaining[count:]
    split = {"train": sorted(remaining), "validation": sorted(set(validation)), "test": sorted(set(test))}
    validate_split(split, rooms)
    return split


def validate_split(split, room_ids):
    if set(split) != {"train", "validation", "test"} or any(not group for group in split.values()):
        raise ValueError("split requires nonempty train, validation and test room lists")
    assigned = [room for group in split.values() for room in group]
    if len(set(assigned)) != len(assigned) or set(assigned) != set(room_ids):
        raise ValueError("room splits overlap or do not partition dataset room IDs")
