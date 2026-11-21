"""Command line interface for the room-geometry research pipeline."""

import argparse
from pathlib import Path

from rgi.config import Config, read_json, write_json
from rgi.data import generate_synthetic, preprocess, split_rooms
from rgi.diagnostics import physics_check, pilot
from rgi.pipeline import evaluate, infer, train


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    defaults = commands.add_parser("config", help="write a JSON file containing all defaults")
    defaults.add_argument("output", type=Path)

    measured = commands.add_parser("preprocess", help="process a measurement manifest")
    measured.add_argument("manifest", type=Path)
    measured.add_argument("output", type=Path)
    measured.add_argument("--config", type=Path)

    synthetic = commands.add_parser("synthetic", help="generate synthetic room-shell data")
    synthetic.add_argument("output", type=Path)
    synthetic.add_argument("--config", type=Path)

    training = commands.add_parser("train", help="train or fine-tune and select a checkpoint on validation rooms")
    training.add_argument("dataset", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--config", type=Path)
    training.add_argument("--init", type=Path, help="initialize model weights from a pretrained checkpoint")
    training.add_argument("--split-file", type=Path, help="reuse an exact train/validation/test room partition")
    training.add_argument("--test-room", action="append", default=[], help="force a test room; repeat as needed")
    training.add_argument("--val-room", action="append", default=[], help="force a validation room; repeat as needed")

    evaluation = commands.add_parser("evaluate", help="score a checkpoint on its saved holdout rooms")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--split", choices=("test", "validation"), default="test")

    inference = commands.add_parser("infer", help="infer geometry from an exact excitation and untouched WAV take")
    inference.add_argument("excitation", type=Path)
    inference.add_argument("recording", type=Path)
    inference.add_argument("checkpoint", type=Path)
    inference.add_argument("output", type=Path)
    inference.add_argument("--excitation-channel", type=int, required=True)
    inference.add_argument("--recording-channel", type=int, required=True)
    inference.add_argument("--excitation-rate", type=int)
    inference.add_argument("--recording-rate", type=int)
    inference.add_argument("--sweep-start-seconds", type=float)
    inference.add_argument("--calibration", type=Path, help="JSON calibration object; reference paths are relative to that file")

    probe = commands.add_parser("physics-check", help="run deterministic ISM, finite-difference and pose-sensitivity probes")
    probe.add_argument("output", type=Path)

    recording_pilot = commands.add_parser("pilot", help="compare repeats and furniture changes at one registered pose")
    recording_pilot.add_argument("dataset", type=Path)
    recording_pilot.add_argument("output", type=Path)
    recording_pilot.add_argument("measurement_id", nargs="+")
    recording_pilot.add_argument("--config", type=Path)
    recording_pilot.add_argument("--checkpoint", type=Path)
    recording_pilot.add_argument("--shift-m", type=float, default=0.02)
    return root


def main():
    root = parser()
    args = root.parse_args()
    try:
        if args.command == "config":
            write_json(args.output, Config().to_dict())
        elif args.command == "preprocess":
            print(f"Processed {preprocess(args.manifest, args.output, Config.load(args.config))} measurements")
        elif args.command == "synthetic":
            print(f"Generated {generate_synthetic(args.output, Config.load(args.config))} measurements")
        elif args.command == "train":
            config = Config.load(args.config)
            if args.split_file:
                if args.test_room or args.val_room:
                    raise ValueError("--split-file cannot be combined with explicit validation/test rooms")
                split = read_json(args.split_file)
            else:
                rooms = [row["room_id"] for row in read_json(args.dataset)["measurements"]]
                split = split_rooms(rooms, config.training.seed, test_rooms=args.test_room, val_rooms=args.val_room)
            print(train(args.dataset, config, split, args.output, init=args.init))
        elif args.command == "evaluate":
            report = evaluate(args.dataset, args.checkpoint, args.output, split=args.split)
            print(f"{report['room_count']} {args.split} rooms; metrics: {args.output / 'metrics.json'}")
        elif args.command == "infer":
            result = infer(args.excitation, args.recording, args.checkpoint, args.output,
                           excitation_channel=args.excitation_channel, recording_channel=args.recording_channel,
                           excitation_rate=args.excitation_rate, recording_rate=args.recording_rate,
                           sweep_start_seconds=args.sweep_start_seconds,
                           calibration=read_json(args.calibration) if args.calibration else None,
                           calibration_root=args.calibration.parent if args.calibration else None)
            print("Valid geometry" if result["valid"] else f"Invalid geometry: {result['invalid_reason']}")
        elif args.command == "physics-check":
            physics_check(args.output)
            print(f"Physics checks passed: {args.output}")
        elif args.command == "pilot":
            pilot(args.dataset, args.output, Config.load(args.config), args.measurement_id, args.shift_m, args.checkpoint)
            print(f"Pilot results: {args.output}")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        root.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
