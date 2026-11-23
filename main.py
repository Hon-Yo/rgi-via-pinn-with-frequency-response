"""Command line interface for the room-geometry research pipeline."""

import argparse
from pathlib import Path

from rgi.audio import process_wavs, save_feature
from rgi.config import Config, read_json, write_json
from rgi.data import generate_synthetic, preprocess, reference_feature, split_rooms
from rgi.diagnostics import physics_check, pilot
from rgi.pipeline import evaluate, infer, train
from rgi.report import plot_scan_shell
from rgi.scan import extract_shell


def _wav_arguments(command):
    command.add_argument("excitation", type=Path)
    command.add_argument("recording", type=Path)
    command.add_argument("--excitation-channel", type=int, required=True)
    command.add_argument("--recording-channel", type=int, required=True)
    command.add_argument("--excitation-rate", type=int)
    command.add_argument("--recording-rate", type=int)
    command.add_argument("--sweep-start-seconds", type=float)
    command.add_argument("--calibration", type=Path, help="JSON calibration object; reference paths are relative to that file")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    defaults = commands.add_parser("config", help="write a JSON file containing all defaults")
    defaults.add_argument("output", type=Path)

    measured = commands.add_parser("preprocess", help="process a measurement manifest")
    measured.add_argument("manifest", type=Path)
    measured.add_argument("output", type=Path)
    measured.add_argument("--config", type=Path)

    scan = commands.add_parser("obj-to-shell", help="extract a room-shell label from a registered OBJ")
    scan.add_argument("obj", type=Path)
    scan.add_argument("output", type=Path, help="geometry JSON output; writes a review SVG beside it")
    scan.add_argument("--meters-per-unit", required=True, type=float)
    scan.add_argument("--up-axis", choices=("x", "y", "z"), default="z")
    scan.add_argument("--registration", type=Path, help="optional JSON 4×4 rigid transform after scaling and up-axis conversion")
    scan.add_argument("--floor", required=True, type=float, help="floor Z in the registered metric frame")
    scan.add_argument("--ceiling", required=True, type=float, help="ceiling Z in the registered metric frame")
    scan.add_argument("--inside", required=True, type=float, nargs=2, metavar=("X", "Y"),
                      help="known clear interior location in registered meters")
    scan.add_argument("--slice-fractions", type=float, nargs="+", default=(0.74, 0.80, 0.86, 0.92, 0.97),
                      help="section heights as fractions of the floor-to-ceiling span")
    scan.add_argument("--simplify-m", type=float, default=0.03, help="footprint simplification tolerance (default: 0.03 m)")
    scan.add_argument("--agreement-m", type=float, default=0.12, help="maximum inter-slice boundary discrepancy (default: 0.12 m)")
    scan.add_argument("--max-gap-m", type=float, default=0.0, help="optional maximum gap bridge length (default: 0, off)")
    scan.add_argument("--pillars-json", type=Path, help="optional JSON array of manually identified rectangular pillars")
    scan.add_argument("--config", type=Path, help="project config (sets the candidate wall limit)")

    feature = commands.add_parser("audio-feature", help="extract one shared-grid NPZ from a WAV pair")
    _wav_arguments(feature)
    feature.add_argument("output", type=Path)
    feature.add_argument("--config", type=Path)

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
    _wav_arguments(inference)
    inference.add_argument("checkpoint", type=Path)
    inference.add_argument("output", type=Path)

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
        elif args.command == "obj-to-shell":
            if args.output.suffix.lower() != ".json":
                raise ValueError("shell output must have the .json extension")
            shell, metadata, traces = extract_shell(
                args.obj, meters_per_unit=args.meters_per_unit, up_axis=args.up_axis, inside=args.inside,
                floor=args.floor, ceiling=args.ceiling, slice_fractions=args.slice_fractions,
                simplify_m=args.simplify_m, agreement_m=args.agreement_m, max_gap_m=args.max_gap_m,
                max_walls=Config.load(args.config).model.max_walls,
                registration=read_json(args.registration) if args.registration else None,
                pillars=read_json(args.pillars_json) if args.pillars_json else (),
            )
            preview = args.output.with_suffix(".svg")
            plot_scan_shell(preview, shell, traces, args.inside)
            write_json(args.output, {**shell.to_dict(), "obj_extraction": metadata})
            print(f"Extracted {len(shell.planes)} wall planes; review {preview} before training")
        elif args.command == "audio-feature":
            audio = Config.load(args.config).audio
            calibration = read_json(args.calibration) if args.calibration else {"mode": "none"}
            reference = reference_feature(calibration, args.calibration.parent if args.calibration else Path.cwd(), audio)
            feature = process_wavs(args.excitation, args.recording, audio, excitation_channel=args.excitation_channel,
                                   recording_channel=args.recording_channel, excitation_rate=args.excitation_rate,
                                   recording_rate=args.recording_rate, sweep_start_seconds=args.sweep_start_seconds,
                                   calibration=calibration["mode"], reference=reference)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_feature(args.output, feature)
            print(f"{feature.valid_mask.sum()}/{len(feature.valid_mask)} valid bins ({feature.calibration}): {args.output}")
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
