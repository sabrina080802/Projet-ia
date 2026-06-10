"""Command-line entrypoint for the SHARP training pipelines.

Examples:
    # Full local pipeline (extract -> validate -> prepare -> train -> evaluate)
    python run.py local --dataset-id <HUB_DATASET_ID>

    # Run individual local stages
    python run.py local --stage prepare
    python run.py local --stage train

    # Cloud: prepare + upload dataset + register a HUB model
    python run.py cloud setup --dataset-id <HUB_DATASET_ID>

    # Cloud: pull metrics / download best weights after training
    python run.py cloud metrics
    python run.py cloud download --dest runs/cloud/best.pt

    # Cloud: hosted inference on a single image
    python run.py cloud predict path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json

from sharp.logging_utils import get_logger

logger = get_logger("run")


def _run_local(args: argparse.Namespace) -> None:
    from sharp.local import evaluation, extraction, preparation, training, validation
    from sharp.local import pipeline as local_pipeline

    if args.stage == "all":
        result = local_pipeline.run(dataset_id=args.dataset_id, strict=not args.no_strict)
        logger.info("Test metrics: %s", json.dumps(result.metrics, indent=2))
        return

    if args.stage == "extract":
        extraction.extract(dataset_id=args.dataset_id)
    elif args.stage == "validate":
        report = validation.validate(args.raw_dir)
        logger.info("Validation ok=%s", report.ok)
    elif args.stage == "prepare":
        preparation.prepare()
    elif args.stage == "train":
        training.train(args.data_yaml)
    elif args.stage == "evaluate":
        metrics = evaluation.evaluate(args.weights, args.data_yaml)
        logger.info("Test metrics: %s", json.dumps(metrics, indent=2))


def _run_cloud(args: argparse.Namespace) -> None:
    from sharp.cloud import evaluation as cloud_eval
    from sharp.cloud import inference
    from sharp.cloud import pipeline as cloud_pipeline
    from sharp.cloud import train as cloud_train

    if args.action == "setup":
        result = cloud_pipeline.setup(dataset_id=args.dataset_id, reuse_remote=args.reuse_remote)
        logger.info("dataset_id=%s model_id=%s", result.dataset_id, result.model_id)
    elif args.action == "train":
        cloud_train.train_programmatic(model_id=args.model_id)
    elif args.action == "metrics":
        logger.info(json.dumps(cloud_eval.fetch_metrics(model_id=args.model_id), indent=2))
    elif args.action == "download":
        cloud_eval.download_weights(model_id=args.model_id, dest=args.dest)
    elif args.action == "predict":
        result = inference.predict(args.image)
        logger.info("Total fingers: %d", result.total_fingers)
        for det in result.detections:
            logger.info("  %s (%.2f)", det.label, det.confidence)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(description="SHARP training pipelines.")
    sub = parser.add_subparsers(dest="pipeline", required=True)

    local = sub.add_parser("local", help="Run the local training pipeline.")
    local.add_argument(
        "--stage",
        choices=["all", "extract", "validate", "prepare", "train", "evaluate"],
        default="all",
    )
    local.add_argument("--dataset-id", default=None, help="HUB dataset id override.")
    local.add_argument("--raw-dir", default=None, help="Raw dataset dir (validate stage).")
    local.add_argument("--data-yaml", default=None, help="data.yaml path (train/evaluate).")
    local.add_argument("--weights", default=None, help="Weights path (evaluate stage).")
    local.add_argument("--no-strict", action="store_true", help="Ignore validation errors.")
    local.set_defaults(func=_run_local)

    cloud = sub.add_parser("cloud", help="Run the Ultralytics HUB cloud pipeline.")
    cloud.add_argument(
        "action",
        choices=["setup", "train", "metrics", "download", "predict"],
    )
    cloud.add_argument("--dataset-id", default=None, help="HUB dataset id.")
    cloud.add_argument("--model-id", default=None, help="HUB model id.")
    cloud.add_argument(
        "--reuse-remote", action="store_true", help="Train off an existing HUB dataset."
    )
    cloud.add_argument("--dest", default="runs/cloud/best.pt", help="Download destination.")
    cloud.add_argument("image", nargs="?", help="Image path for the predict action.")
    cloud.set_defaults(func=_run_cloud)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected pipeline."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
