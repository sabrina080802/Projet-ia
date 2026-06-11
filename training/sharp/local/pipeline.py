"""Orchestrator for the local pipeline.

Chains the five stages end to end: extraction -> validation -> preparation ->
training -> evaluation. Each stage stays in its own module so responsibilities
remain separated; this file only wires them together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sharp.local import evaluation, extraction, preparation, training, validation
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Artifacts produced by a full local pipeline run."""

    data_yaml: Path
    best_weights: Path
    metrics: dict[str, float]


def run(dataset: str | None = None, strict: bool = True) -> PipelineResult:
    """Execute the full local training pipeline.

    Args:
        dataset: Optional Platform dataset slug, ul:// URI or local dir override
            for the extraction stage.
        strict: When True, abort if validation finds any problem.

    Returns:
        A :class:`PipelineResult` with the dataset, weights and test metrics.

    Raises:
        RuntimeError: If ``strict`` is True and validation fails.
    """
    logger.info("== Stage 1/5: extraction ==")
    raw_dir = extraction.extract(dataset=dataset)

    logger.info("== Stage 2/5: validation ==")
    report = validation.validate(raw_dir)
    if not report.ok:
        logger.warning(
            "Validation found %d corrupt images and %d invalid labels",
            len(report.corrupt_images),
            len(report.invalid_labels),
        )
        if strict:
            raise RuntimeError("Validation failed; re-run with strict=False to ignore.")

    logger.info("== Stage 3/5: preparation ==")
    data_yaml = preparation.prepare(raw_dir=raw_dir)

    logger.info("== Stage 4/5: training ==")
    best_weights = training.train(data_yaml)

    logger.info("== Stage 5/5: evaluation ==")
    metrics = evaluation.evaluate(best_weights, data_yaml)

    logger.info("Local pipeline complete. Best weights: %s", best_weights)
    return PipelineResult(data_yaml=data_yaml, best_weights=best_weights, metrics=metrics)
