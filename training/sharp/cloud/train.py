"""Cloud stage - create a Platform project/model and launch training.

The Ultralytics Platform offers two ways to run a job; this module wraps both:

* :func:`train_cloud` dispatches a real **cloud-GPU** run through the REST API
  (``POST /api/training/start``). It needs a project + model created first
  (:func:`ensure_project` / :func:`create_model`, done by ``cloud setup``) and a
  paid GPU plan.
* :func:`train_local` runs training **on this machine** via the ``ultralytics``
  package, streaming metrics and the final weights to the Platform dashboard
  (free, no subscription). Passing ``project=<username>/<project>`` is what makes
  Ultralytics push the run to the Platform and create/target the model.

Both train the configured base model on the ``ul://`` dataset.
"""

from __future__ import annotations

from typing import Any

from config import settings
from sharp import platform
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def _model_id(value: dict[str, Any]) -> str | None:
    """Best-effort extraction of an id from a Platform create response.

    The exact envelope is not documented, so probe the common shapes
    (``{"id": ...}``, ``{"modelId": ...}``, ``{"model": {"id": ...}}`` ...).
    """
    for key in ("modelId", "projectId", "id", "_id"):
        if isinstance(value.get(key), str):
            return value[key]
    for key in ("model", "project", "data"):
        nested = value.get(key)
        if isinstance(nested, dict):
            for inner in ("id", "_id", "modelId", "projectId"):
                if isinstance(nested.get(inner), str):
                    return nested[inner]
    return None


def ensure_project(name: str | None = None) -> str:
    """Return the Platform project id, creating the project if needed.

    Args:
        name: Project slug/name; defaults to ``settings.ultralytics.project``.

    Returns:
        The project id (reused from ``settings.ultralytics.project_id`` when set).
    """
    if settings.ultralytics.project_id:
        return settings.ultralytics.project_id

    slug = name or settings.ultralytics.project
    client = platform.platform_client()
    response = client.post(
        "projects",
        json={"name": slug, "slug": slug, "description": "SHARP finger detection"},
    )
    project_id = _model_id(response)
    if not project_id:
        raise RuntimeError(f"Could not read a project id from the Platform response: {response}")
    logger.info("Created Platform project %s (id=%s)", slug, project_id)
    return project_id


def create_model(project_id: str | None = None, name: str | None = None) -> str:
    """Create a model entry under a project and return its id.

    The model starts empty; :func:`train_cloud` (or a streamed
    :func:`train_local` run) fills it with weights and metrics.

    Args:
        project_id: Target project; defaults to :func:`ensure_project`.
        name: Model slug/name; defaults to ``settings.ultralytics.model``.

    Returns:
        The new model id.
    """
    client = platform.platform_client()
    project_id = project_id or ensure_project()
    slug = name or settings.ultralytics.model
    response = client.post("models", json={"projectId": project_id, "name": slug, "slug": slug})
    model_id = _model_id(response)
    if not model_id:
        raise RuntimeError(f"Could not read a model id from the Platform response: {response}")
    logger.info("Created Platform model %s (id=%s)", slug, model_id)
    return model_id


def train_cloud(
    model_id: str | None = None,
    project_id: str | None = None,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Start a cloud-GPU training run via REST (requires a paid GPU plan).

    Args:
        model_id: Model to train; defaults to ``settings.ultralytics.model_id``.
        project_id: Owning project; defaults to ``settings.ultralytics.project_id``.
        gpu_type: GPU slug (e.g. ``rtx-pro-6000``, ``a100-80gb-sxm``); defaults to
            ``settings.ultralytics.gpu_type``.

    Returns:
        The raw Platform response for the submitted job.

    Raises:
        RuntimeError: If the model/project ids are missing (run ``cloud setup``).
    """
    model_id = model_id or settings.ultralytics.model_id
    project_id = project_id or settings.ultralytics.project_id
    if not model_id or not project_id:
        raise RuntimeError(
            "train_cloud needs ultralytics.model_id and ultralytics.project_id. "
            "Run `python run.py cloud setup` first (or fill them in settings.toml)."
        )

    body = {
        "modelId": model_id,
        "projectId": project_id,
        "gpuType": gpu_type or settings.ultralytics.gpu_type,
        "trainArgs": {
            "model": settings.ultralytics.base_model,
            "data": platform.dataset_uri(),
            "epochs": settings.train.epochs,
            "imgsz": settings.train.imgsz,
            "batch": settings.train.batch,
        },
    }
    logger.info(
        "Starting Platform cloud-GPU training (gpu=%s) for model %s", body["gpuType"], model_id
    )
    response = platform.platform_client().post("training/start", json=body)
    logger.info(
        "Cloud job submitted. Monitor it on the dashboard or with `cloud metrics --model-id %s`.",
        model_id,
    )
    return response


def train_local() -> str:
    """Train on this machine while streaming metrics/weights to the Platform.

    Uses ``model.train(project=<username>/<project>, ...)`` so the run is pushed
    to the Platform and the model is created/updated there automatically. The
    ``ul://`` dataset is downloaded locally on first use.

    Returns:
        The ``ul://`` URI of the resulting model.
    """
    from ultralytics import YOLO

    platform.configure_ultralytics_auth()
    project = platform.project_ref()
    base = settings.ultralytics.base_model
    logger.info("Training %s locally, streaming to Platform project %s", base, project)

    model = YOLO(base)
    model.train(
        data=platform.dataset_uri(),
        epochs=settings.train.epochs,
        imgsz=settings.train.imgsz,
        batch=settings.train.batch,
        patience=settings.train.patience,
        project=project,
        name=settings.ultralytics.model,
    )
    model_uri = platform.model_uri()
    logger.info("Training finished; model available at %s", model_uri)
    return model_uri
