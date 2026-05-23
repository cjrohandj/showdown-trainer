"""YAML configuration helpers for policy MLP training and collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent.
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    if YAML_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyYAML is required for config files. Install it with `pip install -r requirements.txt`."
        ) from YAML_IMPORT_ERROR

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def train_config_from_yaml(
    path: str | Path,
    *,
    data_path: str | None = None,
    output_path: str | None = None,
    resume_checkpoint_path: str | None = None,
    smoke: bool = False,
):
    """Build ``train.TrainConfig`` from a student-style YAML file."""

    from train import TrainConfig

    config = load_config(path)
    model = _mapping(config.get("model"))
    training = _mapping(config.get("training"))

    hidden_sizes = _hidden_sizes(model)
    updates = _optional_int(training.get("updates"))
    eval_every = _int(training.get("eval_every"), TrainConfig.eval_every)
    epochs = _int(training.get("epochs"), TrainConfig.epochs)

    if smoke:
        smoke_updates = _optional_int(training.get("smoke_updates"))
        if smoke_updates is not None:
            updates = smoke_updates
        elif updates is None:
            epochs = _int(training.get("smoke_epochs"), 1)
        eval_every = _int(training.get("smoke_eval_every"), eval_every)

    resolved_data_path = data_path or training.get("data_path")
    if not resolved_data_path:
        raise ValueError("training.data_path is required when no data_path override is supplied")

    return TrainConfig(
        data_path=str(resolved_data_path),
        output_path=str(output_path or training.get("output_path") or TrainConfig.output_path),
        resume_checkpoint_path=(
            resume_checkpoint_path
            if resume_checkpoint_path is not None
            else _optional_str(training.get("resume_checkpoint_path"))
        ),
        epochs=epochs,
        updates=updates,
        eval_every=eval_every,
        batch_size=_int(training.get("batch_size"), TrainConfig.batch_size),
        learning_rate=_float(training.get("learning_rate"), TrainConfig.learning_rate),
        weight_decay=_float(training.get("weight_decay"), TrainConfig.weight_decay),
        validation_split=_float(
            training.get("validation_split"), TrainConfig.validation_split
        ),
        seed=_int(config.get("seed"), TrainConfig.seed),
        torch_num_threads=_optional_int(config.get("torch_num_threads")),
        hidden_sizes=hidden_sizes,
        dropout=_float(model.get("dropout"), TrainConfig.dropout),
        hash_buckets=_int(model.get("hash_buckets"), TrainConfig.hash_buckets),
        num_workers=_int(training.get("num_workers"), TrainConfig.num_workers),
        use_result_join=_bool(training.get("use_result_join"), TrainConfig.use_result_join),
        save_best_only=_bool(training.get("save_best_only"), TrainConfig.save_best_only),
        device=str(config.get("device") or training.get("device") or TrainConfig.device),
        grad_clip_norm=_float(
            training.get("grad_clip_norm"), TrainConfig.grad_clip_norm
        ),
    )


def collection_config_from_yaml(path: str | Path) -> dict[str, Any]:
    """Return the ``collection`` section from a YAML config."""

    return _mapping(load_config(path).get("collection"))


def _hidden_sizes(model: dict[str, Any]) -> tuple[int, ...]:
    if "hidden_sizes" in model and model["hidden_sizes"] is not None:
        value = model["hidden_sizes"]
        if isinstance(value, str):
            return tuple(int(part.strip()) for part in value.split(",") if part.strip())
        return tuple(int(item) for item in value)

    hidden_dim = model.get("hidden_dim")
    num_layers = model.get("num_layers")
    if hidden_dim is not None and num_layers is not None:
        return tuple(int(hidden_dim) for _ in range(int(num_layers)))

    from train import TrainConfig

    return TrainConfig.hidden_sizes


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _float(value: object, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _bool(value: object, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
