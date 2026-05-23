"""Train a small policy MLP to imitate offline MCTS distributions."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from dataset import attach_results, iter_decision_records
from encode import EncoderConfig, encode_battle_state, feature_size
from map import ACTION_SLOTS

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError as exc:  # pragma: no cover - environment dependent.
    torch = None
    nn = None
    DataLoader = None
    random_split = None
    TORCH_IMPORT_ERROR = exc

    class Dataset:
        pass

else:
    TORCH_IMPORT_ERROR = None


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    output_path: str = "checkpoints/policy_mlp.pt"
    resume_checkpoint_path: str | None = None
    epochs: int = 20
    updates: int | None = None
    eval_every: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_split: float = 0.1
    seed: int = 1337
    torch_num_threads: int | None = None
    hidden_sizes: tuple[int, ...] = (1024, 512)
    dropout: float = 0.1
    hash_buckets: int = 4096
    num_workers: int = 0
    use_result_join: bool = True
    save_best_only: bool = False
    device: str = "auto"
    grad_clip_norm: float = 1.0


if nn is not None:

    class PolicyMLP(nn.Module):
        """Simple masked-action policy network."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int = ACTION_SLOTS,
            hidden_sizes: tuple[int, ...] = (1024, 512),
            dropout: float = 0.1,
        ):
            super().__init__()

            layers: list[nn.Module] = []
            previous_dim = input_dim
            for hidden_dim in hidden_sizes:
                layers.extend(
                    [
                        nn.Linear(previous_dim, hidden_dim),
                        nn.ReLU(),
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout),
                    ]
                )
                previous_dim = hidden_dim

            layers.append(nn.Linear(previous_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.net(features)

else:

    class PolicyMLP:
        """Placeholder used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            _require_torch()


class PolicyJsonlDataset(Dataset):
    """Loads JSONL decision rows and encodes them lazily."""

    def __init__(
        self,
        path: str | Path,
        *,
        encoder_config: EncoderConfig,
        use_result_join: bool = True,
    ):
        self.path = Path(path)
        record_iter = attach_results(self.path) if use_result_join else iter_decision_records(self.path)
        self.records = [record for record in record_iter if _is_trainable(record)]
        self.encoder_config = encoder_config

        if not self.records:
            raise ValueError(f"No trainable decision records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        state = dict(record["state"])
        state.setdefault("action_mask", record["action_mask"])
        encoded = encode_battle_state(state, config=self.encoder_config)

        target = _normalize_target(record["mcts_target"])
        action_mask = [1 if value else 0 for value in record["action_mask"][:ACTION_SLOTS]]
        action_mask = (action_mask + [0] * ACTION_SLOTS)[:ACTION_SLOTS]

        return {
            "features": torch.tensor(encoded.features, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "example_id": str(record.get("example_id", "")),
        }


def train(config: TrainConfig) -> dict[str, Any]:
    _require_torch()
    if config.torch_num_threads is not None:
        torch.set_num_threads(config.torch_num_threads)
    _seed_everything(config.seed)

    encoder_config = EncoderConfig(hash_buckets=config.hash_buckets)
    dataset = PolicyJsonlDataset(
        config.data_path,
        encoder_config=encoder_config,
        use_result_join=config.use_result_join,
    )
    train_dataset, validation_dataset = _split_dataset(
        dataset,
        validation_split=config.validation_split,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = _resolve_device(config.device)
    input_dim = feature_size(encoder_config)
    model = PolicyMLP(
        input_dim=input_dim,
        output_dim=ACTION_SLOTS,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_validation_loss = float("inf")
    best_checkpoint: dict[str, Any] | None = None
    latest_checkpoint: dict[str, Any] | None = None
    start_epoch = 1
    start_update = 1
    checkpoint_epoch = 0

    if config.resume_checkpoint_path:
        checkpoint = torch.load(config.resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        start_update = int(checkpoint.get("update", 0) or 0) + 1
        checkpoint_epoch = int(checkpoint.get("epoch", 0) or 0)
        best_checkpoint = checkpoint
        latest_checkpoint = checkpoint
        best_validation_loss = float(
            checkpoint.get("validation_metrics", {}).get("loss", best_validation_loss)
        )

    if config.updates is not None:
        if config.updates <= 0:
            raise ValueError("updates must be positive when supplied")

        final_update = start_update + config.updates - 1
        eval_every = max(1, config.eval_every)
        batch_iter = iter(train_loader)
        running_metrics = {"loss": 0.0, "top1": 0.0, "top3": 0.0}
        running_examples = 0

        for update in range(start_update, final_update + 1):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)

            batch_metrics, batch_size = run_train_batch(
                model,
                batch,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=config.grad_clip_norm,
            )
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key] * batch_size
            running_examples += batch_size

            if update != final_update and update % eval_every != 0:
                continue

            train_metrics = _average_metrics(running_metrics, running_examples)
            running_metrics = {"loss": 0.0, "top1": 0.0, "top3": 0.0}
            running_examples = 0

            validation_metrics = run_epoch(
                model,
                validation_loader,
                device=device,
                optimizer=None,
            )

            print(
                "update={update} "
                "train_loss={train_loss:.4f} train_top1={train_top1:.3f} train_top3={train_top3:.3f} "
                "val_loss={val_loss:.4f} val_top1={val_top1:.3f} val_top3={val_top3:.3f}".format(
                    update=update,
                    train_loss=train_metrics["loss"],
                    train_top1=train_metrics["top1"],
                    train_top3=train_metrics["top3"],
                    val_loss=validation_metrics["loss"],
                    val_top1=validation_metrics["top1"],
                    val_top3=validation_metrics["top3"],
                )
            )

            latest_checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                train_config=config,
                encoder_config=encoder_config,
                epoch=checkpoint_epoch,
                update=update,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                input_dim=input_dim,
            )

            if validation_metrics["loss"] <= best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_checkpoint = latest_checkpoint
    else:
        for epoch in range(start_epoch, start_epoch + config.epochs):
            train_metrics = run_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                grad_clip_norm=config.grad_clip_norm,
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device=device,
                optimizer=None,
            )

            print(
                "epoch={epoch} "
                "train_loss={train_loss:.4f} train_top1={train_top1:.3f} train_top3={train_top3:.3f} "
                "val_loss={val_loss:.4f} val_top1={val_top1:.3f} val_top3={val_top3:.3f}".format(
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    train_top1=train_metrics["top1"],
                    train_top3=train_metrics["top3"],
                    val_loss=validation_metrics["loss"],
                    val_top1=validation_metrics["top1"],
                    val_top3=validation_metrics["top3"],
                )
            )

            latest_checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                train_config=config,
                encoder_config=encoder_config,
                epoch=epoch,
                update=int(latest_checkpoint.get("update", 0) or 0)
                if latest_checkpoint
                else 0,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                input_dim=input_dim,
            )

            if validation_metrics["loss"] <= best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_checkpoint = latest_checkpoint

    checkpoint_to_save = best_checkpoint if config.save_best_only else latest_checkpoint
    if checkpoint_to_save is None:
        raise RuntimeError("Training did not produce a checkpoint")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_to_save, output_path)

    return {
        "output_path": str(output_path),
        "resumed_from": config.resume_checkpoint_path,
        "examples": len(dataset),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "best_validation_loss": best_validation_loss,
        "saved_epoch": checkpoint_to_save["epoch"],
        "saved_update": checkpoint_to_save.get("update"),
        "saved_checkpoint_kind": "best" if config.save_best_only else "latest",
        "feature_size": input_dim,
    }


def run_train_batch(
    model: PolicyMLP,
    batch: Mapping[str, torch.Tensor | list[str]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> tuple[dict[str, float], int]:
    model.train(True)
    features = batch["features"].to(device)
    target = batch["target"].to(device)
    action_mask = batch["action_mask"].to(device)

    logits = model(features)
    loss = masked_soft_cross_entropy(logits, target, action_mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()

    metrics = policy_metrics(logits, target, action_mask)
    metrics["loss"] = float(loss.item())
    return metrics, features.shape[0]


def run_epoch(
    model: PolicyMLP,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            action_mask = batch["action_mask"].to(device)

            logits = model(features)
            loss = masked_soft_cross_entropy(logits, target, action_mask)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                optimizer.step()

            batch_size = features.shape[0]
            metrics = policy_metrics(logits, target, action_mask)
            total_loss += float(loss.item()) * batch_size
            total_top1 += metrics["top1"] * batch_size
            total_top3 += metrics["top3"] * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return {"loss": 0.0, "top1": 0.0, "top3": 0.0}

    return {
        "loss": total_loss / total_examples,
        "top1": total_top1 / total_examples,
        "top3": total_top3 / total_examples,
    }


def masked_soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    safe_mask = _safe_action_mask(action_mask)
    masked_logits = logits.masked_fill(~safe_mask, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    masked_target = target * safe_mask.float()
    target_total = masked_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    masked_target = masked_target / target_total

    return -(masked_target * log_probs).sum(dim=-1).mean()


def policy_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[str, float]:
    safe_mask = _safe_action_mask(action_mask)
    masked_logits = logits.masked_fill(~safe_mask, -1e9)
    target_slot = target.argmax(dim=-1)
    topk = masked_logits.topk(k=min(3, ACTION_SLOTS), dim=-1).indices

    top1 = (topk[:, 0] == target_slot).float().mean().item()
    top3 = (topk == target_slot.unsqueeze(1)).any(dim=-1).float().mean().item()
    return {"top1": top1, "top3": top3}


def build_checkpoint(
    *,
    model: PolicyMLP,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    encoder_config: EncoderConfig,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    input_dim: int,
    update: int | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "update": update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model": {
            "class": "PolicyMLP",
            "input_dim": input_dim,
            "output_dim": ACTION_SLOTS,
            "hidden_sizes": train_config.hidden_sizes,
            "dropout": train_config.dropout,
        },
        "train_config": asdict(train_config),
        "encoder_config": asdict(encoder_config),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }


def load_policy_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PolicyMLP, dict[str, Any]]:
    _require_torch()
    checkpoint = torch.load(path, map_location=map_location)
    model_config = checkpoint["model"]
    model = PolicyMLP(
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        hidden_sizes=tuple(model_config["hidden_sizes"]),
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_policy(
    model: PolicyMLP,
    state: Mapping[str, Any],
    *,
    encoder_config: EncoderConfig | None = None,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Return a masked probability distribution over action slots."""

    _require_torch()
    encoder_config = encoder_config or EncoderConfig()
    encoded = encode_battle_state(state, config=encoder_config)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        features = torch.tensor(encoded.features, dtype=torch.float32, device=device)
        action_mask = torch.tensor(encoded.action_mask, dtype=torch.bool, device=device)
        logits = model(features.unsqueeze(0))[0]
        safe_mask = _safe_action_mask(action_mask.unsqueeze(0))[0]
        logits = logits.masked_fill(~safe_mask, -1e9)
        probs = torch.softmax(logits, dim=-1)

    return probs.cpu().tolist()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", help="Path to JSONL dataset from dataset.py")
    parser.add_argument("--config", help="YAML config path, e.g. configs/student.yaml")
    parser.add_argument("--smoke", action="store_true", help="Use smoke training settings from config")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--resume-checkpoint-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--validation-split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--hidden-sizes", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hash-buckets", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Save the best validation checkpoint instead of the latest continuation checkpoint.",
    )
    parser.add_argument(
        "--no-result-join",
        action="store_true",
        help="Do not join later result rows onto decision rows before training.",
    )
    args = parser.parse_args()

    if args.config:
        from trainer_config import train_config_from_yaml

        config = train_config_from_yaml(
            args.config,
            data_path=args.data_path,
            output_path=args.output_path,
            resume_checkpoint_path=args.resume_checkpoint_path,
            smoke=args.smoke,
        )
        return _apply_cli_overrides(config, args)

    if args.data_path is None:
        parser.error("data_path is required unless --config supplies training.data_path")

    hidden_sizes = tuple(
        int(value.strip())
        for value in (args.hidden_sizes or "1024,512").split(",")
        if value.strip()
    )

    return TrainConfig(
        data_path=args.data_path,
        output_path=args.output_path or TrainConfig.output_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        epochs=args.epochs if args.epochs is not None else TrainConfig.epochs,
        updates=args.updates,
        eval_every=args.eval_every if args.eval_every is not None else TrainConfig.eval_every,
        batch_size=args.batch_size if args.batch_size is not None else TrainConfig.batch_size,
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else TrainConfig.learning_rate
        ),
        weight_decay=(
            args.weight_decay if args.weight_decay is not None else TrainConfig.weight_decay
        ),
        validation_split=(
            args.validation_split
            if args.validation_split is not None
            else TrainConfig.validation_split
        ),
        seed=args.seed if args.seed is not None else TrainConfig.seed,
        torch_num_threads=args.torch_num_threads,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout if args.dropout is not None else TrainConfig.dropout,
        hash_buckets=(
            args.hash_buckets if args.hash_buckets is not None else TrainConfig.hash_buckets
        ),
        num_workers=(
            args.num_workers if args.num_workers is not None else TrainConfig.num_workers
        ),
        use_result_join=not args.no_result_join,
        save_best_only=args.save_best_only,
        device=args.device or TrainConfig.device,
        grad_clip_norm=(
            args.grad_clip_norm
            if args.grad_clip_norm is not None
            else TrainConfig.grad_clip_norm
        ),
    )


def _apply_cli_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    overrides: dict[str, Any] = {}
    for arg_name, field_name in (
        ("epochs", "epochs"),
        ("updates", "updates"),
        ("eval_every", "eval_every"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("validation_split", "validation_split"),
        ("seed", "seed"),
        ("torch_num_threads", "torch_num_threads"),
        ("dropout", "dropout"),
        ("hash_buckets", "hash_buckets"),
        ("num_workers", "num_workers"),
        ("device", "device"),
        ("grad_clip_norm", "grad_clip_norm"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value

    if args.hidden_sizes:
        overrides["hidden_sizes"] = tuple(
            int(value.strip())
            for value in args.hidden_sizes.split(",")
            if value.strip()
        )
    if args.no_result_join:
        overrides["use_result_join"] = False
    if args.save_best_only:
        overrides["save_best_only"] = True

    return replace(config, **overrides)


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


def _is_trainable(record: Mapping[str, Any]) -> bool:
    return (
        isinstance(record.get("state"), Mapping)
        and isinstance(record.get("mcts_target"), list)
        and any(float(value) > 0 for value in record["mcts_target"])
    )


def _normalize_target(target: list[float]) -> list[float]:
    values = [float(value) for value in target[:ACTION_SLOTS]]
    values = (values + [0.0] * ACTION_SLOTS)[:ACTION_SLOTS]
    total = sum(values)
    if total > 0:
        return [value / total for value in values]
    return values


def _average_metrics(metrics: Mapping[str, float], examples: int) -> dict[str, float]:
    if examples <= 0:
        return {"loss": 0.0, "top1": 0.0, "top3": 0.0}
    return {key: float(value) / examples for key, value in metrics.items()}


def _safe_action_mask(action_mask: torch.Tensor) -> torch.Tensor:
    safe_mask = action_mask.clone()
    empty_rows = ~safe_mask.any(dim=-1)
    if empty_rows.any():
        safe_mask[empty_rows] = True
    return safe_mask


def _require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch is required for training. Install it with `pip install -r requirements.txt`."
        ) from TORCH_IMPORT_ERROR


def _split_dataset(
    dataset: PolicyJsonlDataset,
    *,
    validation_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if len(dataset) == 1 or validation_split <= 0:
        return dataset, dataset

    validation_size = max(1, int(round(len(dataset) * validation_split)))
    validation_size = min(validation_size, len(dataset) - 1)
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    return train_dataset, validation_dataset


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
