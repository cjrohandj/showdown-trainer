"""Collect policy-training rows from offline Showdex-guided poke-engine MCTS."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from dataset import TrainingDatasetWriter
from map import aggregate_mcts_policy
from offline_mcts import make_backend, sample_offline_position
from showdex_distributions import ShowdexCatalog
from trainer_config import collection_config_from_yaml


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--output-path")
    parser.add_argument("--positions", type=int)
    parser.add_argument("--search-time-ms", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--hypotheses", type=int)
    parser.add_argument("--pokemon-format")
    parser.add_argument("--generation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("poke-engine", "toy", "auto"))
    parser.add_argument("--randoms-preset-path")
    parser.add_argument("--randoms-stats-path")
    parser.add_argument("--use-fixture", action="store_true")
    parser.add_argument(
        "--allow-toy-backend",
        action="store_true",
        help="Allow the deterministic toy backend for smoke tests when poke-engine is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = collection_config_from_yaml(args.config)
    if not _bool(settings.get("enabled"), True):
        print("collection.enabled is false; nothing to collect")
        return

    _apply_overrides(settings, args)
    summary = run_collection(settings)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_collection(settings: dict[str, Any]) -> dict[str, Any]:
    showdex = _mapping(settings.get("showdex"))
    mcts = _mapping(settings.get("mcts"))

    seed = _int(settings.get("seed", mcts.get("seed")), 1337)
    rng = random.Random(seed)
    pokemon_format = str(
        settings.get("pokemon_format")
        or showdex.get("pokemon_format")
        or "gen9randombattle"
    )
    generation = str(settings.get("generation") or _generation_from_format(pokemon_format))
    positions = _int(settings.get("positions", mcts.get("positions")), _int(settings.get("run_count"), 8))
    search_time_ms = _int(
        settings.get("search_time_ms", mcts.get("search_time_ms")),
        50,
    )
    threads = _int(settings.get("threads", mcts.get("threads")), 1)
    hypotheses = _int(
        settings.get("hypotheses", mcts.get("hypotheses_per_position")),
        4,
    )
    backend_name = str(settings.get("backend") or mcts.get("backend") or "poke-engine")
    allow_toy_backend = _bool(
        settings.get("allow_toy_backend", mcts.get("allow_toy_backend")),
        False,
    )
    reveal_opponent_set = _bool(showdex.get("reveal_opponent_set"), False)

    catalog = _load_catalog(settings)
    backend = make_backend(backend_name, allow_toy_backend=allow_toy_backend)

    output_path = Path(str(settings.get("output_path") or "training_data/mcts_run.jsonl"))
    writer = TrainingDatasetWriter(
        output_path,
        run_id=str(settings.get("run_id") or "showdex-poke-engine-mcts"),
        run_metadata={
            "source": "showdex-poke-engine-mcts",
            "backend": backend.name,
            "pokemon_format": pokemon_format,
            "generation": generation,
            "positions": positions,
            "hypotheses_per_position": hypotheses,
            "search_time_ms": search_time_ms,
            "threads": threads,
            "seed": seed,
        },
    )

    written = 0
    skipped = 0
    retries = _int(settings.get("retry_limit", mcts.get("retry_limit")), 10)
    for index in range(positions):
        for attempt in range(retries + 1):
            battle_tag = f"offline-{index:05d}-{attempt:02d}"
            try:
                position = sample_offline_position(
                    catalog,
                    rng,
                    hypotheses=hypotheses,
                    pokemon_format=pokemon_format,
                    generation=generation,
                    battle_tag=battle_tag,
                    reveal_opponent_set=reveal_opponent_set,
                )
                mcts_results = [
                    (
                        backend.search(
                            hypothesis_state,
                            duration_ms=search_time_ms,
                            threads=threads,
                        ),
                        1.0 / len(position.hypothesis_states),
                        hypothesis_index,
                    )
                    for hypothesis_index, hypothesis_state in enumerate(
                        position.hypothesis_states
                    )
                ]
                policy = aggregate_mcts_policy(mcts_results)
                if not policy:
                    raise ValueError("MCTS returned an empty policy")

                chosen_action = max(policy, key=policy.get)
                writer.write_decision(
                    state=position.observed_state,
                    mcts_policy=policy,
                    chosen_action=chosen_action,
                    strict=_bool(settings.get("strict"), False),
                    include_state=_bool(settings.get("include_state"), True),
                    metadata={
                        "source": "showdex-poke-engine-mcts",
                        "backend": backend.name,
                        "search_time_ms": search_time_ms,
                        "threads": threads,
                        "hypotheses_per_position": hypotheses,
                    },
                )
                written += 1
                break
            except Exception as exc:
                if attempt >= retries:
                    skipped += 1
                    logger.warning("Skipping position %s after %s attempts: %s", index, attempt + 1, exc)
                else:
                    logger.debug("Retrying position %s after backend/sample error: %s", index, exc)

    return {
        "output_path": str(output_path),
        "backend": backend.name,
        "positions_requested": positions,
        "examples_written": written,
        "positions_skipped": skipped,
    }


def _load_catalog(settings: dict[str, Any]) -> ShowdexCatalog:
    showdex = _mapping(settings.get("showdex"))
    use_fixture = _bool(settings.get("use_fixture", showdex.get("use_embedded_fixture")), False)
    if use_fixture:
        return ShowdexCatalog.fixture()

    data_dir = Path(str(showdex.get("data_dir") or "showdex_cache"))
    pokemon_format = str(settings.get("pokemon_format") or showdex.get("pokemon_format") or "gen9randombattle")
    randoms_stats_path = (
        settings.get("randoms_stats_path")
        or showdex.get("randoms_stats_path")
        or data_dir / f"{pokemon_format}-stats.json"
    )
    randoms_preset_path = (
        settings.get("randoms_preset_path")
        or showdex.get("randoms_preset_path")
        or data_dir / f"{pokemon_format}.json"
    )
    return ShowdexCatalog.from_files(
        randoms_stats_path=randoms_stats_path,
        randoms_preset_path=randoms_preset_path,
        prefer_stats=_bool(showdex.get("prefer_stats"), True),
    )


def _apply_overrides(settings: dict[str, Any], args: argparse.Namespace) -> None:
    showdex = _mapping(settings.get("showdex"))
    mcts = _mapping(settings.get("mcts"))
    settings["showdex"] = showdex
    settings["mcts"] = mcts

    for arg_name, key in (
        ("output_path", "output_path"),
        ("positions", "positions"),
        ("search_time_ms", "search_time_ms"),
        ("threads", "threads"),
        ("hypotheses", "hypotheses"),
        ("pokemon_format", "pokemon_format"),
        ("generation", "generation"),
        ("seed", "seed"),
        ("backend", "backend"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            settings[key] = value
    if args.randoms_preset_path:
        showdex["randoms_preset_path"] = args.randoms_preset_path
    if args.randoms_stats_path:
        showdex["randoms_stats_path"] = args.randoms_stats_path
    if args.use_fixture:
        showdex["use_embedded_fixture"] = True
    if args.allow_toy_backend:
        settings["allow_toy_backend"] = True


def _generation_from_format(pokemon_format: str) -> str:
    text = str(pokemon_format)
    if text.startswith("gen") and len(text) >= 4 and text[3].isdigit():
        return f"gen{text[3]}"
    return "gen9"


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _bool(value: object, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


if __name__ == "__main__":
    main()
