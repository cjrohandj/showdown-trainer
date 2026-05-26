"""Run distributed full-game trajectory MCTS collection on Modal.

Typical flow:

1. Install and authenticate the Modal CLI locally.
2. Upload the local Showdex cache into the named Modal volume:
   `modal run modal_collect.py::upload_showdex_cache`
3. Launch a sharded trajectory collection run:
   `modal run modal_collect.py::main --shards 16 --games-per-shard 50`
4. Download shards later with the Modal CLI, for example:
   `modal volume get showdown-trainer-trajectory-mcts runs/my-run ./downloaded-runs`

This wrapper keeps the existing collection logic in ``collect_trajectory_mcts.py``
as the single source of truth. Each remote job overrides the config for one
shard and calls ``run_collection(settings, args)``.

The local ``main()`` entrypoint is intentionally fire-and-forget. It submits the
requested shards with ``spawn_map()`` and exits without waiting for them to
finish, which makes it much more tolerant of laptop sleep or network drops.
"""

from __future__ import annotations

import json
import secrets
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import modal


APP_NAME = "showdown-trainer-trajectory-mcts"
VOLUME_NAME = "showdown-trainer-trajectory-mcts"
VOLUME_MOUNT_PATH = "/vol"
REPO_ROOT = "/root/showdown-trainer"
CACHE_SUBDIR = "showdex_cache"
RUNS_SUBDIR = "runs"
DEFAULT_CONFIG_PATH = "configs/student.yaml"
DEFAULT_POKEMON_FORMAT = "gen9randombattle"
DEFAULT_SPECIES_PATH = "showdex_cache/showdown_species.json"
DEFAULT_BRIDGE_SCRIPT = "scripts/showdown_bridge.js"

LOCAL_REPO_ROOT = Path(__file__).resolve().parent

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("nodejs", "npm")
    .pip_install_from_requirements(str(LOCAL_REPO_ROOT / "requirements-collector.txt"))
    .add_local_dir(
        LOCAL_REPO_ROOT,
        remote_path=REPO_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".venv",
            ".venv/**",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            "node_modules",
            "node_modules/**",
            "training_data",
            "training_data/**",
            "checkpoints",
            "checkpoints/**",
            "dist",
            "dist/**",
            "showdex_cache",
            "showdex_cache/**",
        ],
    )
    .workdir(REPO_ROOT)
    .run_commands("npm install")
)

app = modal.App(APP_NAME, image=image)


def _remote_cache_file(filename: str) -> str:
    return str(Path(VOLUME_MOUNT_PATH) / CACHE_SUBDIR / filename)


def _remote_output_file(run_name: str, shard_id: int) -> str:
    return str(
        Path(VOLUME_MOUNT_PATH)
        / RUNS_SUBDIR
        / run_name
        / "shards"
        / f"shard-{shard_id:05d}.jsonl"
    )


def _ensure_remote_repo_on_path() -> None:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


@app.function(
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 6,
)
def run_shard(
    shard_id: int,
    run_name: str,
    games: int,
    seed: int,
    search_time_ms: int,
    hypotheses: int,
    threads: int,
    max_turns: int,
    config_path: str = DEFAULT_CONFIG_PATH,
    pokemon_format: str = DEFAULT_POKEMON_FORMAT,
    bridge_script: str = DEFAULT_BRIDGE_SCRIPT,
    species_filename: str = "showdown_species.json",
) -> dict[str, Any]:
    """Run one full-game trajectory MCTS shard remotely on Modal."""

    volume.reload()
    _ensure_remote_repo_on_path()

    from collect_trajectory_mcts import run_collection
    from trainer_config import collection_config_from_yaml

    settings = collection_config_from_yaml(Path(REPO_ROOT) / config_path)
    showdex = dict(settings.get("showdex") or {})
    settings["showdex"] = showdex

    output_path = Path(_remote_output_file(run_name, shard_id))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings["seed"] = int(seed)
    settings["search_time_ms"] = int(search_time_ms)
    settings["hypotheses"] = int(hypotheses)
    settings["threads"] = int(threads)
    settings["pokemon_format"] = pokemon_format
    settings["output_path"] = str(output_path)
    settings["run_id"] = f"{run_name}-shard-{shard_id:05d}"
    settings["species_data_path"] = _remote_cache_file(species_filename)
    settings["randoms_preset_path"] = _remote_cache_file(f"{pokemon_format}.json")
    settings["randoms_stats_path"] = _remote_cache_file(f"{pokemon_format}-stats.json")

    showdex["data_dir"] = str(Path(VOLUME_MOUNT_PATH) / CACHE_SUBDIR)
    showdex["randoms_preset_path"] = settings["randoms_preset_path"]
    showdex["randoms_stats_path"] = settings["randoms_stats_path"]

    args = Namespace(
        output_path=str(output_path),
        games=int(games),
        max_turns=int(max_turns),
        search_time_ms=int(search_time_ms),
        threads=int(threads),
        hypotheses=int(hypotheses),
        pokemon_format=pokemon_format,
        generation=settings.get("generation"),
        seed=int(seed),
        backend=settings.get("backend"),
        bridge_script=str(Path(REPO_ROOT) / bridge_script),
        species_data_path=settings["species_data_path"],
        randoms_preset_path=settings["randoms_preset_path"],
        randoms_stats_path=settings["randoms_stats_path"],
        allow_toy_backend=bool(settings.get("allow_toy_backend")),
    )

    summary = run_collection(settings, args)
    volume.commit()

    return {
        "shard_id": shard_id,
        "run_name": run_name,
        "seed": seed,
        "games": games,
        **summary,
    }


@app.local_entrypoint()
def upload_showdex_cache(
    pokemon_format: str = DEFAULT_POKEMON_FORMAT,
    preset_path: str = "showdex_cache/gen9randombattle.json",
    stats_path: str = "showdex_cache/gen9randombattle-stats.json",
    species_path: str = DEFAULT_SPECIES_PATH,
) -> None:
    """Upload local Showdex cache files into the shared Modal volume."""

    local_preset = (LOCAL_REPO_ROOT / preset_path).resolve()
    local_stats = (LOCAL_REPO_ROOT / stats_path).resolve()
    local_species = (LOCAL_REPO_ROOT / species_path).resolve()
    if not local_preset.exists():
        raise FileNotFoundError(f"Missing preset JSON: {local_preset}")
    if not local_stats.exists():
        raise FileNotFoundError(f"Missing stats JSON: {local_stats}")
    if not local_species.exists():
        raise FileNotFoundError(f"Missing species JSON: {local_species}")

    with volume.batch_upload(force=True) as batch:
        batch.put_file(local_preset, f"/{CACHE_SUBDIR}/{pokemon_format}.json")
        batch.put_file(local_stats, f"/{CACHE_SUBDIR}/{pokemon_format}-stats.json")
        batch.put_file(local_species, f"/{CACHE_SUBDIR}/showdown_species.json")

    print(
        json.dumps(
            {
                "volume": VOLUME_NAME,
                "pokemon_format": pokemon_format,
                "preset_path": str(local_preset),
                "stats_path": str(local_stats),
                "species_path": str(local_species),
                "remote_prefix": f"/{CACHE_SUBDIR}",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def main(
    run_name: str = "gen9rb-trajectory",
    shards: int = 16,
    games_per_shard: int = 10,
    seed: int = 0,
    search_time_ms: int = 1500,
    hypotheses: int = 4,
    threads: int = 1,
    max_turns: int = 300,
    config_path: str = DEFAULT_CONFIG_PATH,
    pokemon_format: str = DEFAULT_POKEMON_FORMAT,
    bridge_script: str = DEFAULT_BRIDGE_SCRIPT,
) -> None:
    """Submit a sharded full-game trajectory MCTS collection run."""

    if not seed:
        seed = secrets.randbits(31)

    shard_ids = list(range(shards))
    run_names = [run_name] * shards
    games = [games_per_shard] * shards
    seeds = [seed + shard_id for shard_id in shard_ids]
    search_times = [search_time_ms] * shards
    hypothesis_counts = [hypotheses] * shards
    thread_counts = [threads] * shards
    max_turn_counts = [max_turns] * shards
    config_paths = [config_path] * shards
    pokemon_formats = [pokemon_format] * shards
    bridge_scripts = [bridge_script] * shards

    run_shard.spawn_map(
        shard_ids,
        run_names,
        games,
        seeds,
        search_times,
        hypothesis_counts,
        thread_counts,
        max_turn_counts,
        config_paths,
        pokemon_formats,
        bridge_scripts,
    )

    submission = {
        "status": "submitted",
        "run_name": run_name,
        "volume": VOLUME_NAME,
        "base_seed": seed,
        "shards_requested": shards,
        "games_requested": shards * games_per_shard,
        "remote_run_dir": str(Path("/") / RUNS_SUBDIR / run_name),
        "shard_ids": shard_ids,
    }
    print(json.dumps(submission, indent=2, sort_keys=True))
