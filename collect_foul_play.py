"""Collect policy-training rows from Foul Play MCTS battles."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

from dataset import TrainingDatasetWriter
from map import aggregate_mcts_policy
from trainer_config import collection_config_from_yaml


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--output-path")
    parser.add_argument("--foul-play-dir")
    parser.add_argument("--run-count", type=int)
    parser.add_argument("--search-time-ms", type=int)
    parser.add_argument("--search-parallelism", type=int)
    parser.add_argument("--pokemon-format")
    parser.add_argument("--bot-mode")
    parser.add_argument("--username")
    parser.add_argument("--password")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = collection_config_from_yaml(args.config)
    if not _bool(settings.get("enabled"), True):
        print("collection.enabled is false; nothing to collect")
        return

    _apply_overrides(settings, args)
    foul_play_dir = Path(str(settings.get("foul_play_dir") or "../foul-play")).expanduser()
    if not foul_play_dir.exists():
        raise FileNotFoundError(
            f"Foul Play directory not found: {foul_play_dir}. "
            "Clone https://github.com/pmariglia/foul-play and pass --foul-play-dir."
        )

    sys.path.insert(0, str(foul_play_dir.resolve()))
    asyncio.run(run_collection(settings))


async def run_collection(settings: dict[str, Any]) -> None:
    from config import FoulPlayConfig, init_logging
    from data import all_move_json, pokedex
    from data.mods.apply_mods import apply_mods
    from fp.run_battle import pokemon_battle
    from fp.websocket_client import PSWebsocketClient
    from teams import TeamListIterator, load_team

    _configure_foul_play(settings)
    init_logging(FoulPlayConfig.log_level, FoulPlayConfig.log_to_file)
    apply_mods(FoulPlayConfig.pokemon_format)

    output_path = Path(str(settings.get("output_path") or "training_data/mcts_run.jsonl"))
    writer = TrainingDatasetWriter(
        output_path,
        run_id=_optional_str(settings.get("run_id")),
        run_metadata={
            "source": "foul-play",
            "pokemon_format": FoulPlayConfig.pokemon_format,
            "search_time_ms": FoulPlayConfig.search_time_ms,
            "parallelism": FoulPlayConfig.parallelism,
        },
    )
    current_battle_ids: set[str] = set()
    _install_collecting_pick_move(settings, writer, current_battle_ids)

    ps_websocket_client = await PSWebsocketClient.create(
        FoulPlayConfig.username,
        FoulPlayConfig.password,
        FoulPlayConfig.websocket_uri,
    )
    try:
        FoulPlayConfig.user_id = await ps_websocket_client.login()
        if FoulPlayConfig.avatar is not None:
            await ps_websocket_client.avatar(FoulPlayConfig.avatar)

        team_iterator = (
            None
            if FoulPlayConfig.team_list is None
            else TeamListIterator(FoulPlayConfig.team_list)
        )

        for _ in range(FoulPlayConfig.run_count):
            if FoulPlayConfig.requires_team():
                team_name = (
                    team_iterator.get_next_team()
                    if team_iterator is not None
                    else FoulPlayConfig.team_name
                )
                team_packed, team_dict, _team_file_name = load_team(team_name)
                await ps_websocket_client.update_team(team_packed)
            else:
                team_dict = None

            await _start_battle(FoulPlayConfig, ps_websocket_client)
            current_battle_ids.clear()
            winner = await pokemon_battle(
                ps_websocket_client,
                FoulPlayConfig.pokemon_format,
                team_dict,
            )
            for battle_id in sorted(current_battle_ids):
                writer.write_result(battle_id=battle_id, winner=winner)
    finally:
        await ps_websocket_client.close()

    logger.info(
        "Collected Foul Play MCTS examples at %s; pokedex=%s moves=%s",
        output_path,
        len(pokedex),
        len(all_move_json),
    )


def _install_collecting_pick_move(
    settings: dict[str, Any],
    writer: TrainingDatasetWriter,
    current_battle_ids: set[str],
) -> None:
    import fp.run_battle as run_battle

    strict = _bool(settings.get("strict"), False)
    include_state = _bool(settings.get("include_state"), True)

    async def collecting_async_pick_move(battle):
        battle_copy = deepcopy(battle)
        if not battle_copy.team_preview:
            battle_copy.user.update_from_request_json(battle_copy.request_json)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            chosen_action, mcts_policy = await loop.run_in_executor(
                pool, _find_move_and_policy, battle_copy
            )

        if mcts_policy:
            try:
                record = writer.write_decision(
                    battle=battle_copy,
                    mcts_policy=mcts_policy,
                    chosen_action=chosen_action,
                    strict=strict,
                    include_state=include_state,
                    metadata={
                        "source": "foul-play",
                        "search_time_ms": settings.get("search_time_ms"),
                        "search_parallelism": settings.get("search_parallelism"),
                    },
                )
                current_battle_ids.add(record.battle_id)
            except Exception:
                if strict:
                    raise
                logger.exception("Skipping an unmappable Foul Play MCTS decision")

        battle.user.last_selected_move = run_battle.LastUsedMove(
            battle.user.active.name,
            chosen_action.removesuffix("-tera").removesuffix("-mega"),
            battle.turn,
        )
        return run_battle.format_decision(battle_copy, chosen_action)

    run_battle.async_pick_move = collecting_async_pick_move


def _find_move_and_policy(battle) -> tuple[str, dict[str, float]]:
    import fp.search.main as search_main
    from config import FoulPlayConfig

    battle = deepcopy(battle)
    if battle.team_preview:
        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)

    if battle.battle_type == search_main.BattleType.RANDOM_BATTLE:
        num_battles, search_time_per_battle = search_main.search_time_num_battles_randombattles(
            battle
        )
        battles = search_main.prepare_random_battles(battle, num_battles)
    elif battle.battle_type == search_main.BattleType.BATTLE_FACTORY:
        num_battles, search_time_per_battle = search_main.search_time_num_battles_standard_battle(
            battle
        )
        battles = search_main.prepare_random_battles(battle, num_battles)
    elif battle.battle_type == search_main.BattleType.STANDARD_BATTLE:
        num_battles, search_time_per_battle = search_main.search_time_num_battles_standard_battle(
            battle
        )
        battles = search_main.prepare_battles(battle, num_battles)
    else:
        raise ValueError(f"Unsupported battle type: {battle.battle_type}")

    with ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism) as executor:
        futures = []
        for index, (prepared_battle, chance) in enumerate(battles):
            state = search_main.battle_to_poke_engine_state(prepared_battle).to_string()
            future = executor.submit(
                search_main.get_result_from_mcts,
                state,
                search_time_per_battle,
                index,
            )
            futures.append((future, chance, index))

    mcts_results = [(future.result(), chance, index) for future, chance, index in futures]
    return search_main.select_move_from_mcts_results(mcts_results), aggregate_mcts_policy(
        mcts_results
    )


def _configure_foul_play(settings: dict[str, Any]) -> None:
    from config import BotModes, FoulPlayConfig, SaveReplay

    username = _configured_secret(settings, "ps_username", "ps_username_env")
    if not username:
        raise ValueError(
            "Set collection.ps_username or the environment variable named by "
            "collection.ps_username_env."
        )

    FoulPlayConfig.websocket_uri = str(
        settings.get("websocket_uri") or "wss://sim3.psim.us/showdown/websocket"
    )
    FoulPlayConfig.username = username
    FoulPlayConfig.password = _configured_secret(
        settings, "ps_password", "ps_password_env"
    )
    FoulPlayConfig.avatar = _optional_str(settings.get("ps_avatar"))
    FoulPlayConfig.bot_mode = BotModes[str(settings.get("bot_mode") or "search_ladder")]
    FoulPlayConfig.pokemon_format = str(
        settings.get("pokemon_format") or "gen9randombattle"
    )
    FoulPlayConfig.smogon_stats = _optional_str(settings.get("smogon_stats_format"))
    FoulPlayConfig.search_time_ms = _int(settings.get("search_time_ms"), 100)
    FoulPlayConfig.parallelism = _int(settings.get("search_parallelism"), 1)
    FoulPlayConfig.run_count = _int(settings.get("run_count"), 1)
    FoulPlayConfig.team_name = _optional_str(settings.get("team_name")) or FoulPlayConfig.pokemon_format
    FoulPlayConfig.team_list = _optional_str(settings.get("team_list"))
    FoulPlayConfig.user_to_challenge = _optional_str(settings.get("user_to_challenge"))
    FoulPlayConfig.save_replay = SaveReplay[str(settings.get("save_replay") or "never")]
    FoulPlayConfig.room_name = _optional_str(settings.get("room_name"))
    FoulPlayConfig.log_level = str(settings.get("log_level") or "INFO")
    FoulPlayConfig.log_to_file = _bool(settings.get("log_to_file"), False)
    FoulPlayConfig.validate_config()


async def _start_battle(FoulPlayConfig, ps_websocket_client) -> None:
    if FoulPlayConfig.bot_mode.name == "challenge_user":
        await ps_websocket_client.challenge_user(
            FoulPlayConfig.user_to_challenge,
            FoulPlayConfig.pokemon_format,
        )
        return
    if FoulPlayConfig.bot_mode.name == "accept_challenge":
        await ps_websocket_client.accept_challenge(
            FoulPlayConfig.pokemon_format,
            FoulPlayConfig.room_name,
        )
        return
    if FoulPlayConfig.bot_mode.name == "search_ladder":
        await ps_websocket_client.search_for_match(FoulPlayConfig.pokemon_format)
        return
    raise ValueError(f"Invalid Bot Mode: {FoulPlayConfig.bot_mode}")


def _apply_overrides(settings: dict[str, Any], args: argparse.Namespace) -> None:
    for arg_name, key in (
        ("output_path", "output_path"),
        ("foul_play_dir", "foul_play_dir"),
        ("run_count", "run_count"),
        ("search_time_ms", "search_time_ms"),
        ("search_parallelism", "search_parallelism"),
        ("pokemon_format", "pokemon_format"),
        ("bot_mode", "bot_mode"),
        ("username", "ps_username"),
        ("password", "ps_password"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            settings[key] = value


def _configured_secret(
    settings: dict[str, Any], value_key: str, env_key: str
) -> str | None:
    explicit = _optional_str(settings.get(value_key))
    if explicit:
        return explicit
    env_name = _optional_str(settings.get(env_key))
    if env_name:
        return _optional_str(os.environ.get(env_name))
    return None


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _bool(value: object, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


if __name__ == "__main__":
    main()
