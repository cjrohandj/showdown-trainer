#!/usr/bin/env python3
"""Play a local Showdown battle against an MCTS-controlled bot.

This starts an in-process Pokemon Showdown battle, shows the human player the
public state before each choice, and sends the opposing side's moves through the
same MCTS backend used by the collector.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import collect_trajectory_mcts as ctm
from map import action_slot_to_decision, legal_action_mask
from trainer_config import collection_config_from_yaml


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--pokemon-format")
    parser.add_argument("--generation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("poke-engine", "toy"))
    parser.add_argument("--search-time-ms", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--hypotheses", type=int)
    parser.add_argument("--bridge-script", default="scripts/showdown_bridge.js")
    parser.add_argument("--species-data-path", default=ctm.DEFAULT_SPECIES_DATA_PATH)
    parser.add_argument("--randoms-preset-path")
    parser.add_argument("--randoms-stats-path")
    parser.add_argument("--allow-toy-backend", action="store_true")
    parser.add_argument("--use-fixture", action="store_true")
    parser.add_argument("--human-side", choices=("p1", "p2"), default="p1")
    parser.add_argument("--human-name", default="human")
    parser.add_argument("--bot-name", default="mcts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = collection_config_from_yaml(args.config)
    _apply_overrides(settings, args)
    run_match(settings, args)


def run_match(settings: dict[str, Any], args: argparse.Namespace) -> None:
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
    search_time_ms = _int(
        settings.get("search_time_ms", mcts.get("search_time_ms")),
        75,
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

    catalog = ctm._load_catalog(settings)
    catalog_by_species = ctm._catalog_by_species(catalog)
    species_data_path = str(settings.get("species_data_path") or args.species_data_path)
    species_dex = ctm.SpeciesDex.from_path(species_data_path)
    backend = ctm.make_backend(backend_name, allow_toy_backend=allow_toy_backend)

    human_side = args.human_side
    bot_side = ctm._opponent(human_side)

    p1_team = ctm._sample_team(catalog, rng)
    p2_team = ctm._sample_team(catalog, rng)

    with ctm.ShowdownBridge(Path(args.bridge_script)) as bridge:
        bridge.send(
            {
                "type": "start",
                "formatid": pokemon_format,
                "seed": [rng.randrange(1, 0x10000) for _ in range(4)],
                "p1": {
                    "name": args.human_name if human_side == "p1" else args.bot_name,
                    "team": ctm._showdown_team(p1_team),
                },
                "p2": {
                    "name": args.human_name if human_side == "p2" else args.bot_name,
                    "team": ctm._showdown_team(p2_team),
                },
            }
        )

        tracker = ctm.PublicBattleTracker()
        requests: dict[str, dict[str, Any]] = {}
        pending_human_choice: str | None = None
        pending_bot_choice: str | None = None
        battle_over = False

        while not battle_over:
            event = bridge.read_event()
            event_type = event.get("type")
            if event_type == "update":
                tracker.apply_update(event.get("messages") or [])
            elif event_type == "request":
                player = str(event.get("player"))
                request = _mapping(event.get("request"))
                if request:
                    requests[player] = request
            elif event_type == "end":
                data = _mapping(event.get("data"))
                winner = ctm._winner_to_player(data.get("winner")) or tracker.winner
                print()
                print(f"Battle ended. Winner: {winner or 'unknown'}")
                return

            if human_side not in requests or bot_side not in requests:
                continue

            human_request = requests[human_side]
            bot_request = requests[bot_side]

            if _request_type(human_request) == "teampreview" or _request_type(bot_request) == "teampreview":
                _handle_team_preview(
                    bridge=bridge,
                    human_side=human_side,
                    bot_side=bot_side,
                    human_request=human_request,
                    bot_request=bot_request,
                    human_name=args.human_name,
                    bot_name=args.bot_name,
                )
                requests.clear()
                continue

            human_state = ctm._observed_state_from_request(
                request=human_request,
                tracker=tracker,
                actor=human_side,
                opponent=bot_side,
                battle_id=str(human_request.get("battle_tag") or "battle"),
                pokemon_format=pokemon_format,
                generation=generation,
                species_dex=species_dex,
            )
            human_state["action_mask"] = ctm._action_mask_from_request(human_request, human_state)

            bot_state = ctm._observed_state_from_request(
                request=bot_request,
                tracker=tracker,
                actor=bot_side,
                opponent=human_side,
                battle_id=str(bot_request.get("battle_tag") or "battle"),
                pokemon_format=pokemon_format,
                generation=generation,
                species_dex=species_dex,
            )
            bot_state["action_mask"] = ctm._action_mask_from_request(bot_request, bot_state)

            if pending_bot_choice is None:
                pending_bot_choice = _choose_bot_action(
                    backend=backend,
                    rng=rng,
                    catalog=catalog,
                    catalog_by_species=catalog_by_species,
                    species_dex=species_dex,
                    observed_state=bot_state,
                    actor_request=bot_request,
                    actor=bot_side,
                    duration_ms=search_time_ms,
                    threads=threads,
                    hypotheses=hypotheses,
                    human_side=human_side,
                    bot_side=bot_side,
                )

            if pending_human_choice is None:
                pending_human_choice = _prompt_human_choice(
                    human_state=human_state,
                    human_request=human_request,
                    human_side=human_side,
                    human_name=args.human_name,
                    bot_name=args.bot_name,
                )

            bridge.send({"type": "choice", "player": bot_side, "choice": pending_bot_choice})
            bridge.send({"type": "choice", "player": human_side, "choice": pending_human_choice})
            requests.clear()
            pending_human_choice = None
            pending_bot_choice = None

    raise RuntimeError("Battle bridge closed unexpectedly")


def _handle_team_preview(
    *,
    bridge: ctm.ShowdownBridge,
    human_side: str,
    bot_side: str,
    human_request: Mapping[str, Any],
    bot_request: Mapping[str, Any],
    human_name: str,
    bot_name: str,
) -> None:
    print()
    print("Team preview")
    print(f"{human_name} ({human_side}): {_preview_team(human_request)}")
    print(f"{bot_name} ({bot_side}): {_preview_team(bot_request)}")
    order = input("Enter your team order as a 6-digit permutation (default 123456): ").strip()
    if not order:
        order = "123456"
    if not _is_valid_team_order(order):
        raise ValueError("Team order must be a permutation of 123456")
    bridge.send({"type": "choice", "player": human_side, "choice": f"team {order}"})
    bridge.send({"type": "choice", "player": bot_side, "choice": "team 123456"})


def _choose_bot_action(
    *,
    backend: Any,
    rng: random.Random,
    catalog: Any,
    catalog_by_species: Any,
    species_dex: ctm.SpeciesDex,
    observed_state: Mapping[str, Any],
    actor_request: Mapping[str, Any],
    actor: str,
    duration_ms: int,
    threads: int,
    hypotheses: int,
    human_side: str,
    bot_side: str,
) -> str:
    policy = ctm._search_policy(
        backend=backend,
        rng=rng,
        catalog=catalog,
        catalog_by_species=catalog_by_species,
        species_dex=species_dex,
        observed_state=observed_state,
        actor_request=actor_request,
        actor=actor,
        duration_ms=duration_ms,
        threads=threads,
        hypotheses=hypotheses,
    )
    legal_policy = ctm._filter_policy_to_mask(observed_state, policy)
    chosen_action = max(legal_policy, key=legal_policy.get)
    return ctm._decision_to_showdown_choice(
        observed_state,
        actor_request,
        chosen_action,
    )


def _prompt_human_choice(
    *,
    human_state: Mapping[str, Any],
    human_request: Mapping[str, Any],
    human_side: str,
    human_name: str,
    bot_name: str,
) -> str:
    legal_slots = [
        slot
        for slot, allowed in enumerate(legal_action_mask(human_state))
        if allowed
    ]
    if not legal_slots:
        raise RuntimeError("No legal actions available for the human player")

    options: list[tuple[int, str]] = []
    for index, slot in enumerate(legal_slots, start=1):
        decision = action_slot_to_decision(human_state, slot)
        choice = ctm._decision_to_showdown_choice(human_state, human_request, decision)
        options.append((index, choice))

    print(render_public_state(human_state, human_side=human_side, human_name=human_name, bot_name=bot_name))
    print("Legal actions:")
    for index, choice in options:
        print(f"  {index}. {choice}")

    while True:
        raw = input("Choose an action by number or type a legal Showdown command: ").strip()
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(options):
                return options[picked - 1][1]
        for _, choice in options:
            if raw == choice:
                return choice
        print("Invalid choice. Try one of the listed numbers or commands.")


def render_public_state(
    state: Mapping[str, Any],
    *,
    human_side: str,
    human_name: str,
    bot_name: str,
) -> str:
    user_side = _mapping(state.get("user"))
    opp_side = _mapping(state.get("opponent"))
    lines: list[str] = []
    lines.append("Public battle state")
    lines.append(f"Turn: {state.get('turn')}")
    lines.append(
        f"Weather: {state.get('weather', 'none')} | Field: {state.get('field', 'none')} | "
        f"Trick room: {bool(state.get('trick_room'))} | Gravity: {bool(state.get('gravity'))}"
    )
    lines.append("")
    lines.append(f"{human_name} ({human_side})")
    lines.append(f"  Active: {format_pokemon(_mapping(user_side.get('active')))}")
    lines.append(f"  Side conditions: {format_side_conditions(user_side.get('side_conditions'))}")
    lines.append("  Reserve:")
    for pokemon in _sequence(user_side.get("reserve"), 5):
        if pokemon:
            lines.append(f"    - {format_pokemon(_mapping(pokemon))}")
    lines.append("")
    lines.append(f"{bot_name} ({ctm._opponent(human_side)})")
    lines.append(f"  Active: {format_pokemon(_mapping(opp_side.get('active')))}")
    lines.append(f"  Side conditions: {format_side_conditions(opp_side.get('side_conditions'))}")
    lines.append("  Reserve:")
    for pokemon in _sequence(opp_side.get("reserve"), 5):
        if pokemon:
            lines.append(f"    - {format_pokemon(_mapping(pokemon))}")
    return "\n".join(lines)


def format_pokemon(pokemon: Mapping[str, Any]) -> str:
    if not pokemon:
        return "-"
    name = _pretty_name(pokemon.get("name") or pokemon.get("base_name"))
    hp = _int(pokemon.get("hp"), 0)
    max_hp = _int(pokemon.get("max_hp"), 0)
    hp_text = "0% fnt" if _is_fainted(pokemon) else (f"{hp}/{max_hp}" if max_hp else str(hp))
    parts = [name, hp_text]
    status = _text(pokemon.get("status"))
    if status and status != "none" and status != "fnt":
        parts.append(status)
    boosts = format_boosts(_mapping(pokemon.get("boosts")))
    if boosts:
        parts.append(f"boosts {boosts}")
    ability = _text(pokemon.get("ability"))
    item = _text(pokemon.get("item"))
    if ability:
        parts.append(f"ability {ability}")
    if item:
        parts.append(f"item {item}")
    moves = [move.get("name") for move in _sequence(pokemon.get("moves"), 4) if _mapping(move).get("name")]
    if moves:
        parts.append("moves " + ", ".join(_pretty_move(move) for move in moves))
    return " | ".join(parts)


def format_side_conditions(conditions: object) -> str:
    mapping = _mapping(conditions)
    if not mapping:
        return "none"
    pieces = []
    for key, value in sorted(mapping.items()):
        if value:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces) if pieces else "none"


def format_boosts(boosts: Mapping[str, Any]) -> str:
    labels = {
        "attack": "atk",
        "defense": "def",
        "special-attack": "spa",
        "special-defense": "spd",
        "speed": "spe",
        "accuracy": "acc",
        "evasion": "eva",
    }
    pieces: list[str] = []
    for stat, label in labels.items():
        value = _int(boosts.get(stat), 0)
        if value:
            sign = "+" if value > 0 else ""
            pieces.append(f"{label}{sign}{value}")
    return ", ".join(pieces)


def _is_fainted(pokemon: Mapping[str, Any]) -> bool:
    if pokemon.get("fainted") is not None:
        return bool(pokemon.get("fainted"))
    if pokemon.get("alive") is not None:
        return not bool(pokemon.get("alive"))
    status = _text(pokemon.get("status"))
    if status == "fnt":
        return True
    hp_fraction = pokemon.get("hp_fraction")
    try:
        return hp_fraction is not None and float(hp_fraction) <= 0.0
    except (TypeError, ValueError):
        return False


def _preview_team(request: Mapping[str, Any]) -> str:
    side = _mapping(request.get("side"))
    names = []
    for pokemon in _sequence(side.get("pokemon"), 6):
        pokemon_map = _mapping(pokemon)
        detail = str(pokemon_map.get("details") or "")
        name = detail.split(",")[0].strip() if detail else str(pokemon_map.get("ident") or "unknown")
        names.append(_pretty_name(name))
    return ", ".join(name for name in names if name)


def _is_valid_team_order(order: str) -> bool:
    return len(order) == 6 and sorted(order) == list("123456")


def _request_type(request: Mapping[str, Any]) -> str:
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        return "teampreview"
    if request.get("forceSwitch"):
        return "switch"
    return "move"


def _pretty_move(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def _pretty_name(value: object) -> str:
    text = str(value or "").replace("_", "-").strip()
    if not text:
        return ""
    return "-".join(part[:1].upper() + part[1:] for part in text.split("-") if part)


def _text(value: object) -> str:
    return "" if value in (None, "", "none") else str(value)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sequence(value: object, length: int) -> list[Any]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, (str, bytes, bytearray)):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    return (items + [None] * length)[:length]


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _generation_from_format(pokemon_format: str) -> str:
    text = str(pokemon_format)
    if text.startswith("gen") and len(text) >= 4 and text[3].isdigit():
        return f"gen{text[3]}"
    return "gen9"


def _apply_overrides(settings: dict[str, Any], args: argparse.Namespace) -> None:
    showdex = _mapping(settings.get("showdex"))
    mcts = _mapping(settings.get("mcts"))
    settings["showdex"] = showdex
    settings["mcts"] = mcts

    for arg_name, key in (
        ("pokemon_format", "pokemon_format"),
        ("generation", "generation"),
        ("seed", "seed"),
        ("backend", "backend"),
        ("search_time_ms", "search_time_ms"),
        ("threads", "threads"),
        ("hypotheses", "hypotheses"),
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
    settings["species_data_path"] = args.species_data_path


if __name__ == "__main__":
    main()
