"""Collect full-game Showdown trajectories with MCTS policy targets."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset import TrainingDatasetWriter
from map import (
    ACTION_SLOTS,
    ActionMappingError,
    aggregate_mcts_policy,
    decision_to_action_slot,
)
from offline_mcts import make_backend
from showdex_distributions import (
    DEFAULT_IVS,
    DEFAULT_RANDOMS_EVS,
    STAT_ORDER,
    PokemonDistribution,
    SampledPokemon,
    ShowdexCatalog,
    normalize_id,
)
from trainer_config import collection_config_from_yaml


logger = logging.getLogger(__name__)

SHOWDOWN_STAT_KEYS = {
    "hp": "hp",
    "attack": "atk",
    "defense": "def",
    "special-attack": "spa",
    "special-defense": "spd",
    "speed": "spe",
}
STAT_ALIASES = {
    "atk": "attack",
    "def": "defense",
    "spa": "special-attack",
    "spd": "special-defense",
    "spe": "speed",
    "hp": "hp",
}
PLAYER_IDS = ("p1", "p2")
DEFAULT_SPECIES_DATA_PATH = "showdex_cache/showdown_species.json"


@dataclass(frozen=True)
class SpeciesInfo:
    id: str
    name: str
    types: tuple[str, str]
    base_stats: dict[str, int]
    weight_kg: float = 0.0


class SpeciesDex:
    def __init__(self, entries: Mapping[str, SpeciesInfo]):
        self.entries = dict(entries)

    @classmethod
    def from_path(cls, path: str | Path | None) -> "SpeciesDex":
        if not path:
            return cls({})
        data_path = Path(path)
        if not data_path.exists():
            logger.warning("Species metadata cache not found: %s", data_path)
            return cls({})
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        entries: dict[str, SpeciesInfo] = {}
        for key, value in _mapping(raw).items():
            row = _mapping(value)
            species_id = normalize_id(row.get("id") or key)
            types = [
                normalize_id(item) or "typeless"
                for item in _sequence(row.get("types"), 2)
                if item
            ]
            types = (types + ["typeless", "typeless"])[:2]
            base_stats = _normalize_base_stats(row.get("base_stats") or row.get("baseStats"))
            entries[species_id] = SpeciesInfo(
                id=species_id,
                name=str(row.get("name") or species_id),
                types=(types[0], types[1]),
                base_stats=base_stats,
                weight_kg=float(row.get("weight_kg") or row.get("weightkg") or 0.0),
            )
        return cls(entries)

    def get(self, species: object) -> SpeciesInfo | None:
        species_id = normalize_id(species)
        if species_id in self.entries:
            return self.entries[species_id]
        return self.entries.get(_base_species_id(species_id))

    def types(self, species: object) -> list[str]:
        info = self.get(species)
        return list(info.types) if info else ["typeless", "typeless"]

    def base_stats(self, species: object) -> dict[str, int]:
        info = self.get(species)
        return dict(info.base_stats) if info else _default_stats()

    def weight_kg(self, species: object) -> float:
        info = self.get(species)
        return float(info.weight_kg) if info else 0.0


@dataclass
class VisiblePokemon:
    species: str = ""
    level: int = 100
    hp_fraction: float = 1.0
    status: str = "none"
    alive: bool = True
    moves: set[str] = field(default_factory=set)
    item: str = ""
    ability: str = ""
    boosts: dict[str, int] = field(default_factory=dict)


@dataclass
class PublicBattleTracker:
    turn: int = 1
    weather: str = "none"
    field_condition: str = "none"
    side_conditions: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"p1": {}, "p2": {}}
    )
    active: dict[str, VisiblePokemon] = field(default_factory=dict)
    seen: dict[str, dict[str, VisiblePokemon]] = field(
        default_factory=lambda: {"p1": {}, "p2": {}}
    )
    last_used_moves: dict[str, str] = field(default_factory=dict)
    winner: str | None = None

    def apply_update(self, messages: Iterable[str]) -> None:
        for line in messages:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            tag = parts[1]
            if tag == "turn" and len(parts) > 2:
                self.turn = _int(parts[2], self.turn)
            elif tag == "win" and len(parts) > 2:
                self.winner = _winner_to_player(parts[2])
            elif tag == "tie":
                self.winner = None
            elif tag in {"switch", "drag", "replace"} and len(parts) >= 5:
                player = _player_from_ident(parts[2])
                pokemon = _visible_from_details(parts[3])
                _apply_condition(pokemon, parts[4])
                self.active[player] = pokemon
                if pokemon.species:
                    self.seen[player][pokemon.species] = pokemon
            elif tag == "poke" and len(parts) >= 4:
                player = parts[2]
                pokemon = _visible_from_details(parts[3])
                if pokemon.species:
                    self.seen[player][pokemon.species] = pokemon
            elif tag == "move" and len(parts) >= 4:
                player = _player_from_ident(parts[2])
                move = normalize_id(parts[3])
                if player and move:
                    self.last_used_moves[player] = move
                if player in self.active and move:
                    self.active[player].moves.add(move)
            elif tag == "faint" and len(parts) >= 3:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].alive = False
                    self.active[player].hp_fraction = 0.0
            elif tag in {"-damage", "-heal", "-sethp"} and len(parts) >= 4:
                player = _player_from_ident(parts[2])
                pokemon = self._pokemon_for_ident(player, parts[2])
                if pokemon is not None:
                    _apply_condition(pokemon, parts[3])
            elif tag == "-status" and len(parts) >= 4:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].status = normalize_id(parts[3]) or "none"
            elif tag == "-curestatus" and len(parts) >= 3:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].status = "none"
            elif tag in {"-boost", "-unboost", "-setboost"} and len(parts) >= 5:
                player = _player_from_ident(parts[2])
                if player not in self.active:
                    continue
                stat = STAT_ALIASES.get(normalize_id(parts[3]), normalize_id(parts[3]))
                amount = _int(parts[4], 0)
                if tag == "-boost":
                    self.active[player].boosts[stat] = self.active[player].boosts.get(stat, 0) + amount
                elif tag == "-unboost":
                    self.active[player].boosts[stat] = self.active[player].boosts.get(stat, 0) - amount
                else:
                    self.active[player].boosts[stat] = amount
            elif tag == "-clearboost" and len(parts) >= 3:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].boosts.clear()
            elif tag == "-weather" and len(parts) >= 3:
                self.weather = normalize_id(parts[2]) or "none"
            elif tag in {"-fieldstart", "-fieldend"} and len(parts) >= 3:
                self.field_condition = normalize_id(parts[2]) if tag == "-fieldstart" else "none"
            elif tag in {"-sidestart", "-sideend"} and len(parts) >= 4:
                side = _player_from_side(parts[2])
                condition = normalize_id(parts[3])
                if not side or not condition:
                    continue
                if tag == "-sidestart":
                    self.side_conditions[side][condition] = self.side_conditions[side].get(condition, 0) + 1
                else:
                    self.side_conditions[side].pop(condition, None)
            elif tag == "-item" and len(parts) >= 4:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].item = normalize_id(parts[3])
            elif tag == "-enditem" and len(parts) >= 3:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].item = ""
            elif tag == "-ability" and len(parts) >= 4:
                player = _player_from_ident(parts[2])
                if player in self.active:
                    self.active[player].ability = normalize_id(parts[3])

    def _pokemon_for_ident(self, player: str, ident: object) -> VisiblePokemon | None:
        if player not in PLAYER_IDS:
            return None
        species = _species_from_ident(ident)
        active = self.active.get(player)
        if active is not None and (
            not species or normalize_id(active.species) == species
        ):
            return active
        return self.seen[player].get(species)


class ShowdownBridge:
    def __init__(self, script_path: Path):
        self.script_path = script_path
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "ShowdownBridge":
        self.process = subprocess.Popen(
            ["node", str(self.script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.send({"type": "stop"})
            except Exception:
                pass
            self.process.terminate()

    def send(self, command: Mapping[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("Showdown bridge is not running")
        self.process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def read_event(self) -> dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise RuntimeError("Showdown bridge is not running")
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"Showdown bridge exited unexpectedly: {stderr.strip()}")
        event = json.loads(line)
        if event.get("type") == "error":
            raise RuntimeError(f"Showdown bridge error: {event}")
        return event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--output-path", default="training_data/trajectory_mcts.jsonl")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--search-time-ms", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--hypotheses", type=int)
    parser.add_argument("--pokemon-format")
    parser.add_argument("--generation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("poke-engine", "toy"), default=None)
    parser.add_argument("--bridge-script", default="scripts/showdown_bridge.js")
    parser.add_argument("--species-data-path", default=DEFAULT_SPECIES_DATA_PATH)
    parser.add_argument("--randoms-preset-path")
    parser.add_argument("--randoms-stats-path")
    parser.add_argument("--allow-toy-backend", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = collection_config_from_yaml(args.config)
    _apply_overrides(settings, args)
    summary = run_collection(settings, args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_collection(settings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    showdex = _mapping(settings.get("showdex"))
    mcts = _mapping(settings.get("mcts"))
    seed = _resolve_run_seed(settings, args, mcts)
    settings["seed"] = seed
    rng = random.Random(seed)
    logger.info("Using run seed %s", seed)
    pokemon_format = str(settings.get("pokemon_format") or showdex.get("pokemon_format") or "gen9randombattle")
    generation = str(settings.get("generation") or _generation_from_format(pokemon_format))
    search_time_ms = _int(settings.get("search_time_ms", mcts.get("search_time_ms")), 75)
    threads = _int(settings.get("threads", mcts.get("threads")), 1)
    hypotheses = _int(settings.get("hypotheses", mcts.get("hypotheses_per_position")), 4)
    backend_name = str(settings.get("backend") or mcts.get("backend") or "poke-engine")

    catalog = _load_catalog(settings)
    catalog_by_species = _catalog_by_species(catalog)
    species_data_path = str(settings.get("species_data_path") or args.species_data_path)
    species_dex = SpeciesDex.from_path(species_data_path)
    backend = make_backend(
        backend_name,
        allow_toy_backend=bool(args.allow_toy_backend or settings.get("allow_toy_backend")),
    )
    writer = TrainingDatasetWriter(
        args.output_path,
        run_id=str(settings.get("run_id") or "showdown-trajectory-mcts"),
        run_metadata={
            "source": "showdown-trajectory-mcts",
            "backend": backend.name,
            "pokemon_format": pokemon_format,
            "generation": generation,
            "games": args.games,
            "hypotheses_per_decision": hypotheses,
            "search_time_ms": search_time_ms,
            "threads": threads,
            "seed": seed,
            "species_data_path": species_data_path,
        },
    )

    examples_written = 0
    games_completed = 0
    bridge_script = Path(args.bridge_script)
    with ShowdownBridge(bridge_script) as bridge:
        for game_index in range(args.games):
            battle_id = f"trajectory-{game_index:05d}"
            p1_team = _sample_team(catalog, rng)
            p2_team = _sample_team(catalog, rng)
            result = _run_game(
                bridge=bridge,
                writer=writer,
                backend=backend,
                rng=rng,
                catalog=catalog,
                catalog_by_species=catalog_by_species,
                species_dex=species_dex,
                battle_id=battle_id,
                p1_team=p1_team,
                p2_team=p2_team,
                pokemon_format=pokemon_format,
                generation=generation,
                search_time_ms=search_time_ms,
                threads=threads,
                hypotheses=hypotheses,
                max_turns=args.max_turns,
            )
            examples_written += result["examples_written"]
            games_completed += 1 if result.get("completed") else 0

    return {
        "output_path": str(args.output_path),
        "backend": backend.name,
        "games_requested": args.games,
        "games_completed": games_completed,
        "examples_written": examples_written,
        "species_metadata_entries": len(species_dex.entries),
    }


def _run_game(
    *,
    bridge: ShowdownBridge,
    writer: TrainingDatasetWriter,
    backend: Any,
    rng: random.Random,
    catalog: ShowdexCatalog,
    catalog_by_species: Mapping[str, Sequence[PokemonDistribution]],
    species_dex: SpeciesDex,
    battle_id: str,
    p1_team: Sequence[SampledPokemon],
    p2_team: Sequence[SampledPokemon],
    pokemon_format: str,
    generation: str,
    search_time_ms: int,
    threads: int,
    hypotheses: int,
    max_turns: int,
) -> dict[str, Any]:
    tracker = PublicBattleTracker()
    requests: dict[str, dict[str, Any]] = {}
    examples_written = 0
    ply = 0
    awaiting_public_update = False
    showdown_rng = random.SystemRandom()
    showdown_seed = [showdown_rng.randrange(1, 0x10000) for _ in range(4)]
    bridge.send(
        {
            "type": "start",
            "formatid": pokemon_format,
            "seed": showdown_seed,
            "p1": {"name": "p1", "team": _showdown_team(p1_team)},
            "p2": {"name": "p2", "team": _showdown_team(p2_team)},
        }
    )

    while tracker.turn <= max_turns:
        event = bridge.read_event()
        event_type = event.get("type")
        if event_type == "update":
            tracker.apply_update(event.get("messages") or [])
            awaiting_public_update = False
        elif event_type == "request":
            player = str(event.get("player"))
            request = _mapping(event.get("request"))
            if request:
                requests[player] = request
        elif event_type == "end":
            data = _mapping(event.get("data"))
            winner = _winner_to_player(data.get("winner")) or tracker.winner
            writer.write_result(battle_id=battle_id, winner=winner)
            return {"completed": True, "winner": winner, "examples_written": examples_written}

        if awaiting_public_update:
            continue

        if not all(player in requests for player in PLAYER_IDS):
            ready_players = _ready_players(requests)
            if not ready_players:
                continue
        else:
            ready_players = _ready_players(requests)
            if not ready_players:
                continue

        preview_players = [
            player
            for player in PLAYER_IDS
            if player in requests and _request_type(requests[player]) == "teampreview"
        ]
        if preview_players:
            if len(preview_players) < len(PLAYER_IDS):
                continue
            bridge.send({"type": "choice", "player": "p1", "choice": "team 123456"})
            bridge.send({"type": "choice", "player": "p2", "choice": "team 123456"})
            awaiting_public_update = True
            requests.clear()
            continue

        if not _public_opponents_ready(tracker, ready_players):
            continue

        choices: dict[str, str] = {}
        for player in ready_players:
            opponent = _opponent(player)
            state = _observed_state_from_request(
                request=requests[player],
                tracker=tracker,
                actor=player,
                opponent=opponent,
                battle_id=battle_id,
                pokemon_format=pokemon_format,
                generation=generation,
                species_dex=species_dex,
            )
            state["action_mask"] = _action_mask_from_request(requests[player], state)
            mcts_policy = _search_policy(
                backend=backend,
                rng=rng,
                catalog=catalog,
                catalog_by_species=catalog_by_species,
                species_dex=species_dex,
                observed_state=state,
                actor_request=requests[player],
                actor=player,
                actor_last_used_move=tracker.last_used_moves.get(player, ""),
                duration_ms=search_time_ms,
                threads=threads,
                hypotheses=hypotheses,
            )
            legal_policy = _filter_policy_to_mask(state, mcts_policy)
            if not legal_policy:
                request_type = _request_type(requests[player])
                logger.warning(
                    "Recovering empty legal policy: battle=%s ply=%s player=%s request_type=%s mask=%s raw_policy_keys=%s",
                    battle_id,
                    ply,
                    player,
                    request_type,
                    state.get("action_mask"),
                    sorted(mcts_policy),
                )
                if request_type == "move" and _active_moves_from_request(requests[player]):
                    legal_policy = {"struggle": 1.0}
                else:
                    continue
            chosen_action = max(legal_policy, key=legal_policy.get)
            choices[player] = _decision_to_showdown_choice(
                state,
                requests[player],
                chosen_action,
            )
            writer.write_decision(
                state=state,
                mcts_policy=legal_policy,
                chosen_action=chosen_action,
                strict=False,
                include_state=True,
                metadata={
                    "source": "showdown-trajectory-mcts",
                    "battle_id": battle_id,
                    "actor": player,
                    "opponent": opponent,
                    "ply": ply,
                    "request_type": _request_type(requests[player]),
                    "showdown_choice": choices[player],
                    "hypotheses_per_decision": hypotheses,
                    "search_time_ms": search_time_ms,
                    "threads": threads,
                },
            )
            examples_written += 1

        for player in PLAYER_IDS:
            if player in choices:
                bridge.send({"type": "choice", "player": player, "choice": choices[player]})
                requests.pop(player, None)
        if choices:
            awaiting_public_update = True
        for player in list(requests):
            if _request_type(requests[player]) == "wait":
                requests.pop(player, None)
        ply += 1

    writer.write_result(battle_id=battle_id, winner=tracker.winner)
    return {"completed": False, "winner": tracker.winner, "examples_written": examples_written}


def _search_policy(
    *,
    backend: Any,
    rng: random.Random,
    catalog: ShowdexCatalog,
    catalog_by_species: Mapping[str, Sequence[PokemonDistribution]],
    species_dex: SpeciesDex,
    observed_state: Mapping[str, Any],
    actor_request: Mapping[str, Any],
    actor: str,
    actor_last_used_move: str,
    duration_ms: int,
    threads: int,
    hypotheses: int,
) -> dict[str, float]:
    results = []
    for index in range(max(1, hypotheses)):
        hypothesis = dict(observed_state)
        hypothesis["user"] = _side_from_request(
            actor_request,
            observed=False,
            species_dex=species_dex,
            last_used_move=actor_last_used_move,
        )
        hypothesis["opponent"] = _sample_opponent_side(
            rng,
            catalog,
            catalog_by_species,
            species_dex,
            _mapping(observed_state.get("opponent")),
        )
        hypothesis["action_mask"] = list(observed_state.get("action_mask") or [])
        results.append(
            (
                backend.search(hypothesis, duration_ms=duration_ms, threads=threads),
                1.0 / max(1, hypotheses),
                index,
            )
        )
    return aggregate_mcts_policy(results)


def _observed_state_from_request(
    *,
    request: Mapping[str, Any],
    tracker: PublicBattleTracker,
    actor: str,
    opponent: str,
    battle_id: str,
    pokemon_format: str,
    generation: str,
    species_dex: SpeciesDex,
) -> dict[str, Any]:
    state = {
        "schema_version": 1,
        "battle_tag": battle_id,
        "pokemon_format": normalize_id(pokemon_format),
        "generation": normalize_id(generation),
        "battle_type": "SHOWDOWN_TRAJECTORY_MCTS",
        "turn": tracker.turn,
        "started": True,
        "team_preview": _request_type(request) == "teampreview",
        "force_switch": bool(request.get("forceSwitch")),
        "wait": bool(request.get("wait")),
        "weather": tracker.weather,
        "weather_turns_remaining": 0,
        "field": tracker.field_condition,
        "field_turns_remaining": 0,
        "trick_room": tracker.field_condition == "trickroom",
        "trick_room_turns_remaining": 0,
        "gravity": tracker.field_condition == "gravity",
        "user": _side_from_request(
            request,
            observed=False,
            species_dex=species_dex,
            public_side_conditions=dict(tracker.side_conditions.get(actor, {})),
            last_used_move=tracker.last_used_moves.get(actor, ""),
        ),
        "opponent": _public_side(tracker, opponent, species_dex),
    }
    return state


def _side_from_request(
    request: Mapping[str, Any],
    *,
    observed: bool,
    species_dex: SpeciesDex,
    public_side_conditions: Mapping[str, Any] | None = None,
    last_used_move: str = "",
) -> dict[str, Any]:
    side = _mapping(request.get("side"))
    pokemon = [
        _pokemon_from_request(entry, observed=observed, species_dex=species_dex)
        for entry in _sequence(side.get("pokemon"))
    ]
    active = next((entry for entry in pokemon if entry.get("active")), pokemon[0] if pokemon else {})
    reserve = [entry for entry in pokemon if entry is not active][:5]
    revival_target = _is_revival_request_from_entries(
        request,
        active,
        reserve,
        last_used_move=last_used_move,
    )
    if revival_target:
        for entry in reserve:
            entry["reviving"] = bool(entry.get("fainted"))
    active_moves = _active_moves_from_request(request)
    if active_moves:
        active["moves"] = active_moves
    active_context = _mapping(_sequence(request.get("active"), 1)[0])
    active["can_terastallize"] = bool(active_context.get("canTerastallize"))
    trapped = bool(active_context.get("trapped") or active_context.get("maybeTrapped"))
    return {
        "name": normalize_id(side.get("id") or side.get("name") or "p1"),
        "account_name": str(side.get("name") or ""),
        "active": active,
        "reserve": reserve,
        "fainted_count": sum(1 for entry in pokemon if entry.get("fainted")),
        "trapped": trapped,
        "baton_passing": False,
        "shed_tailing": False,
        "wish": [0, 0],
        "future_sight": [0, ""],
        "side_conditions": dict(public_side_conditions or {}),
        "last_selected_move": {"pokemon_name": "", "move": "", "turn": 0},
        "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
        "has_team_dict": True,
    }


def _public_side(
    tracker: PublicBattleTracker,
    player: str,
    species_dex: SpeciesDex,
) -> dict[str, Any]:
    active = _pokemon_from_visible(tracker.active.get(player), species_dex)
    seen = list(tracker.seen[player].values())
    reserve = [
        _pokemon_from_visible(entry, species_dex)
        for entry in seen
        if normalize_id(entry.species) != normalize_id(active.get("name"))
    ][:5]
    return {
        "name": player,
        "account_name": player,
        "active": active,
        "reserve": reserve,
        "fainted_count": sum(1 for entry in seen if not entry.alive),
        "trapped": False,
        "baton_passing": False,
        "shed_tailing": False,
        "wish": [0, 0],
        "future_sight": [0, ""],
        "side_conditions": dict(tracker.side_conditions.get(player, {})),
        "last_selected_move": {"pokemon_name": "", "move": "", "turn": 0},
        "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
        "has_team_dict": False,
    }


def _sample_opponent_side(
    rng: random.Random,
    catalog: ShowdexCatalog,
    catalog_by_species: Mapping[str, Sequence[PokemonDistribution]],
    species_dex: SpeciesDex,
    observed_side: Mapping[str, Any],
) -> dict[str, Any]:
    observed_pokemon = [_mapping(observed_side.get("active"))] + [
        _mapping(entry) for entry in _sequence(observed_side.get("reserve"), 5) if entry
    ]
    sampled: list[dict[str, Any]] = []
    seen_species: set[str] = set()
    for observed in observed_pokemon:
        species = normalize_id(observed.get("name") or observed.get("base_name"))
        if not species:
            continue
        pokemon = _sample_species(rng, catalog_by_species, species)
        data = _pokemon_from_sample(pokemon, observed=False, species_dex=species_dex)
        _merge_public_pokemon(data, observed)
        sampled.append(data)
        seen_species.add(species)

    while len(sampled) < 6:
        pokemon = _sample_team(catalog, rng, team_size=1)[0]
        if pokemon.species in seen_species:
            continue
        sampled.append(_pokemon_from_sample(pokemon, observed=False, species_dex=species_dex))
        seen_species.add(pokemon.species)

    return {
        "name": str(observed_side.get("name") or "opponent"),
        "account_name": str(observed_side.get("account_name") or "opponent"),
        "active": sampled[0],
        "reserve": sampled[1:6],
        "fainted_count": int(observed_side.get("fainted_count") or 0),
        "trapped": False,
        "baton_passing": False,
        "shed_tailing": False,
        "wish": [0, 0],
        "future_sight": [0, ""],
        "side_conditions": dict(_mapping(observed_side.get("side_conditions"))),
        "last_selected_move": {"pokemon_name": "", "move": "", "turn": 0},
        "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
        "has_team_dict": False,
    }


def _pokemon_from_request(
    entry: Mapping[str, Any],
    *,
    observed: bool,
    species_dex: SpeciesDex,
) -> dict[str, Any]:
    details = _parse_details(entry.get("details"))
    condition = _parse_condition(entry.get("condition"))
    species = details["species"]
    base_stats = species_dex.base_stats(species)
    stats = _stats_from_request(entry.get("stats"), fallback=base_stats)
    moves = [
        _move_dict(move)
        for move in entry.get("moves", []) or []
        if normalize_id(move)
    ]
    return {
        "name": details["species"],
        "base_name": details["species"],
        "nickname": None,
        "index": int(entry.get("slot") or 0),
        "active": bool(entry.get("active")),
        "alive": not condition["fainted"],
        "fainted": condition["fainted"],
        "reviving": False,
        "hp": condition["hp"],
        "max_hp": condition["max_hp"],
        "hp_fraction": condition["hp_fraction"],
        "status": condition["status"],
        "status_at_switch_in": "",
        "hp_at_switch_in": condition["hp"],
        "types": species_dex.types(species),
        "ability": normalize_id(entry.get("ability") or entry.get("baseAbility")) if not observed else "",
        "original_ability": normalize_id(entry.get("baseAbility") or entry.get("ability")) if not observed else "",
        "item": normalize_id(entry.get("item")) if not observed else "",
        "removed_item": "",
        "item_inferred": observed,
        "nature": "hardy",
        "evs": _stat_vector_from_request(entry.get("evs"), DEFAULT_RANDOMS_EVS, default=0),
        "ivs": _stat_vector_from_request(entry.get("ivs"), DEFAULT_IVS, default=31),
        "base_stats": base_stats,
        "stats": stats,
        "boosts": {},
        "speed_range": {"min": stats["speed"], "max": stats["speed"], "unbounded_max": False},
        "moves": moves,
        "moves_used_since_switch_in": [],
        "volatile_statuses": [],
        "volatile_status_durations": {},
        "rest_turns": 0,
        "sleep_turns": 0,
        "substitute_hit": False,
        "terastallized": details["terastallized"],
        "tera_type": details["tera_type"],
        "can_terastallize": False,
        "can_mega_evo": False,
        "can_ultra_burst": False,
        "can_dynamax": False,
        "is_mega": False,
        "mega_name": "",
        "knocked_off": False,
        "unknown_forme": False,
        "forme_changed": False,
        "zoroark_disguised_as": "",
        "can_have_choice_item": True,
        "impossible_items": [],
        "impossible_abilities": [],
        "hidden_power_possibilities": [],
        "showdown_slot": int(entry.get("slot") or 0),
        "weight_kg": species_dex.weight_kg(species),
    }


def _pokemon_from_visible(
    pokemon: VisiblePokemon | None,
    species_dex: SpeciesDex,
) -> dict[str, Any]:
    if pokemon is None or not pokemon.species:
        return {}
    hp = int(round(max(0.0, min(1.0, pokemon.hp_fraction)) * 100))
    base_stats = species_dex.base_stats(pokemon.species)
    estimated_stats = _calculated_stats(
        base_stats,
        level=pokemon.level,
        evs=DEFAULT_RANDOMS_EVS,
        ivs=DEFAULT_IVS,
    )
    return {
        "name": normalize_id(pokemon.species),
        "base_name": normalize_id(pokemon.species),
        "level": pokemon.level,
        "alive": pokemon.alive,
        "fainted": not pokemon.alive,
        "hp": hp,
        "max_hp": 100,
        "hp_fraction": pokemon.hp_fraction,
        "status": pokemon.status or "none",
        "types": species_dex.types(pokemon.species),
        "ability": pokemon.ability,
        "original_ability": pokemon.ability,
        "item": pokemon.item,
        "item_inferred": True,
        "nature": "hardy",
        "evs": [int(DEFAULT_RANDOMS_EVS[stat]) for stat in STAT_ORDER],
        "ivs": [int(DEFAULT_IVS[stat]) for stat in STAT_ORDER],
        "base_stats": base_stats,
        "stats": estimated_stats,
        "boosts": dict(pokemon.boosts),
        "speed_range": {"min": estimated_stats["speed"], "max": estimated_stats["speed"], "unbounded_max": False},
        "moves": [_move_dict(move) for move in sorted(pokemon.moves)],
        "tera_type": "typeless",
        "terastallized": False,
        "can_terastallize": False,
        "weight_kg": species_dex.weight_kg(pokemon.species),
    }


def _pokemon_from_sample(
    pokemon: SampledPokemon,
    *,
    observed: bool,
    species_dex: SpeciesDex,
) -> dict[str, Any]:
    base_stats = species_dex.base_stats(pokemon.species)
    stats = _calculated_stats(
        base_stats,
        level=pokemon.level,
        evs=pokemon.evs,
        ivs=pokemon.ivs,
    )
    max_hp = stats["hp"]
    return {
        "name": normalize_id(pokemon.species),
        "base_name": normalize_id(pokemon.species),
        "level": pokemon.level,
        "alive": True,
        "fainted": False,
        "hp": max_hp,
        "max_hp": max_hp,
        "hp_fraction": 1.0,
        "status": "none",
        "types": species_dex.types(pokemon.species),
        "ability": normalize_id(pokemon.ability) if not observed else "",
        "original_ability": normalize_id(pokemon.ability) if not observed else "",
        "item": normalize_id(pokemon.item) if not observed else "",
        "item_inferred": observed,
        "nature": "hardy",
        "evs": [int(pokemon.evs.get(stat, 0)) for stat in STAT_ORDER],
        "ivs": [int(pokemon.ivs.get(stat, 31)) for stat in STAT_ORDER],
        "base_stats": base_stats,
        "stats": stats,
        "boosts": {},
        "speed_range": {"min": stats["speed"], "max": stats["speed"], "unbounded_max": False},
        "moves": [_move_dict(move) for move in pokemon.moves],
        "tera_type": normalize_id(pokemon.tera_type) or "typeless",
        "terastallized": False,
        "can_terastallize": False,
        "weight_kg": species_dex.weight_kg(pokemon.species),
    }


def _merge_public_pokemon(target: dict[str, Any], observed: Mapping[str, Any]) -> None:
    for key in ("hp", "max_hp", "hp_fraction", "status", "alive", "fainted", "boosts"):
        if key in observed:
            target[key] = observed[key]
    revealed_moves = [move for move in _sequence(observed.get("moves")) if move]
    if revealed_moves:
        target["moves"] = revealed_moves + [
            move for move in target.get("moves", []) if move.get("name") not in {m.get("name") for m in revealed_moves}
        ]
        target["moves"] = target["moves"][:4]


def _active_moves_from_request(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    active = _mapping(_sequence(request.get("active"), 1)[0])
    moves = []
    for move in _sequence(active.get("moves"), 4):
        move_map = _mapping(move)
        if not move_map:
            continue
        moves.append(
            {
                "name": normalize_id(move_map.get("id") or move_map.get("move")),
                "current_pp": _int(move_map.get("pp"), 1),
                "max_pp": _int(move_map.get("maxpp"), _int(move_map.get("pp"), 1)),
                "disabled": bool(move_map.get("disabled")),
                "can_z": False,
            }
        )
    return moves


def _stat_vector_from_request(
    values: object,
    fallback: Mapping[str, int],
    *,
    default: int,
) -> list[int]:
    source = _mapping(values)
    if source:
        return [
            _int(source.get(SHOWDOWN_STAT_KEYS[stat], source.get(stat)), int(fallback.get(stat, default)))
            for stat in STAT_ORDER
        ]

    sequence = _sequence(values, len(STAT_ORDER))
    if any(item is not None for item in sequence):
        return [
            _int(item, int(fallback.get(stat, default)))
            for stat, item in zip(STAT_ORDER, sequence, strict=True)
        ]

    return [int(fallback.get(stat, default)) for stat in STAT_ORDER]


def _action_mask_from_request(request: Mapping[str, Any], state: Mapping[str, Any]) -> list[int]:
    mask = [0] * ACTION_SLOTS
    request_type = _request_type(request)
    user = _mapping(state.get("user"))
    active_context = _mapping(_sequence(request.get("active"), 1)[0])
    force_switch = request_type == "switch"
    revival_target = any(
        bool(_mapping(pokemon).get("reviving"))
        for pokemon in _sequence(user.get("reserve"), 5)[:5]
    )
    trapped = bool(active_context.get("trapped"))

    if not force_switch:
        for index, move in enumerate(_active_moves_from_request(request)[:4]):
            if not move.get("disabled") and move.get("current_pp", 1) != 0:
                mask[index] = 1
        if active_context.get("canTerastallize"):
            for index in range(4):
                if mask[index]:
                    mask[9 + index] = 1

    if not trapped or force_switch:
        for index, pokemon in enumerate(_sequence(user.get("reserve"), 5)[:5]):
            pokemon = _mapping(pokemon)
            if not pokemon:
                continue
            if revival_target:
                if pokemon.get("fainted"):
                    mask[4 + index] = 1
            elif pokemon.get("alive", True) and not pokemon.get("fainted"):
                mask[4 + index] = 1

    if request_type == "move" and not any(mask):
        active_moves = _active_moves_from_request(request)
        if active_moves:
            # Use the first move slot as a Struggle surrogate when Showdown would
            # force Struggle because no other move/switch option is available.
            mask[0] = 1

    return mask


def _filter_policy_to_mask(state: Mapping[str, Any], policy: Mapping[str, float]) -> dict[str, float]:
    mask = list(state.get("action_mask") or [])
    filtered: dict[str, float] = {}
    for decision, probability in policy.items():
        try:
            slot = decision_to_action_slot(state, decision)
        except ActionMappingError:
            continue
        if slot < len(mask) and mask[slot]:
            filtered[decision] = float(probability)
    if not filtered:
        for slot, allowed in enumerate(mask[:ACTION_SLOTS]):
            if allowed:
                from map import action_slot_to_decision

                filtered[action_slot_to_decision(state, slot)] = 1.0
                break
    total = sum(filtered.values())
    return {decision: value / total for decision, value in filtered.items()} if total else filtered


def _decision_to_showdown_choice(
    state: Mapping[str, Any],
    request: Mapping[str, Any],
    decision: str,
) -> str:
    slot = decision_to_action_slot(state, decision)
    if 0 <= slot < 4:
        return f"move {slot + 1}"
    if 9 <= slot < 13:
        return f"move {slot - 8} terastallize"
    reserve_index = slot - 4
    reserve = _sequence(_mapping(state.get("user")).get("reserve"), 5)
    pokemon = _mapping(reserve[reserve_index])
    showdown_slot = int(pokemon.get("showdown_slot") or 0)
    if showdown_slot <= 0:
        showdown_slot = _find_showdown_switch_slot(request, pokemon)
    return f"switch {showdown_slot}"


def _find_showdown_switch_slot(request: Mapping[str, Any], pokemon: Mapping[str, Any]) -> int:
    target = normalize_id(pokemon.get("name") or pokemon.get("base_name"))
    for index, entry in enumerate(_sequence(_mapping(request.get("side")).get("pokemon")), start=1):
        entry_map = _mapping(entry)
        details = _parse_details(entry_map.get("details"))
        if normalize_id(details["species"]) == target:
            return index
    raise ValueError(f"Unable to map switch target to Showdown slot: {target}")


def _showdown_team(team: Sequence[SampledPokemon]) -> list[dict[str, Any]]:
    return [
        {
            "name": pokemon.species,
            "species": pokemon.species,
            "item": pokemon.item,
            "ability": pokemon.ability,
            "moves": list(pokemon.moves),
            "nature": "Serious",
            "evs": {SHOWDOWN_STAT_KEYS[key]: int(pokemon.evs.get(key, 0)) for key in STAT_ORDER},
            "ivs": {SHOWDOWN_STAT_KEYS[key]: int(pokemon.ivs.get(key, 31)) for key in STAT_ORDER},
            "level": pokemon.level,
            "teraType": pokemon.tera_type,
        }
        for pokemon in team
    ]


def _sample_team(catalog: ShowdexCatalog, rng: random.Random, *, team_size: int = 6) -> list[SampledPokemon]:
    return [entry.sample(rng) for entry in catalog.sample_team_distributions(rng, team_size=team_size)]


def _sample_species(
    rng: random.Random,
    catalog_by_species: Mapping[str, Sequence[PokemonDistribution]],
    species: str,
) -> SampledPokemon:
    entries = list(catalog_by_species.get(normalize_id(species)) or [])
    if not entries:
        return SampledPokemon(
            species=normalize_id(species),
            level=100,
            ability="none",
            item="none",
            moves=("tackle",),
            tera_type="typeless",
            evs=dict(DEFAULT_RANDOMS_EVS),
            ivs=dict(DEFAULT_IVS),
        )
    return rng.choice(entries).sample(rng)


def _catalog_by_species(catalog: ShowdexCatalog) -> dict[str, list[PokemonDistribution]]:
    grouped: dict[str, list[PokemonDistribution]] = defaultdict(list)
    for entry in catalog.entries:
        grouped[normalize_id(entry.species)].append(entry)
    return dict(grouped)


def _load_catalog(settings: Mapping[str, Any]) -> ShowdexCatalog:
    showdex = _mapping(settings.get("showdex"))
    return ShowdexCatalog.from_files(
        randoms_stats_path=settings.get("randoms_stats_path")
        or showdex.get("randoms_stats_path"),
        randoms_preset_path=settings.get("randoms_preset_path")
        or showdex.get("randoms_preset_path"),
        prefer_stats=_bool(showdex.get("prefer_stats"), True),
    )


def _apply_overrides(settings: dict[str, Any], args: argparse.Namespace) -> None:
    for key in (
        "output_path",
        "search_time_ms",
        "threads",
        "hypotheses",
        "pokemon_format",
        "generation",
        "seed",
        "backend",
        "species_data_path",
        "randoms_preset_path",
        "randoms_stats_path",
    ):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value


def _resolve_run_seed(
    settings: Mapping[str, Any],
    args: argparse.Namespace,
    mcts: Mapping[str, Any],
) -> int:
    if args.seed is not None:
        return int(args.seed)
    value = settings.get("seed")
    if value is not None and value != 1337:
        return _int(value, 1337)
    value = mcts.get("seed")
    if value is not None and value != 1337:
        return _int(value, 1337)
    return random.SystemRandom().randrange(1, 2_147_483_647)


def _request_type(request: Mapping[str, Any]) -> str:
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        return "teampreview"
    if request.get("forceSwitch"):
        return "switch"
    return "move"


def _is_revival_request_from_entries(
    request: Mapping[str, Any],
    active: Mapping[str, Any],
    reserve: Sequence[Mapping[str, Any]],
    *,
    last_used_move: str = "",
) -> bool:
    if not request.get("forceSwitch"):
        return False
    if normalize_id(last_used_move) != "revivalblessing":
        return False
    if not active or active.get("fainted") or not active.get("alive", True):
        return False
    return any(bool(pokemon.get("fainted")) for pokemon in reserve if pokemon)


def _ready_players(requests: Mapping[str, Mapping[str, Any]]) -> list[str]:
    actionable = [
        player
        for player in PLAYER_IDS
        if player in requests and _request_type(requests[player]) not in {"wait", "teampreview"}
    ]
    if len(actionable) == 2:
        return actionable
    if len(actionable) == 1 and _request_type(requests[actionable[0]]) == "switch":
        return actionable
    return []


def _public_opponents_ready(
    tracker: PublicBattleTracker,
    players: Sequence[str],
) -> bool:
    for player in players:
        opponent = _opponent(player)
        active = tracker.active.get(opponent)
        if active is None or not active.species:
            return False
    return True


def _move_dict(move: object) -> dict[str, Any]:
    return {
        "name": normalize_id(move),
        "current_pp": 16,
        "max_pp": 16,
        "disabled": False,
        "can_z": False,
    }


def _visible_from_details(details: object) -> VisiblePokemon:
    parsed = _parse_details(details)
    return VisiblePokemon(
        species=parsed["species"],
        level=parsed["level"],
        hp_fraction=1.0,
        status="none",
        alive=True,
    )


def _parse_details(details: object) -> dict[str, Any]:
    parts = [part.strip() for part in str(details or "").split(",")]
    species = normalize_id(parts[0]) if parts and parts[0] else ""
    level = 100
    tera_type = "typeless"
    terastallized = False
    for part in parts[1:]:
        lower = part.lower()
        if lower.startswith("l") and lower[1:].isdigit():
            level = int(lower[1:])
        elif lower.startswith("tera:"):
            tera_type = normalize_id(lower.split(":", 1)[1]) or "typeless"
            terastallized = True
    return {
        "species": species,
        "level": level,
        "tera_type": tera_type,
        "terastallized": terastallized,
    }


def _parse_condition(condition: object) -> dict[str, Any]:
    text = str(condition or "")
    if not text:
        return {"hp": 100, "max_hp": 100, "hp_fraction": 1.0, "status": "none", "fainted": False}
    parts = text.split()
    hp_part = parts[0]
    status = normalize_id(parts[1]) if len(parts) > 1 else "none"
    if status == "fnt" or hp_part == "0":
        return {"hp": 0, "max_hp": 100, "hp_fraction": 0.0, "status": "fnt", "fainted": True}
    if "/" in hp_part:
        left, right = hp_part.split("/", 1)
        hp = _int(left, 100)
        max_hp = max(1, _int(right, 100))
        return {"hp": hp, "max_hp": max_hp, "hp_fraction": hp / max_hp, "status": status, "fainted": False}
    hp = _int(hp_part, 100)
    return {"hp": hp, "max_hp": 100, "hp_fraction": hp / 100.0, "status": status, "fainted": False}


def _apply_condition(pokemon: VisiblePokemon, condition: object) -> None:
    parsed = _parse_condition(condition)
    pokemon.hp_fraction = parsed["hp_fraction"]
    pokemon.status = parsed["status"]
    pokemon.alive = not parsed["fainted"]


def _stats_from_request(
    stats: object,
    *,
    fallback: Mapping[str, int] | None = None,
) -> dict[str, int]:
    source = _mapping(stats)
    if not source:
        return dict(fallback or _default_stats())
    fallback_stats = fallback or _default_stats()
    return {
        stat: _int(source.get(alias), int(fallback_stats.get(stat, 100)))
        for stat, alias in SHOWDOWN_STAT_KEYS.items()
    }


def _normalize_base_stats(stats: object) -> dict[str, int]:
    source = _mapping(stats)
    if not source:
        return _default_stats()
    values = {}
    for stat in STAT_ORDER:
        showdown_key = SHOWDOWN_STAT_KEYS[stat]
        values[stat] = _int(source.get(stat, source.get(showdown_key)), 100)
    return values


def _calculated_stats(
    base_stats: Mapping[str, int],
    *,
    level: int,
    evs: Mapping[str, int],
    ivs: Mapping[str, int],
) -> dict[str, int]:
    values = {}
    for stat in STAT_ORDER:
        base = int(base_stats.get(stat, 100))
        iv = int(ivs.get(stat, 31))
        ev = int(evs.get(stat, 0))
        raw = int(((2 * base + iv + ev // 4) * level) / 100)
        if stat == "hp":
            values[stat] = raw + level + 10
        else:
            values[stat] = raw + 5
    return values


def _default_stats() -> dict[str, int]:
    return {
        "hp": 100,
        "attack": 100,
        "defense": 100,
        "special-attack": 100,
        "special-defense": 100,
        "speed": 100,
    }


def _player_from_ident(ident: object) -> str:
    text = str(ident or "")
    return text[:2] if text[:2] in PLAYER_IDS else ""


def _species_from_ident(ident: object) -> str:
    text = str(ident or "")
    if ":" not in text:
        return ""
    return normalize_id(text.split(":", 1)[1].strip())


def _player_from_side(side: object) -> str:
    text = str(side or "").lower()
    if "p1" in text:
        return "p1"
    if "p2" in text:
        return "p2"
    return ""


def _winner_to_player(winner: object) -> str | None:
    text = normalize_id(winner)
    if text in PLAYER_IDS:
        return text
    if text == "player1":
        return "p1"
    if text == "player2":
        return "p2"
    return None


def _opponent(player: str) -> str:
    return "p2" if player == "p1" else "p1"


def _generation_from_format(pokemon_format: str) -> str:
    normalized = normalize_id(pokemon_format)
    if normalized.startswith("gen") and len(normalized) >= 4 and normalized[3].isdigit():
        return f"gen{normalized[3]}"
    return "gen9"


def _base_species_id(species_id: str) -> str:
    suffixes = (
        "mega",
        "megax",
        "megay",
        "gmax",
        "totem",
        "alola",
        "galar",
        "hisui",
        "paldea",
    )
    for suffix in suffixes:
        if species_id.endswith(suffix) and len(species_id) > len(suffix):
            return species_id[: -len(suffix)]
    return species_id


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object, length: int | None = None) -> list[Any]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str | bytes | bytearray):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    if length is None:
        return items
    return (items + [None] * length)[:length]


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("trajectory collection failed")
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise
