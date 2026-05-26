"""Offline MCTS collection helpers built around poke-engine states."""

from __future__ import annotations

import inspect
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace as NS
from typing import Any

from map import ACTION_SLOTS, action_slot_to_decision, legal_action_mask
from showdex_distributions import (
    DEFAULT_RANDOMS_EVS,
    STAT_ORDER,
    PokemonDistribution,
    SampledPokemon,
    ShowdexCatalog,
    distribution_to_json,
    normalize_id,
)


@dataclass(frozen=True)
class OfflinePosition:
    """One observed position plus concrete hidden-state hypotheses."""

    observed_state: dict[str, Any]
    hypothesis_states: list[dict[str, Any]]


class PokeEngineBackend:
    """Thin adapter around poke-engine's Python MCTS binding."""

    name = "poke-engine"

    def __init__(self) -> None:
        try:
            from poke_engine import (
                Move,
                Pokemon,
                Side,
                SideConditions,
                State,
                monte_carlo_tree_search,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "poke-engine is required for the offline MCTS backend. "
                "Install it with `pip install poke-engine`."
            ) from exc

        self.Move = Move
        self.Pokemon = Pokemon
        self.Side = Side
        self.SideConditions = SideConditions
        self.State = State
        self.monte_carlo_tree_search = monte_carlo_tree_search
        self._mcts_accepts_threads = (
            len(inspect.signature(monte_carlo_tree_search).parameters) >= 3
        )

    def search(
        self,
        state: Mapping[str, Any],
        *,
        duration_ms: int,
        threads: int = 1,
    ) -> Any:
        engine_state = self.to_engine_state(state)
        if self._mcts_accepts_threads:
            return self.monte_carlo_tree_search(
                engine_state,
                int(duration_ms),
                max(1, int(threads)),
            )
        return self.monte_carlo_tree_search(engine_state, int(duration_ms))

    def to_engine_state(self, state: Mapping[str, Any]) -> Any:
        return self.State(
            side_one=self._side(_mapping(state.get("user"))),
            side_two=self._side(_mapping(state.get("opponent"))),
            weather=_weather(state.get("weather")),
            weather_turns_remaining=int(state.get("weather_turns_remaining") or 0),
            terrain=_terrain(state.get("field")),
            terrain_turns_remaining=int(state.get("field_turns_remaining") or 0),
            trick_room=bool(state.get("trick_room")),
            trick_room_turns_remaining=int(state.get("trick_room_turns_remaining") or 0),
            team_preview=bool(state.get("team_preview")),
        )

    def _side(self, side: Mapping[str, Any]) -> Any:
        active = _mapping(side.get("active"))
        reserve = [_mapping(pokemon) for pokemon in _sequence(side.get("reserve"), 5)]
        pokemon = [
            self._pokemon(active, can_terastallize=bool(active.get("can_terastallize", True)))
        ] + [self._pokemon(slot, can_terastallize=False) for slot in reserve if slot]
        boosts = _mapping(active.get("boosts"))
        return self.Side(
            pokemon=pokemon,
            active_index="0",
            force_trapped=bool(side.get("trapped")),
            baton_passing=bool(side.get("baton_passing")),
            shed_tailing=bool(side.get("shed_tailing")),
            side_conditions=self._side_conditions(_mapping(side.get("side_conditions"))),
            wish=_wish(side.get("wish")),
            future_sight=_future_sight(side.get("future_sight")),
            attack_boost=int(boosts.get("attack") or 0),
            defense_boost=int(boosts.get("defense") or 0),
            special_attack_boost=int(boosts.get("special-attack") or 0),
            special_defense_boost=int(boosts.get("special-defense") or 0),
            speed_boost=int(boosts.get("speed") or 0),
            accuracy_boost=int(boosts.get("accuracy") or 0),
            evasion_boost=int(boosts.get("evasion") or 0),
            last_used_move=_last_used_move(side.get("last_used_move")),
        )

    def _pokemon(self, pokemon: Mapping[str, Any], *, can_terastallize: bool = False) -> Any:
        moves = [
            self.Move(
                id=normalize_id(move.get("name") or move.get("id")),
                pp=int(move.get("current_pp") or move.get("pp") or 16),
                disabled=bool(move.get("disabled")),
            )
            for move in [_mapping(move) for move in _sequence(pokemon.get("moves"), 4)]
            if normalize_id(move.get("name") or move.get("id"))
        ]
        stats = _mapping(pokemon.get("stats"))
        types = _types(pokemon)
        kwargs = {
            "id": normalize_id(pokemon.get("name") or pokemon.get("species") or "pikachu"),
            "level": int(pokemon.get("level") or 100),
            "types": types,
            "base_types": types,
            "hp": _int_or_default(pokemon.get("hp"), _int_or_default(pokemon.get("max_hp"), 100)),
            "maxhp": _int_or_default(pokemon.get("max_hp"), _int_or_default(pokemon.get("hp"), 100)),
            "ability": normalize_id(pokemon.get("ability")) or "none",
            "item": normalize_id(pokemon.get("item")) or "none",
            "nature": normalize_id(pokemon.get("nature")) or "serious",
            "evs": _ev_tuple(_mapping(pokemon.get("evs"))),
            "attack": int(stats.get("attack") or 100),
            "defense": int(stats.get("defense") or 100),
            "special_attack": int(stats.get("special-attack") or 100),
            "special_defense": int(stats.get("special-defense") or 100),
            "speed": int(stats.get("speed") or 100),
            "status": _status(pokemon.get("status")),
            "rest_turns": int(pokemon.get("rest_turns") or 0),
            "sleep_turns": int(pokemon.get("sleep_turns") or 0),
            "weight_kg": float(pokemon.get("weight_kg") or 0.0),
            "moves": moves,
            "terastallized": bool(pokemon.get("terastallized")),
            "tera_type": normalize_id(pokemon.get("tera_type")) or "typeless",
            "can_terastallize": bool(pokemon.get("can_terastallize", can_terastallize)),
        }
        try:
            return self.Pokemon(**kwargs)
        except TypeError as exc:
            if "can_terastallize" not in str(exc):
                raise
            kwargs.pop("can_terastallize", None)
            return self.Pokemon(**kwargs)

    def _side_conditions(self, conditions: Mapping[str, Any]) -> Any:
        return self.SideConditions(
            spikes=int(conditions.get("spikes") or 0),
            toxic_spikes=int(conditions.get("toxicspikes") or conditions.get("toxic_spikes") or 0),
            stealth_rock=int(conditions.get("stealthrock") or conditions.get("stealth_rock") or 0),
            sticky_web=int(conditions.get("stickyweb") or conditions.get("sticky_web") or 0),
            tailwind=int(conditions.get("tailwind") or 0),
            reflect=int(conditions.get("reflect") or 0),
            light_screen=int(conditions.get("lightscreen") or conditions.get("light_screen") or 0),
            aurora_veil=int(conditions.get("auroraveil") or conditions.get("aurora_veil") or 0),
            safeguard=int(conditions.get("safeguard") or 0),
            healing_wish=int(conditions.get("healingwish") or conditions.get("healing_wish") or 0),
        )


class ToyMctsBackend:
    """Deterministic backend used only for local smoke tests."""

    name = "toy"

    def search(
        self,
        state: Mapping[str, Any],
        *,
        duration_ms: int,
        threads: int = 1,
    ) -> Any:
        del duration_ms, threads
        side_one = []
        legal_slots = [
            slot
            for slot, allowed in enumerate(legal_action_mask(state)[:ACTION_SLOTS])
            if allowed
        ]
        total_visits = 0
        for rank, slot in enumerate(legal_slots):
            decision = action_slot_to_decision(state, slot)
            visits = max(1, 64 - rank * 8)
            total_visits += visits
            side_one.append(
                NS(move_choice=decision, total_score=float(visits), visits=visits)
            )
        return NS(side_one=side_one, side_two=[], total_visits=total_visits)


def make_backend(name: str, *, allow_toy_backend: bool = False) -> PokeEngineBackend | ToyMctsBackend:
    normalized = normalize_id(name or "poke-engine")
    if normalized == "toy":
        if not allow_toy_backend:
            raise ValueError("Toy backend is only allowed when allow_toy_backend=true")
        return ToyMctsBackend()
    return PokeEngineBackend()


def sample_offline_position(
    catalog: ShowdexCatalog,
    rng: random.Random,
    *,
    hypotheses: int,
    pokemon_format: str,
    generation: str,
    battle_tag: str,
    reveal_opponent_set: bool = False,
) -> OfflinePosition:
    user_distributions = catalog.sample_team_distributions(rng, team_size=6)
    opponent_distributions = catalog.sample_team_distributions(rng, team_size=6)
    user_side = [distribution.sample(rng) for distribution in user_distributions]

    observed_opponent = [
        distribution.observed(reveal_set=reveal_opponent_set)
        for distribution in opponent_distributions
    ]
    observed_state = build_training_state(
        user_side,
        observed_opponent,
        pokemon_format=pokemon_format,
        generation=generation,
        battle_tag=battle_tag,
        observed=True,
    )

    hypothesis_states = []
    for index in range(max(1, int(hypotheses))):
        opponent_side = [distribution.sample(rng) for distribution in opponent_distributions]
        hypothesis_states.append(
            build_training_state(
                user_side,
                opponent_side,
                pokemon_format=pokemon_format,
                generation=generation,
                battle_tag=f"{battle_tag}-h{index}",
                observed=False,
            )
        )

    return OfflinePosition(
        observed_state=observed_state,
        hypothesis_states=hypothesis_states,
    )


def build_training_state(
    user_side: Sequence[SampledPokemon],
    opponent_side: Sequence[SampledPokemon],
    *,
    pokemon_format: str,
    generation: str,
    battle_tag: str,
    observed: bool,
) -> dict[str, Any]:
    state = {
        "schema_version": 1,
        "battle_tag": battle_tag,
        "pokemon_format": normalize_id(pokemon_format),
        "generation": normalize_id(generation),
        "battle_type": "OFFLINE_SHOWDEX_MCTS",
        "turn": 1,
        "started": True,
        "team_preview": False,
        "force_switch": False,
        "wait": False,
        "weather": "none",
        "weather_turns_remaining": 0,
        "field": "none",
        "field_turns_remaining": 0,
        "trick_room": False,
        "trick_room_turns_remaining": 0,
        "gravity": False,
        "user": _serialize_side("p1", user_side, observed=False),
        "opponent": _serialize_side("p2", opponent_side, observed=observed),
    }
    state["action_mask"] = legal_action_mask(state)
    return state


def _serialize_side(
    name: str,
    pokemon: Sequence[SampledPokemon],
    *,
    observed: bool,
) -> dict[str, Any]:
    active = pokemon[0]
    reserve = list(pokemon[1:6])
    return {
        "name": name,
        "account_name": name,
        "active": _serialize_pokemon(active, observed=observed, can_terastallize=True),
        "reserve": [_serialize_pokemon(slot, observed=observed, can_terastallize=False) for slot in reserve],
        "fainted_count": 0,
        "trapped": False,
        "baton_passing": False,
        "shed_tailing": False,
        "wish": [0, 0],
        "future_sight": [0, ""],
        "side_conditions": {},
        "last_selected_move": {"pokemon_name": "", "move": "", "turn": 0},
        "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
        "has_team_dict": False,
    }


def _serialize_pokemon(
    pokemon: SampledPokemon,
    *,
    observed: bool,
    can_terastallize: bool,
) -> dict[str, Any]:
    max_hp = _rough_hp(pokemon)
    stats = _rough_stats(pokemon)
    data = {
        "name": normalize_id(pokemon.species),
        "base_name": normalize_id(pokemon.species),
        "nickname": None,
        "index": None,
        "level": pokemon.level,
        "alive": True,
        "fainted": False,
        "reviving": False,
        "hp": max_hp,
        "max_hp": max_hp,
        "hp_fraction": 1.0,
        "status": "none",
        "status_at_switch_in": "",
        "hp_at_switch_in": max_hp,
        "types": ["normal", "typeless"],
        "ability": normalize_id(pokemon.ability) if pokemon.ability or not observed else "",
        "original_ability": normalize_id(pokemon.ability) if pokemon.ability or not observed else "",
        "item": normalize_id(pokemon.item) if pokemon.item or not observed else "",
        "removed_item": "",
        "item_inferred": observed,
        "nature": "hardy",
        "evs": [int(pokemon.evs.get(stat, 0)) for stat in STAT_ORDER],
        "ivs": [int(pokemon.ivs.get(stat, 31)) for stat in STAT_ORDER],
        "base_stats": stats,
        "stats": stats,
        "boosts": {},
        "speed_range": {"min": stats["speed"], "max": stats["speed"], "unbounded_max": False},
        "moves": [
            {
                "name": move,
                "current_pp": 16,
                "max_pp": 16,
                "disabled": False,
                "can_z": False,
            }
            for move in pokemon.moves
        ],
        "moves_used_since_switch_in": [],
        "volatile_statuses": [],
        "volatile_status_durations": {},
        "rest_turns": 0,
        "sleep_turns": 0,
        "substitute_hit": False,
        "terastallized": False,
        "tera_type": normalize_id(pokemon.tera_type) or "typeless",
        "can_terastallize": bool(can_terastallize),
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
    }

    if observed and pokemon.distributions:
        for key, value in pokemon.distributions.items():
            data[key] = distribution_to_json(value)

    return data


def _rough_hp(pokemon: SampledPokemon) -> int:
    return max(1, int(100 + pokemon.level + pokemon.evs.get("hp", DEFAULT_RANDOMS_EVS["hp"]) / 4))


def _rough_stats(pokemon: SampledPokemon) -> dict[str, int]:
    del pokemon
    return {
        "hp": 100,
        "attack": 100,
        "defense": 100,
        "special-attack": 100,
        "special-defense": 100,
        "speed": 100,
    }


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object, length: int) -> list[Any]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str | bytes | bytearray):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    return (items + [None] * length)[:length]


def _types(pokemon: Mapping[str, Any]) -> tuple[str, str]:
    values = [normalize_id(value) for value in _sequence(pokemon.get("types"), 2)]
    values = [value for value in values if value]
    return ((values + ["typeless", "typeless"])[:2][0], (values + ["typeless", "typeless"])[:2][1])


def _status(value: object) -> str:
    normalized = normalize_id(value) or "none"
    return {
        "brn": "burn",
        "frz": "freeze",
        "fnt": "none",
        "fainted": "none",
        "par": "paralyze",
        "psn": "poison",
        "slp": "sleep",
        "tox": "toxic",
    }.get(normalized, normalized)


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ev_tuple(evs: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    merged = dict(DEFAULT_RANDOMS_EVS)
    for key, value in evs.items():
        if key in merged:
            merged[key] = int(value)
    return (
        merged["hp"],
        merged["attack"],
        merged["defense"],
        merged["special-attack"],
        merged["special-defense"],
        merged["speed"],
    )


def _wish(value: object) -> tuple[int, int]:
    items = _sequence(value, 2)
    return (int(items[0] or 0), int(items[1] or 0))


def _future_sight(value: object) -> tuple[int, str]:
    items = _sequence(value, 2)
    return (int(items[0] or 0), str(items[1] or "0"))


def _last_used_move(value: object) -> str:
    value = _mapping(value)
    move = normalize_id(value.get("move"))
    if not move:
        return "move:none"
    return "move:0"


def _weather(value: object) -> str:
    value = normalize_id(value)
    return {
        "sunnyday": "sun",
        "raindance": "rain",
        "sandstorm": "sand",
        "hail": "hail",
        "snow": "snow",
        "snowscape": "snow",
        "desolateland": "harshsun",
        "primordialsea": "heavyrain",
        "deltastream": "none",
        "strongwinds": "none",
    }.get(value, value or "none")


def _terrain(value: object) -> str:
    value = normalize_id(value)
    if value in {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}:
        return value
    return "none"
