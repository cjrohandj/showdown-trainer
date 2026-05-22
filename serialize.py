"""Serialize Foul Play battle objects into JSON-ready training states."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

try:
    from map import legal_action_mask
except ImportError:  # pragma: no cover - useful when copied into another repo.
    legal_action_mask = None


SCHEMA_VERSION = 1


def normalize_name(value: object) -> str:
    """Normalize Showdown-style identifiers while keeping unknowns empty."""

    if value is None:
        return ""
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _field(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum_name(value: object) -> object:
    if isinstance(value, Enum):
        return value.name
    return value


def _number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _fraction(numerator: object, denominator: object) -> float | None:
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def _jsonable(value: object) -> Any:
    value = _enum_name(value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(_jsonable(key)): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set | frozenset):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return str(value)


def _compact_mapping(
    value: object,
    *,
    keep_zero: bool = False,
    normalize_keys: bool = True,
) -> dict[str, Any]:
    if value is None:
        return {}

    mapping = dict(value)
    result: dict[str, Any] = {}
    for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0])):
        item = _jsonable(item)
        if not keep_zero and item in (None, False, 0, "", [], {}):
            continue
        key = normalize_name(key) if normalize_keys else str(key)
        result[key] = item
    return result


def _normalized_list(value: object) -> list[str]:
    if value is None:
        return []
    return [normalize_name(item) for item in list(value)]


def _sorted_normalized_set(value: object) -> list[str]:
    if value is None:
        return []
    return sorted(normalize_name(item) for item in set(value))


def _tuple_or_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return [_jsonable(value)]


def _is_alive(pokemon: object) -> bool:
    is_alive = getattr(pokemon, "is_alive", None)
    if callable(is_alive):
        return bool(is_alive())
    hp = _number(_field(pokemon, "hp", 0))
    return bool(hp and hp > 0)


def serialize_last_move(last_move: object) -> dict[str, Any]:
    """Serialize Foul Play's LastUsedMove namedtuple."""

    return {
        "pokemon_name": normalize_name(_field(last_move, "pokemon_name", "")),
        "move": normalize_name(_field(last_move, "move", "")),
        "turn": _number(_field(last_move, "turn", 0)),
    }


def serialize_speed_range(pokemon: object) -> dict[str, Any]:
    speed_range = _field(pokemon, "speed_range")
    minimum = _number(_field(speed_range, "min", 0))
    maximum = _number(_field(speed_range, "max"))

    return {
        "min": minimum,
        "max": maximum,
        "unbounded_max": maximum is None,
    }


def serialize_move(move: object) -> dict[str, Any]:
    """Serialize a Foul Play Move or a plain move string."""

    if isinstance(move, str):
        return {
            "name": normalize_name(move),
            "current_pp": None,
            "max_pp": None,
            "disabled": False,
            "can_z": False,
        }

    return {
        "name": normalize_name(_field(move, "name", _field(move, "id", ""))),
        "current_pp": _number(_field(move, "current_pp", _field(move, "pp"))),
        "max_pp": _number(_field(move, "max_pp", _field(move, "maxpp"))),
        "disabled": bool(_field(move, "disabled", False)),
        "can_z": bool(_field(move, "can_z", False)),
    }


def serialize_pokemon(pokemon: object) -> dict[str, Any] | None:
    """Serialize a Pokemon slot. Returns None for empty slots."""

    if pokemon is None:
        return None

    hp = _number(_field(pokemon, "hp"))
    max_hp = _number(_field(pokemon, "max_hp", _field(pokemon, "maxhp")))

    return {
        "name": normalize_name(_field(pokemon, "name", "")),
        "base_name": normalize_name(_field(pokemon, "base_name", "")),
        "nickname": _jsonable(_field(pokemon, "nickname")),
        "index": _number(_field(pokemon, "index")),
        "level": _number(_field(pokemon, "level")),
        "alive": _is_alive(pokemon),
        "fainted": bool(_field(pokemon, "fainted", False)) or hp == 0,
        "reviving": bool(_field(pokemon, "reviving", False)),
        "hp": hp,
        "max_hp": max_hp,
        "hp_fraction": _fraction(hp, max_hp),
        "status": normalize_name(_field(pokemon, "status", "")),
        "status_at_switch_in": normalize_name(
            _field(pokemon, "status_at_switch_in", "")
        ),
        "hp_at_switch_in": _number(_field(pokemon, "hp_at_switch_in")),
        "types": _normalized_list(_field(pokemon, "types", [])),
        "ability": normalize_name(_field(pokemon, "ability", "")),
        "original_ability": normalize_name(_field(pokemon, "original_ability", "")),
        "item": normalize_name(_field(pokemon, "item", "")),
        "removed_item": normalize_name(_field(pokemon, "removed_item", "")),
        "item_inferred": bool(_field(pokemon, "item_inferred", False)),
        "nature": normalize_name(_field(pokemon, "nature", "")),
        "evs": _tuple_or_list(_field(pokemon, "evs", [])),
        "base_stats": _compact_mapping(_field(pokemon, "base_stats", {}), keep_zero=True),
        "stats": _compact_mapping(_field(pokemon, "stats", {}), keep_zero=True),
        "boosts": _compact_mapping(_field(pokemon, "boosts", {})),
        "speed_range": serialize_speed_range(pokemon),
        "moves": [serialize_move(move) for move in list(_field(pokemon, "moves", []) or [])],
        "moves_used_since_switch_in": _sorted_normalized_set(
            _field(pokemon, "moves_used_since_switch_in", set())
        ),
        "volatile_statuses": _sorted_normalized_set(
            _field(pokemon, "volatile_statuses", [])
        ),
        "volatile_status_durations": _compact_mapping(
            _field(pokemon, "volatile_status_durations", {})
        ),
        "rest_turns": _number(_field(pokemon, "rest_turns", 0)),
        "sleep_turns": _number(_field(pokemon, "sleep_turns", 0)),
        "substitute_hit": bool(_field(pokemon, "substitute_hit", False)),
        "terastallized": bool(_field(pokemon, "terastallized", False)),
        "tera_type": normalize_name(_field(pokemon, "tera_type", "")),
        "can_terastallize": bool(_field(pokemon, "can_terastallize", False)),
        "can_mega_evo": bool(_field(pokemon, "can_mega_evo", False)),
        "can_ultra_burst": bool(_field(pokemon, "can_ultra_burst", False)),
        "can_dynamax": bool(_field(pokemon, "can_dynamax", False)),
        "is_mega": bool(_field(pokemon, "is_mega", False)),
        "mega_name": normalize_name(_field(pokemon, "mega_name", "")),
        "knocked_off": bool(_field(pokemon, "knocked_off", False)),
        "unknown_forme": bool(_field(pokemon, "unknown_forme", False)),
        "forme_changed": bool(_field(pokemon, "forme_changed", False)),
        "zoroark_disguised_as": normalize_name(
            _field(pokemon, "zoroark_disguised_as", "")
        ),
        "can_have_choice_item": bool(_field(pokemon, "can_have_choice_item", True)),
        "impossible_items": _sorted_normalized_set(
            _field(pokemon, "impossible_items", set())
        ),
        "impossible_abilities": _sorted_normalized_set(
            _field(pokemon, "impossible_abilities", set())
        ),
        "hidden_power_possibilities": _sorted_normalized_set(
            _field(pokemon, "hidden_power_possibilities", set())
        ),
    }


def serialize_battler(battler: object) -> dict[str, Any]:
    reserve = list(_field(battler, "reserve", []) or [])
    pokemon = [_field(battler, "active")] + reserve

    return {
        "name": _jsonable(_field(battler, "name")),
        "account_name": _jsonable(_field(battler, "account_name")),
        "active": serialize_pokemon(_field(battler, "active")),
        "reserve": [serialize_pokemon(pokemon_slot) for pokemon_slot in reserve],
        "fainted_count": sum(
            1 for pokemon_slot in pokemon if pokemon_slot is not None and not _is_alive(pokemon_slot)
        ),
        "trapped": bool(_field(battler, "trapped", False)),
        "baton_passing": bool(_field(battler, "baton_passing", False)),
        "shed_tailing": bool(_field(battler, "shed_tailing", False)),
        "wish": _tuple_or_list(_field(battler, "wish", (0, 0))),
        "future_sight": _tuple_or_list(_field(battler, "future_sight", (0, ""))),
        "side_conditions": _compact_mapping(_field(battler, "side_conditions", {})),
        "last_selected_move": serialize_last_move(_field(battler, "last_selected_move")),
        "last_used_move": serialize_last_move(_field(battler, "last_used_move")),
        "has_team_dict": _field(battler, "team_dict") is not None,
    }


def serialize_battle_state(
    battle: object,
    *,
    include_request_json: bool = False,
    include_action_mask: bool = False,
) -> dict[str, Any]:
    """Serialize the current Battle into stable, JSON-ready state.

    The result is intentionally raw and loss-tolerant. It preserves enough
    information for later feature engineering without depending on Foul Play at
    training time.
    """

    state = {
        "schema_version": SCHEMA_VERSION,
        "battle_tag": _jsonable(_field(battle, "battle_tag")),
        "pokemon_format": normalize_name(_field(battle, "pokemon_format", "")),
        "generation": normalize_name(_field(battle, "generation", "")),
        "battle_type": _jsonable(_enum_name(_field(battle, "battle_type"))),
        "turn": _number(_field(battle, "turn", 0)),
        "started": bool(_field(battle, "started", False)),
        "team_preview": bool(_field(battle, "team_preview", False)),
        "rqid": _number(_field(battle, "rqid")),
        "force_switch": bool(_field(battle, "force_switch", False)),
        "wait": bool(_field(battle, "wait", False)),
        "time_remaining": _number(_field(battle, "time_remaining")),
        "weather": normalize_name(_field(battle, "weather", "")),
        "weather_turns_remaining": _number(
            _field(battle, "weather_turns_remaining", 0)
        ),
        "weather_source": normalize_name(_field(battle, "weather_source", "")),
        "field": normalize_name(_field(battle, "field", "")),
        "field_turns_remaining": _number(_field(battle, "field_turns_remaining", 0)),
        "trick_room": bool(_field(battle, "trick_room", False)),
        "trick_room_turns_remaining": _number(
            _field(battle, "trick_room_turns_remaining", 0)
        ),
        "gravity": bool(_field(battle, "gravity", False)),
        "user": serialize_battler(_field(battle, "user")),
        "opponent": serialize_battler(_field(battle, "opponent")),
    }

    if include_action_mask and legal_action_mask is not None:
        try:
            state["action_mask"] = legal_action_mask(battle)
        except Exception as exc:  # pragma: no cover - diagnostic path.
            state["action_mask_error"] = str(exc)

    if include_request_json:
        state["request_json"] = _jsonable(_field(battle, "request_json"))

    return state


def battle_state_json(battle: object, **kwargs: Any) -> str:
    """Return a deterministic JSON string for a serialized battle state."""

    return json.dumps(
        serialize_battle_state(battle, **kwargs),
        sort_keys=True,
        separators=(",", ":"),
    )

