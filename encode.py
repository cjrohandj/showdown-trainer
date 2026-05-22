"""Encode serialized battle states into fixed-length MLP feature vectors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from map import ACTION_SLOTS
except ImportError:  # pragma: no cover - useful when copied into another repo.
    ACTION_SLOTS = 13


ENCODER_SCHEMA_VERSION = 1

POKEMON_SLOTS = (
    "active",
    "reserve_1",
    "reserve_2",
    "reserve_3",
    "reserve_4",
    "reserve_5",
)
SIDES = ("user", "opponent")
MOVE_SLOTS = 4

STAT_KEYS = ("attack", "defense", "special-attack", "special-defense", "speed")
BASE_STAT_KEYS = ("hp",) + STAT_KEYS
EV_KEYS = ("hp", "attack", "defense", "special-attack", "special-defense", "speed")
BOOST_KEYS = (
    "attack",
    "defense",
    "special-attack",
    "special-defense",
    "speed",
    "accuracy",
    "evasion",
)
SIDE_CONDITION_KEYS = (
    "auroraveil",
    "healingwish",
    "lightscreen",
    "lunardance",
    "reflect",
    "safeguard",
    "spikes",
    "stealthrock",
    "stickyweb",
    "tailwind",
    "toxiccount",
    "toxicspikes",
    "wish",
)


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the first cheap MLP encoder."""

    hash_buckets: int = 4096
    include_action_mask: bool = True
    include_bias: bool = True


@dataclass(frozen=True)
class EncodedState:
    """A fixed feature vector plus enough metadata to train safely."""

    features: list[float]
    action_mask: list[int]
    scalar_names: list[str]
    hash_buckets: int
    schema_version: int = ENCODER_SCHEMA_VERSION

    @property
    def vector_size(self) -> int:
        return len(self.features)


def encode_json_state(
    json_state: str,
    *,
    config: EncoderConfig | None = None,
) -> EncodedState:
    """Encode a serialized battle state JSON string."""

    return encode_battle_state(json.loads(json_state), config=config)


def encode_battle_object(
    battle: object,
    *,
    config: EncoderConfig | None = None,
) -> EncodedState:
    """Serialize and encode a live Foul Play Battle-like object."""

    from serialize import serialize_battle_state

    return encode_battle_state(
        serialize_battle_state(battle, include_action_mask=True),
        config=config,
    )


def encode_battle_state(
    state: Mapping[str, Any],
    *,
    config: EncoderConfig | None = None,
) -> EncodedState:
    """Encode a serialized battle state into a fixed-length float vector."""

    config = config or EncoderConfig()
    tokens: list[tuple[str, float]] = []
    scalar_names: list[str] = []
    scalar_values: list[float] = []

    def add_scalar(name: str, value: object, scale: float = 1.0) -> None:
        scalar_names.append(name)
        scalar_values.append(_scaled(value, scale))

    def add_bool(name: str, value: object) -> None:
        scalar_names.append(name)
        scalar_values.append(1.0 if value else 0.0)

    def add_token(token: str, weight: float = 1.0) -> None:
        if token and not token.endswith("="):
            tokens.append((token, weight))

    if config.include_bias:
        add_scalar("bias", 1.0)

    add_token(f"format={_text(state.get('pokemon_format'))}")
    add_token(f"generation={_text(state.get('generation'))}")
    add_token(f"battle_type={_text(state.get('battle_type'))}")
    add_token(f"weather={_text(state.get('weather'))}")
    add_token(f"weather_source={_text(state.get('weather_source'))}")
    add_token(f"field={_text(state.get('field'))}")

    add_scalar("battle.turn", state.get("turn"), 100.0)
    add_bool("battle.started", state.get("started"))
    add_bool("battle.team_preview", state.get("team_preview"))
    add_bool("battle.force_switch", state.get("force_switch"))
    add_bool("battle.wait", state.get("wait"))
    add_scalar("battle.time_remaining", state.get("time_remaining"), 300.0)
    add_scalar(
        "battle.weather_turns_remaining",
        state.get("weather_turns_remaining"),
        8.0,
    )
    add_scalar("battle.field_turns_remaining", state.get("field_turns_remaining"), 8.0)
    add_bool("battle.trick_room", state.get("trick_room"))
    add_scalar(
        "battle.trick_room_turns_remaining",
        state.get("trick_room_turns_remaining"),
        5.0,
    )
    add_bool("battle.gravity", state.get("gravity"))

    for side_name in SIDES:
        side = _mapping(state.get(side_name))
        _encode_side(side_name, side, add_scalar, add_bool, add_token)

    action_mask = _action_mask(state)
    if config.include_action_mask:
        for index in range(ACTION_SLOTS):
            add_bool(f"action_mask.{index}", action_mask[index])

    hashed = _hash_tokens(tokens, config.hash_buckets)
    return EncodedState(
        features=hashed + scalar_values,
        action_mask=action_mask,
        scalar_names=scalar_names,
        hash_buckets=config.hash_buckets,
    )


def feature_size(config: EncoderConfig | None = None) -> int:
    """Return the encoder output length for this config."""

    empty_state = {
        "user": {},
        "opponent": {},
        "action_mask": [0] * ACTION_SLOTS,
    }
    return encode_battle_state(empty_state, config=config).vector_size


def _encode_side(side_name, side, add_scalar, add_bool, add_token) -> None:
    add_token(f"{side_name}.name={_text(side.get('name'))}")
    add_bool(f"{side_name}.trapped", side.get("trapped"))
    add_bool(f"{side_name}.baton_passing", side.get("baton_passing"))
    add_bool(f"{side_name}.shed_tailing", side.get("shed_tailing"))
    add_bool(f"{side_name}.has_team_dict", side.get("has_team_dict"))
    add_scalar(f"{side_name}.fainted_count", side.get("fainted_count"), 6.0)

    wish = _sequence(side.get("wish"), 2)
    add_scalar(f"{side_name}.wish.turns", wish[0], 5.0)
    add_scalar(f"{side_name}.wish.hp", wish[1], 500.0)

    future_sight = _sequence(side.get("future_sight"), 2)
    add_scalar(f"{side_name}.future_sight.turns", future_sight[0], 5.0)
    add_token(f"{side_name}.future_sight.source={_text(future_sight[1])}")

    for key in SIDE_CONDITION_KEYS:
        add_scalar(
            f"{side_name}.side_condition.{key}",
            _mapping(side.get("side_conditions")).get(key, 0),
            5.0,
        )
    for key, value in _mapping(side.get("side_conditions")).items():
        if value:
            add_token(f"{side_name}.side_condition={_text(key)}", _scaled(value, 5.0))

    for move_kind in ("last_selected_move", "last_used_move"):
        last_move = _mapping(side.get(move_kind))
        add_token(
            f"{side_name}.{move_kind}.pokemon={_text(last_move.get('pokemon_name'))}"
        )
        add_token(f"{side_name}.{move_kind}.move={_text(last_move.get('move'))}")
        add_scalar(f"{side_name}.{move_kind}.age_turn", last_move.get("turn"), 100.0)

    slots = [side.get("active")] + list(_sequence(side.get("reserve"), 5))
    for slot_name, pokemon in zip(POKEMON_SLOTS, slots, strict=True):
        _encode_pokemon_slot(
            f"{side_name}.{slot_name}",
            _mapping(pokemon),
            add_scalar,
            add_bool,
            add_token,
        )


def _encode_pokemon_slot(prefix, pokemon, add_scalar, add_bool, add_token) -> None:
    present = bool(pokemon)
    add_bool(f"{prefix}.present", present)

    add_token(f"{prefix}.name={_text(pokemon.get('name'))}")
    add_token(f"{prefix}.base_name={_text(pokemon.get('base_name'))}")
    add_token(f"{prefix}.status={_text(pokemon.get('status'))}")
    add_token(
        f"{prefix}.status_at_switch_in={_text(pokemon.get('status_at_switch_in'))}"
    )
    add_token(f"{prefix}.ability={_text(pokemon.get('ability'))}")
    add_token(f"{prefix}.original_ability={_text(pokemon.get('original_ability'))}")
    add_token(f"{prefix}.item={_text(pokemon.get('item'))}")
    add_token(f"{prefix}.removed_item={_text(pokemon.get('removed_item'))}")
    add_token(f"{prefix}.nature={_text(pokemon.get('nature'))}")
    add_token(f"{prefix}.tera_type={_text(pokemon.get('tera_type'))}")
    add_token(f"{prefix}.mega_name={_text(pokemon.get('mega_name'))}")
    add_token(
        f"{prefix}.zoroark_disguised_as={_text(pokemon.get('zoroark_disguised_as'))}"
    )

    for pokemon_type in _sequence(pokemon.get("types"), 2):
        add_token(f"{prefix}.type={_text(pokemon_type)}")

    add_scalar(f"{prefix}.level", pokemon.get("level"), 100.0)
    add_scalar(f"{prefix}.hp_fraction", pokemon.get("hp_fraction"))
    add_scalar(f"{prefix}.hp", pokemon.get("hp"), 500.0)
    add_scalar(f"{prefix}.max_hp", pokemon.get("max_hp"), 500.0)
    add_scalar(f"{prefix}.hp_at_switch_in", pokemon.get("hp_at_switch_in"), 500.0)
    add_bool(f"{prefix}.alive", pokemon.get("alive"))
    add_bool(f"{prefix}.fainted", pokemon.get("fainted"))
    add_bool(f"{prefix}.reviving", pokemon.get("reviving"))
    add_bool(f"{prefix}.item_inferred", pokemon.get("item_inferred"))
    add_bool(f"{prefix}.substitute_hit", pokemon.get("substitute_hit"))
    add_bool(f"{prefix}.terastallized", pokemon.get("terastallized"))
    add_bool(f"{prefix}.can_terastallize", pokemon.get("can_terastallize"))
    add_bool(f"{prefix}.can_mega_evo", pokemon.get("can_mega_evo"))
    add_bool(f"{prefix}.can_ultra_burst", pokemon.get("can_ultra_burst"))
    add_bool(f"{prefix}.can_dynamax", pokemon.get("can_dynamax"))
    add_bool(f"{prefix}.is_mega", pokemon.get("is_mega"))
    add_bool(f"{prefix}.knocked_off", pokemon.get("knocked_off"))
    add_bool(f"{prefix}.unknown_forme", pokemon.get("unknown_forme"))
    add_bool(f"{prefix}.forme_changed", pokemon.get("forme_changed"))
    add_bool(
        f"{prefix}.can_have_choice_item",
        pokemon.get("can_have_choice_item", True),
    )
    add_scalar(f"{prefix}.rest_turns", pokemon.get("rest_turns"), 5.0)
    add_scalar(f"{prefix}.sleep_turns", pokemon.get("sleep_turns"), 5.0)

    speed_range = _mapping(pokemon.get("speed_range"))
    add_scalar(f"{prefix}.speed_range.min", speed_range.get("min"), 1000.0)
    add_scalar(f"{prefix}.speed_range.max", speed_range.get("max"), 1000.0)
    add_bool(f"{prefix}.speed_range.unbounded_max", speed_range.get("unbounded_max"))

    stats = _mapping(pokemon.get("stats"))
    for key in STAT_KEYS:
        add_scalar(f"{prefix}.stats.{key}", stats.get(key), 500.0)

    base_stats = _mapping(pokemon.get("base_stats"))
    for key in BASE_STAT_KEYS:
        add_scalar(f"{prefix}.base_stats.{key}", base_stats.get(key), 255.0)

    evs = _sequence(pokemon.get("evs"), 6)
    for key, value in zip(EV_KEYS, evs, strict=True):
        add_scalar(f"{prefix}.evs.{key}", value, 252.0)

    boosts = _mapping(pokemon.get("boosts"))
    for key in BOOST_KEYS:
        add_scalar(f"{prefix}.boosts.{key}", boosts.get(key), 6.0)

    for key, value in _mapping(pokemon.get("volatile_status_durations")).items():
        if value:
            add_token(f"{prefix}.volatile_duration={_text(key)}", _scaled(value, 5.0))
    for key in _sequence(pokemon.get("volatile_statuses"), 32):
        add_token(f"{prefix}.volatile={_text(key)}")
    for key in _sequence(pokemon.get("moves_used_since_switch_in"), 32):
        add_token(f"{prefix}.move_used={_text(key)}")
    for key in _sequence(pokemon.get("impossible_items"), 64):
        add_token(f"{prefix}.impossible_item={_text(key)}")
    for key in _sequence(pokemon.get("impossible_abilities"), 64):
        add_token(f"{prefix}.impossible_ability={_text(key)}")
    for key in _sequence(pokemon.get("hidden_power_possibilities"), 32):
        add_token(f"{prefix}.hidden_power_possible={_text(key)}")

    moves = _sequence(pokemon.get("moves"), MOVE_SLOTS)
    for move_index, move in enumerate(moves):
        _encode_move_slot(
            f"{prefix}.move_{move_index + 1}",
            _mapping(move),
            add_scalar,
            add_bool,
            add_token,
        )


def _encode_move_slot(prefix, move, add_scalar, add_bool, add_token) -> None:
    add_bool(f"{prefix}.present", bool(move))
    add_token(f"{prefix}.name={_text(move.get('name'))}")

    current_pp = _float(move.get("current_pp"))
    max_pp = _float(move.get("max_pp"))
    add_scalar(
        f"{prefix}.pp_fraction",
        current_pp / max_pp if current_pp is not None and max_pp else 0.0,
    )
    add_scalar(f"{prefix}.current_pp", current_pp, 64.0)
    add_scalar(f"{prefix}.max_pp", max_pp, 64.0)
    add_bool(f"{prefix}.disabled", move.get("disabled"))
    add_bool(f"{prefix}.can_z", move.get("can_z"))


def _hash_tokens(tokens: list[tuple[str, float]], bucket_count: int) -> list[float]:
    if bucket_count <= 0:
        return []

    values = [0.0] * bucket_count

    for token, weight in tokens:
        index = _hash_index(token, bucket_count)
        values[index] += float(weight)
    return values


def _hash_index(token: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % bucket_count


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


def _text(value: object) -> str:
    if value is None:
        return ""
    return "".join(
        ch for ch in str(value).strip().lower() if ch.isalnum() or ch in "-_"
    )


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _scaled(value: object, scale: float = 1.0) -> float:
    parsed = _float(value)
    if parsed is None:
        return 0.0
    if scale:
        parsed /= scale
    return max(-5.0, min(5.0, parsed))


def _action_mask(state: Mapping[str, Any]) -> list[int]:
    mask = list(state.get("action_mask") or [])
    if mask:
        mask = [1 if value else 0 for value in mask[:ACTION_SLOTS]]
        return (mask + [0] * ACTION_SLOTS)[:ACTION_SLOTS]

    mask = [0] * ACTION_SLOTS
    user = _mapping(state.get("user"))
    active = _mapping(user.get("active"))

    for index, move in enumerate(_sequence(active.get("moves"), MOVE_SLOTS)):
        move = _mapping(move)
        current_pp = _float(move.get("current_pp"))
        disabled = bool(move.get("disabled"))
        if move.get("name") and not disabled and current_pp != 0:
            mask[index] = 1

    for index, pokemon in enumerate(_sequence(user.get("reserve"), 5)):
        pokemon = _mapping(pokemon)
        if not pokemon:
            continue
        hp = _float(pokemon.get("hp"))
        alive = bool(pokemon.get("alive", hp is None or hp > 0))
        fainted = bool(pokemon.get("fainted", hp == 0))
        if alive and not fainted:
            mask[4 + index] = 1

    if active.get("can_terastallize"):
        for index in range(MOVE_SLOTS):
            if mask[index]:
                mask[9 + index] = 1

    return mask
