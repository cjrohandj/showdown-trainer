"""Map Showdown / poke-engine MCTS actions into fixed MLP slots.

The MLP should predict small positional action IDs, not global move names.
For the first trainer, use this 13-slot action space:

    0..3   active move slots
    4..8   reserve switch slots
    9..12  terastallized active move slots
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


MOVE_SLOTS = 4
SWITCH_SLOTS = 5
TERA_MOVE_SLOTS = 4
ACTION_SLOTS = MOVE_SLOTS + SWITCH_SLOTS + TERA_MOVE_SLOTS

SWITCH_PREFIX = "switch "
TERA_SUFFIX = "-tera"
MEGA_SUFFIX = "-mega"


class ActionMappingError(ValueError):
    """Raised when an engine decision cannot be mapped to an action slot."""


@dataclass(frozen=True)
class SlotLayout:
    move_start: int = 0
    switch_start: int = MOVE_SLOTS
    tera_start: int = MOVE_SLOTS + SWITCH_SLOTS
    size: int = ACTION_SLOTS


DEFAULT_LAYOUT = SlotLayout()


def normalize_name(value: object) -> str:
    """Normalize Pokemon/Showdown identifiers for robust comparisons."""

    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _field(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _named(obj: object, *preferred_fields: str) -> str:
    if isinstance(obj, str):
        return obj

    for field in preferred_fields:
        value = _field(obj, field)
        if value not in (None, ""):
            return str(value)

    value = _field(obj, "name")
    if value not in (None, ""):
        return str(value)

    return str(obj)


def _user(battle: object) -> object:
    user = _field(battle, "user")
    if user is None:
        raise ActionMappingError("battle has no user side")
    return user


def active_moves(battle: object) -> list[object]:
    user = _user(battle)
    active = _field(user, "active")
    if active is None:
        return []
    return list(_field(active, "moves", []) or [])


def reserve_pokemon(battle: object) -> list[object]:
    return list(_field(_user(battle), "reserve", []) or [])


def _decision_base(decision: str) -> tuple[str, bool, bool]:
    decision = decision.strip()
    is_tera = decision.endswith(TERA_SUFFIX)
    is_mega = decision.endswith(MEGA_SUFFIX)

    if is_tera:
        decision = decision[: -len(TERA_SUFFIX)]
    elif is_mega:
        decision = decision[: -len(MEGA_SUFFIX)]

    return decision, is_tera, is_mega


def decision_to_action_slot(
    battle: object,
    decision: str,
    *,
    layout: SlotLayout = DEFAULT_LAYOUT,
    collapse_mega: bool = True,
) -> int:
    """Convert an engine decision string to a fixed action slot.

    Examples:
        "earthquake" -> move slot matching active.moves
        "earthquake-tera" -> tera move slot matching active.moves
        "switch dragapult" -> switch slot matching user.reserve

    Mega decisions are collapsed onto their base move slot by default because
    this first action space is designed for a cheap gen9/randbats MLP.
    """

    decision = decision.strip()
    if decision.startswith(SWITCH_PREFIX):
        target = normalize_name(decision.removeprefix(SWITCH_PREFIX))
        for index, pokemon in enumerate(reserve_pokemon(battle)):
            if index >= SWITCH_SLOTS:
                break
            if normalize_name(_named(pokemon, "name", "species")) == target:
                return layout.switch_start + index
        raise ActionMappingError(f"unknown switch target: {decision!r}")

    base_decision, is_tera, is_mega = _decision_base(decision)
    if is_mega and not collapse_mega:
        raise ActionMappingError(
            f"mega decision needs a larger action space: {decision!r}"
        )

    move_key = normalize_name(base_decision)
    for index, move in enumerate(active_moves(battle)):
        if index >= MOVE_SLOTS:
            break
        if normalize_name(_named(move, "id", "name", "move")) == move_key:
            return (layout.tera_start if is_tera else layout.move_start) + index

    raise ActionMappingError(f"unknown move decision: {decision!r}")


def action_slot_to_decision(
    battle: object,
    slot: int,
    *,
    layout: SlotLayout = DEFAULT_LAYOUT,
) -> str:
    """Convert a predicted action slot back into an engine decision string."""

    if slot < 0 or slot >= layout.size:
        raise ActionMappingError(f"slot out of range: {slot}")

    moves = active_moves(battle)
    reserve = reserve_pokemon(battle)

    if layout.move_start <= slot < layout.switch_start:
        move_index = slot - layout.move_start
        if move_index >= len(moves):
            raise ActionMappingError(f"empty move slot: {slot}")
        return normalize_name(_named(moves[move_index], "id", "name", "move"))

    if layout.switch_start <= slot < layout.tera_start:
        switch_index = slot - layout.switch_start
        if switch_index >= len(reserve):
            raise ActionMappingError(f"empty switch slot: {slot}")
        return f"{SWITCH_PREFIX}{normalize_name(_named(reserve[switch_index], 'name', 'species'))}"

    tera_index = slot - layout.tera_start
    if tera_index >= len(moves):
        raise ActionMappingError(f"empty tera move slot: {slot}")
    return f"{normalize_name(_named(moves[tera_index], 'id', 'name', 'move'))}{TERA_SUFFIX}"


def legal_action_mask(
    battle: object,
    *,
    layout: SlotLayout = DEFAULT_LAYOUT,
    include_tera: bool = True,
) -> list[int]:
    """Return a 0/1 mask for legal slots under the current observed state."""

    mask = [0] * layout.size
    user = _user(battle)
    active = _field(user, "active")

    if active is not None:
        for index, move in enumerate(active_moves(battle)[:MOVE_SLOTS]):
            disabled = bool(_field(move, "disabled", False))
            pp = _field(move, "current_pp", _field(move, "pp", 1))
            if not disabled and pp != 0:
                mask[layout.move_start + index] = 1

        can_tera = bool(_field(active, "can_terastallize", False))
        if include_tera and can_tera:
            for index in range(MOVE_SLOTS):
                if mask[layout.move_start + index]:
                    mask[layout.tera_start + index] = 1

    for index, pokemon in enumerate(reserve_pokemon(battle)[:SWITCH_SLOTS]):
        hp = _field(pokemon, "hp", 1)
        fainted = callable(getattr(pokemon, "is_alive", None)) and not pokemon.is_alive()
        if hp != 0 and not fainted:
            mask[layout.switch_start + index] = 1

    return mask


def mcts_policy_to_vector(
    battle: object,
    mcts_policy: Mapping[str, float],
    *,
    layout: SlotLayout = DEFAULT_LAYOUT,
    strict: bool = True,
) -> list[float]:
    """Map an MCTS decision policy dict to a normalized fixed-size target vector."""

    target = [0.0] * layout.size
    for decision, probability in mcts_policy.items():
        try:
            slot = decision_to_action_slot(battle, decision, layout=layout)
        except ActionMappingError:
            if strict:
                raise
            continue
        target[slot] += float(probability)

    total = sum(target)
    if total:
        target = [value / total for value in target]

    return target


def aggregate_mcts_policy(mcts_results: Iterable[object]) -> dict[str, float]:
    """Aggregate poke-engine MCTS results into a decision-probability dict.

    Accepts either raw MctsResult-like objects or tuples shaped like Foul Play's
    ``(mcts_result, sample_chance, index)`` entries.
    """

    policy: dict[str, float] = {}

    for item in mcts_results:
        sample_chance = 1.0
        result = item
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            result = item[0]
            if len(item) > 1:
                sample_chance = float(item[1])

        total_visits = float(_field(result, "total_visits", 0) or 0)
        if total_visits <= 0:
            continue

        for option in _field(result, "side_one", []) or []:
            decision = str(_field(option, "move_choice"))
            visits = float(_field(option, "visits", 0) or 0)
            policy[decision] = policy.get(decision, 0.0) + (
                sample_chance * visits / total_visits
            )

    total = sum(policy.values())
    if total:
        policy = {decision: probability / total for decision, probability in policy.items()}

    return dict(sorted(policy.items(), key=lambda item: item[1], reverse=True))


def mcts_results_to_target_vector(
    battle: object,
    mcts_results: Iterable[object],
    *,
    layout: SlotLayout = DEFAULT_LAYOUT,
    strict: bool = True,
) -> list[float]:
    """Aggregate raw MCTS results and map them directly to an MLP target vector."""

    return mcts_policy_to_vector(
        battle,
        aggregate_mcts_policy(mcts_results),
        layout=layout,
        strict=strict,
    )
