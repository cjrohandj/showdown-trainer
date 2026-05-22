"""JSONL writer for MCTS-to-policy-network training data."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from map import (
    ACTION_SLOTS,
    ActionMappingError,
    action_slot_to_decision,
    decision_to_action_slot,
    mcts_policy_to_vector,
)
from serialize import serialize_battle_state


DATASET_SCHEMA_VERSION = 1
DECISION_RECORD = "decision"
RESULT_RECORD = "result"
RUN_RECORD = "run"


@dataclass(frozen=True)
class DecisionRecord:
    """One supervised policy example produced by MCTS."""

    data: dict[str, Any]

    @property
    def example_id(self) -> str:
        return str(self.data["example_id"])

    @property
    def battle_id(self) -> str:
        return str(self.data["battle_id"])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_decision_record(
    *,
    battle: object | None = None,
    state: Mapping[str, Any] | None = None,
    search_result: object | Mapping[str, Any] | None = None,
    mcts_policy: Mapping[str, float] | None = None,
    chosen_action: str | None = None,
    run_id: str | None = None,
    battle_id: str | None = None,
    example_id: str | None = None,
    winner: str | None = None,
    strict: bool = True,
    include_state: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> DecisionRecord:
    """Build a JSON-ready decision training record.

    Pass either a live Foul Play ``battle`` or an already serialized ``state``.
    Pass either ``search_result`` with ``chosen_action``/``mcts_policy`` fields,
    or pass those two values explicitly.
    """

    if state is None:
        if battle is None:
            raise ValueError("build_decision_record requires battle or state")
        state = serialize_battle_state(battle, include_action_mask=True)
    else:
        state = dict(state)

    chosen_action = chosen_action or _field(search_result, "chosen_action")
    mcts_policy = dict(mcts_policy or _field(search_result, "mcts_policy", {}) or {})
    considered_policy = dict(_field(search_result, "considered_policy", {}) or {})

    if not chosen_action:
        raise ValueError("chosen_action is required")
    if not mcts_policy:
        raise ValueError("mcts_policy is required")

    battle_id = battle_id or str(state.get("battle_tag") or "")
    if not battle_id:
        battle_id = f"battle-{uuid.uuid4().hex}"

    mcts_target = mcts_policy_to_vector(state, mcts_policy, strict=strict)
    chosen_action_slot = decision_to_action_slot(state, chosen_action)
    action_mask = _action_mask(state)
    slot_to_decision = build_slot_to_decision(state, action_mask)

    record = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "record_type": DECISION_RECORD,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "battle_id": battle_id,
        "example_id": example_id or uuid.uuid4().hex,
        "turn": state.get("turn"),
        "pokemon_format": state.get("pokemon_format"),
        "generation": state.get("generation"),
        "chosen_action": chosen_action,
        "chosen_action_slot": chosen_action_slot,
        "action_mask": action_mask,
        "mcts_policy": dict(
            sorted(mcts_policy.items(), key=lambda item: item[1], reverse=True)
        ),
        "mcts_target": mcts_target,
        "slot_to_decision": slot_to_decision,
        "winner": winner,
        "metadata": dict(metadata or {}),
    }

    if considered_policy:
        record["considered_policy"] = dict(
            sorted(considered_policy.items(), key=lambda item: item[1], reverse=True)
        )

    if include_state:
        record["state"] = state

    return DecisionRecord(record)


def build_result_record(
    *,
    battle_id: str,
    winner: str | None,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an append-only result row for all examples from a battle."""

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "record_type": RESULT_RECORD,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "battle_id": battle_id,
        "winner": winner,
        "metadata": dict(metadata or {}),
    }


def build_run_record(
    *,
    run_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a row describing the collection run."""

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "record_type": RUN_RECORD,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "metadata": dict(metadata or {}),
    }


class TrainingDatasetWriter:
    """Append-only JSONL writer for policy-training examples."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        write_run_record: bool = True,
        run_metadata: Mapping[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex

        if write_run_record:
            self.append_record(
                build_run_record(run_id=self.run_id, metadata=run_metadata)
            )

    def write_decision(
        self,
        *,
        battle: object | None = None,
        state: Mapping[str, Any] | None = None,
        search_result: object | Mapping[str, Any] | None = None,
        mcts_policy: Mapping[str, float] | None = None,
        chosen_action: str | None = None,
        winner: str | None = None,
        strict: bool = True,
        include_state: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> DecisionRecord:
        record = build_decision_record(
            battle=battle,
            state=state,
            search_result=search_result,
            mcts_policy=mcts_policy,
            chosen_action=chosen_action,
            run_id=self.run_id,
            winner=winner,
            strict=strict,
            include_state=include_state,
            metadata=metadata,
        )
        self.append_record(record.data)
        return record

    def write_result(
        self,
        *,
        battle_id: str,
        winner: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = build_result_record(
            battle_id=battle_id,
            winner=winner,
            run_id=self.run_id,
            metadata=metadata,
        )
        self.append_record(record)
        return record

    def append_record(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            file.write("\n")


def build_slot_to_decision(
    state: Mapping[str, Any],
    action_mask: Iterable[int] | None = None,
) -> dict[str, str]:
    """Build a readable map from legal slot indexes back to decisions."""

    action_mask = list(action_mask or _action_mask(state))
    slot_to_decision: dict[str, str] = {}

    for slot, is_legal in enumerate(action_mask[:ACTION_SLOTS]):
        if not is_legal:
            continue
        try:
            slot_to_decision[str(slot)] = action_slot_to_decision(state, slot)
        except ActionMappingError:
            continue

    return slot_to_decision


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield records from a JSONL dataset."""

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_decision_records(path: str | Path) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(path):
        if record.get("record_type") == DECISION_RECORD:
            yield record


def load_battle_results(path: str | Path) -> dict[str, str | None]:
    """Load battle_id -> winner from result records."""

    results: dict[str, str | None] = {}
    for record in iter_jsonl(path):
        if record.get("record_type") == RESULT_RECORD:
            results[str(record["battle_id"])] = record.get("winner")
    return results


def attach_results(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield decision records with winner filled from later result rows."""

    results = load_battle_results(path)
    for record in iter_decision_records(path):
        if record.get("winner") is None:
            record = dict(record)
            record["winner"] = results.get(str(record["battle_id"]))
        yield record


def _action_mask(state: Mapping[str, Any]) -> list[int]:
    mask = list(state.get("action_mask") or [])
    if mask:
        mask = [1 if value else 0 for value in mask[:ACTION_SLOTS]]
        return (mask + [0] * ACTION_SLOTS)[:ACTION_SLOTS]

    mask = [0] * ACTION_SLOTS
    user = _mapping(state.get("user"))
    active = _mapping(user.get("active"))

    for index, move in enumerate(_sequence(active.get("moves"), 4)):
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
        for index in range(4):
            if mask[index]:
                mask[9 + index] = 1

    return mask


def _field(obj: object, name: str, default: object = None) -> object:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


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


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
