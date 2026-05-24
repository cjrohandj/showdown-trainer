#!/usr/bin/env python3
"""Summarize a Showdown trajectory shard as a P1-view action table."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DECISION_RECORD = "decision"
RESULT_RECORD = "result"
STAT_ORDER = ("hp", "attack", "defense", "special-attack", "special-defense", "speed")
STAT_SHORT = {
    "hp": "HP",
    "attack": "Atk",
    "defense": "Def",
    "special-attack": "SpA",
    "special-defense": "SpD",
    "speed": "Spe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize each battle in a Showdown shard JSONL."
    )
    parser.add_argument("input_path", help="Path to a .jsonl or .jsonl.gz shard")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = summarize_shard(Path(args.input_path))

    if args.format == "json":
        rendered = json.dumps(summaries, indent=2, sort_keys=True)
    else:
        rendered = render_markdown(summaries)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")


def summarize_shard(path: Path) -> list[dict[str, Any]]:
    battles: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for index, record in enumerate(iter_jsonl(path)):
        battle_id = str(record.get("battle_id") or "")
        if not battle_id:
            continue

        battle = battles.setdefault(
            battle_id,
            {
                "battle_id": battle_id,
                "records": [],
                "winner": None,
                "run_id": record.get("run_id"),
            },
        )

        record_type = str(record.get("record_type") or "")
        if record_type == RESULT_RECORD:
            battle["winner"] = record.get("winner")
            continue
        if record_type != DECISION_RECORD and "state" not in record:
            continue

        battle["records"].append({"index": index, "record": record})

    return [build_game_summary(battle) for battle in battles.values()]


def build_game_summary(battle: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(
        battle["records"],
        key=lambda item: (
            _int(_mapping(item["record"].get("metadata")).get("ply"), item["index"]),
            item["index"],
        ),
    )

    p1_rows = [item for item in rows if _actor(item["record"]) == "p1"]
    p1_by_ply: dict[int, dict[str, Any]] = {}
    p2_by_ply: dict[int, dict[str, Any]] = {}
    for item in rows:
        ply = _int(_mapping(item["record"].get("metadata")).get("ply"), item["index"])
        actor = _actor(item["record"])
        if actor == "p1":
            p1_by_ply.setdefault(ply, item["record"])
        elif actor == "p2":
            p2_by_ply.setdefault(ply, item["record"])

    first_p1_state = _mapping(p1_rows[0]["record"].get("state")) if p1_rows else {}
    team = _team_snapshot(first_p1_state.get("user"))
    items = _unique_items(team)

    action_rows: list[dict[str, Any]] = []
    p1_states_by_ply = {
        _int(_mapping(item["record"].get("metadata")).get("ply"), item["index"]): _mapping(item["record"].get("state"))
        for item in p1_rows
    }
    ordered_plys = sorted(set(p1_by_ply) | set(p2_by_ply))

    for ply in ordered_plys:
        p1_record = p1_by_ply.get(ply)
        p2_record = p2_by_ply.get(ply)
        previous_p1_state = _previous_p1_state_before_ply(p1_states_by_ply, ply)
        next_p1_state = _next_p1_state_after_ply(p1_states_by_ply, ply)

        if p1_record:
            state = _mapping(p1_record.get("state"))
            action_rows.append(
                {
                    "turn": state.get("turn"),
                    "ply": ply,
                    "before_view": format_battle_view(state),
                    "p1_action": format_action(p1_record),
                    "p2_action": format_action(p2_record) if p2_record else "—",
                    "next_view": format_battle_view(next_p1_state) if next_p1_state else "Not logged",
                }
            )
            continue

        if p2_record and _request_type(p2_record) == "switch":
            state = _mapping(p2_record.get("state"))
            p1_view_state = _p1_view_from_p2_record(p2_record, previous_p1_state) or next_p1_state
            action_rows.append(
                {
                    "turn": state.get("turn"),
                    "ply": ply,
                    "before_view": format_battle_view(p1_view_state) if p1_view_state else "Not logged",
                    "p1_action": "—",
                    "p2_action": format_action(p2_record),
                    "next_view": format_battle_view(next_p1_state) if next_p1_state else "Not logged",
                }
            )

    return {
        "battle_id": battle["battle_id"],
        "winner": battle.get("winner"),
        "run_id": battle.get("run_id"),
        "p1_team": team,
        "p1_items": items,
        "actions": action_rows,
    }


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, game in enumerate(summaries, start=1):
        lines.append(f"# Game {index}: {game['battle_id']}")
        if game.get("winner") is not None:
            lines.append(f"Winner: {game['winner']}")
        lines.append("")
        lines.append("## P1 visible team")
        if game["p1_items"]:
            lines.append(f"Items: {', '.join(game['p1_items'])}")
            lines.append("")
        lines.append("| Slot | Pokemon | HP | Status | Ability | Item | Stats | EVs | IVs |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for slot_index, pokemon in enumerate(game["p1_team"], start=1):
            lines.append(
                "| {slot} | {name} | {hp} | {status} | {ability} | {item} | {stats} | {evs} | {ivs} |".format(
                    slot=slot_index,
                    name=escape_cell(str(pokemon.get("name") or "")),
                    hp=escape_cell(format_hp(pokemon)),
                    status=escape_cell(str(pokemon.get("status") or "none")),
                    ability=escape_cell(str(pokemon.get("ability") or "none")),
                    item=escape_cell(str(pokemon.get("item") or "none")),
                    stats=escape_cell(format_stat_line(pokemon.get("stats"))),
                    evs=escape_cell(format_stat_line(pokemon.get("evs"))),
                    ivs=escape_cell(format_stat_line(pokemon.get("ivs"))),
                )
            )

        lines.append("")
        lines.append("## Action log")
        lines.append(
            "| # | Turn | P1 view before action | P1 action | P2 action | Next P1 logged view |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row_index, row in enumerate(game["actions"]):
            lines.append(
                "| {idx} | {turn} | {before} | {p1} | {p2} | {next} |".format(
                    idx=row_index,
                    turn=escape_cell(str(row.get("turn") or "")),
                    before=escape_cell(str(row.get("before_view") or "")),
                    p1=escape_cell(str(row.get("p1_action") or "—")),
                    p2=escape_cell(str(row.get("p2_action") or "—")),
                    next=escape_cell(str(row.get("next_view") or "Not logged")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_battle_view(state: Mapping[str, Any]) -> str:
    user = _mapping(state.get("user"))
    opponent = _mapping(state.get("opponent"))
    return f"{format_active_view(_mapping(user.get('active')))} vs {format_active_view(_mapping(opponent.get('active')))}"


def format_active_view(pokemon: Mapping[str, Any]) -> str:
    if not pokemon:
        return "-"
    name = pretty_name(str(pokemon.get("name") or pokemon.get("base_name") or ""))
    if not name:
        return "-"
    hp = _int(pokemon.get("hp"), 0)
    max_hp = _int(pokemon.get("max_hp"), 0)
    hp_fraction = _float(pokemon.get("hp_fraction"))
    status = str(pokemon.get("status") or "none")
    if _is_fainted(pokemon):
        hp_text = "0% fnt"
    elif max_hp == 100:
        hp_text = f"{round(100 * (hp_fraction if hp_fraction is not None else hp / 100.0))}%"
    elif max_hp > 0:
        hp_text = f"{hp}/{max_hp}"
    else:
        hp_text = str(hp)
    parts = [name, hp_text]
    if status and status != "none" and status != "fnt":
        parts.append(status)
    boosts = format_boosts(_mapping(pokemon.get("boosts")))
    if boosts:
        parts.append(boosts)
    return " ".join(parts)


def format_action(record: Mapping[str, Any] | None) -> str:
    if not record:
        return "—"
    action = str(record.get("chosen_action") or _mapping(record.get("metadata")).get("showdown_choice") or "").strip()
    if not action or action in {"wait", "pass"}:
        return "—"
    if action.startswith("switch "):
        return f"→ {pretty_name(action.removeprefix('switch ').strip())}"
    if action.endswith("-tera"):
        return f"{pretty_name(action.removesuffix('-tera'))} (tera)"
    return pretty_action(action)


def pretty_action(value: str) -> str:
    text = value.replace("-", " ").replace("_", " ").strip()
    if not text:
        return "—"
    return " ".join(part.capitalize() for part in text.split())


def pretty_name(value: str) -> str:
    text = value.replace("_", "-").strip()
    if not text:
        return ""
    parts = text.split("-")
    return "-".join(part[:1].upper() + part[1:] for part in parts if part)


def format_boosts(boosts: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for stat in ("attack", "defense", "special-attack", "special-defense", "speed"):
        value = _int(boosts.get(stat), 0)
        if not value:
            continue
        label = STAT_SHORT[stat]
        sign = "+" if value > 0 else ""
        parts.append(f"{label}{sign}{value}")
    return " ".join(parts)


def format_hp(pokemon: Mapping[str, Any]) -> str:
    if _is_fainted(pokemon):
        return "0% fnt"
    hp = _int(pokemon.get("hp"), 0)
    max_hp = _int(pokemon.get("max_hp"), 0)
    hp_fraction = _float(pokemon.get("hp_fraction"))
    if max_hp == 100 and hp_fraction is not None:
        return f"{round(hp_fraction * 100)}%"
    if max_hp > 0:
        return f"{hp}/{max_hp}"
    return str(hp)


def format_stat_line(values: object) -> str:
    source = _mapping(values)
    if source:
        parts = [_int(source.get(stat), 0) for stat in STAT_ORDER]
    else:
        parts = [_int(item, 0) for item in _sequence(values, len(STAT_ORDER))]
    return "/".join(str(value) for value in parts)


def _team_snapshot(side: object) -> list[dict[str, Any]]:
    side_map = _mapping(side)
    pokemon = [_mapping(side_map.get("active"))] + [_mapping(entry) for entry in _sequence(side_map.get("reserve"), 5)]
    return [entry for entry in pokemon if entry]


def _unique_items(team: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for pokemon in team:
        item = str(pokemon.get("item") or "").strip()
        if not item or item == "none" or item in seen:
            continue
        seen.append(item)
    return seen


def _actor(record: Mapping[str, Any]) -> str:
    return str(_mapping(record.get("metadata")).get("actor") or "")


def _request_type(record: Mapping[str, Any]) -> str:
    return str(_mapping(record.get("metadata")).get("request_type") or "")


def _next_p1_state_after_ply(
    p1_states_by_ply: Mapping[int, Mapping[str, Any]],
    ply: int,
) -> dict[str, Any]:
    for candidate_ply in sorted(p1_states_by_ply):
        if candidate_ply > ply:
            return _mapping(p1_states_by_ply[candidate_ply])
    return {}


def _previous_p1_state_before_ply(
    p1_states_by_ply: Mapping[int, Mapping[str, Any]],
    ply: int,
) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for candidate_ply in sorted(p1_states_by_ply):
        if candidate_ply >= ply:
            break
        previous = _mapping(p1_states_by_ply[candidate_ply])
    return previous


def _p1_view_from_p2_record(
    record: Mapping[str, Any],
    previous_p1_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = _mapping(record.get("state"))
    p1_side = _mapping(_mapping(previous_p1_state).get("user")) or _mapping(state.get("opponent"))
    p2_side = _mapping(state.get("user"))
    active = _mapping(p2_side.get("active"))
    chosen_action = str(record.get("chosen_action") or "").strip()

    if chosen_action.startswith("switch "):
        target = chosen_action.removeprefix("switch ").strip()
        for pokemon in _team_snapshot(p2_side):
            name = str(pokemon.get("name") or pokemon.get("base_name") or "").strip()
            if name == target:
                active = pokemon
                break

    return {
        "user": p1_side,
        "opponent": {"active": active},
    }


def _is_fainted(pokemon: Mapping[str, Any]) -> bool:
    if pokemon.get("fainted") is not None:
        return bool(pokemon.get("fainted"))
    if pokemon.get("alive") is not None:
        return not bool(pokemon.get("alive"))
    status = str(pokemon.get("status") or "")
    if status == "fnt":
        return True
    hp_fraction = _float(pokemon.get("hp_fraction"))
    return hp_fraction is not None and hp_fraction <= 0.0


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def escape_cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ").strip()


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    main()
