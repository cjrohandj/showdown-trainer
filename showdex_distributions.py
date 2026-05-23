"""Load and sample Showdex-style Pokemon set distributions.

Showdex fetches random battle sets and usage distributions from the pkmn data
repository. This module keeps the small subset we need for offline MCTS:
species, level, ability/item/move/tera alternatives, and EV/IV tables.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAT_ORDER = ("hp", "attack", "defense", "special-attack", "special-defense", "speed")
SHOWDOWN_STAT_KEYS = {
    "hp": "hp",
    "atk": "attack",
    "def": "defense",
    "spa": "special-attack",
    "spd": "special-defense",
    "spe": "speed",
}
DEFAULT_RANDOMS_EVS = {
    "hp": 85,
    "attack": 85,
    "defense": 85,
    "special-attack": 85,
    "special-defense": 85,
    "speed": 85,
}
DEFAULT_IVS = {
    "hp": 31,
    "attack": 31,
    "defense": 31,
    "special-attack": 31,
    "special-defense": 31,
    "speed": 31,
}


@dataclass(frozen=True)
class WeightedValue:
    value: str
    weight: float = 1.0


@dataclass(frozen=True)
class SampledPokemon:
    species: str
    level: int
    ability: str
    item: str
    moves: tuple[str, ...]
    tera_type: str
    evs: dict[str, int]
    ivs: dict[str, int]
    role: str | None = None
    distributions: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class PokemonDistribution:
    species: str
    level: int
    abilities: tuple[WeightedValue, ...]
    items: tuple[WeightedValue, ...]
    moves: tuple[WeightedValue, ...]
    tera_types: tuple[WeightedValue, ...] = ()
    evs: dict[str, int] | None = None
    ivs: dict[str, int] | None = None
    role: str | None = None
    usage: float = 1.0

    def sample(self, rng: random.Random) -> SampledPokemon:
        moves = _weighted_sample_without_replacement(
            rng,
            self.moves or (WeightedValue("tackle"),),
            k=4,
        )
        return SampledPokemon(
            species=normalize_id(self.species),
            level=int(self.level or 100),
            ability=_sample_one(rng, self.abilities, default="none"),
            item=_sample_one(rng, self.items, default="none"),
            moves=tuple(moves),
            tera_type=_sample_one(rng, self.tera_types, default="typeless"),
            evs=dict(DEFAULT_RANDOMS_EVS | (self.evs or {})),
            ivs=dict(DEFAULT_IVS | (self.ivs or {})),
            role=self.role,
            distributions=self.to_distribution_maps(),
        )

    def observed(self, *, reveal_set: bool = False) -> SampledPokemon:
        """Return the Pokemon as an observed slot with distributions attached."""

        sampled = self.sample(random.Random(0))
        if reveal_set:
            return sampled
        return SampledPokemon(
            species=normalize_id(self.species),
            level=int(self.level or 100),
            ability="",
            item="",
            moves=(),
            tera_type="typeless",
            evs=dict(DEFAULT_RANDOMS_EVS | (self.evs or {})),
            ivs=dict(DEFAULT_IVS | (self.ivs or {})),
            role=self.role,
            distributions=self.to_distribution_maps(),
        )

    def to_distribution_maps(self) -> dict[str, dict[str, float]]:
        return {
            "ability_distribution": _normalized_weight_map(self.abilities),
            "item_distribution": _normalized_weight_map(self.items),
            "move_distribution": _normalized_weight_map(self.moves),
            "tera_type_distribution": _normalized_weight_map(self.tera_types),
        }


class ShowdexCatalog:
    """A sampled catalog of Showdex-compatible presets."""

    def __init__(self, entries: Iterable[PokemonDistribution]):
        self.entries = [entry for entry in entries if entry.moves]
        if not self.entries:
            raise ValueError("ShowdexCatalog requires at least one entry with moves")

    @classmethod
    def from_files(
        cls,
        *,
        randoms_stats_path: str | Path | None = None,
        randoms_preset_path: str | Path | None = None,
        prefer_stats: bool = True,
    ) -> "ShowdexCatalog":
        stats_path = Path(randoms_stats_path).expanduser() if randoms_stats_path else None
        preset_path = Path(randoms_preset_path).expanduser() if randoms_preset_path else None

        if prefer_stats and stats_path and stats_path.exists():
            return cls.from_randoms_stats(_read_json(stats_path))
        if preset_path and preset_path.exists():
            return cls.from_randoms_presets(_read_json(preset_path))
        if stats_path and stats_path.exists():
            return cls.from_randoms_stats(_read_json(stats_path))

        attempted = [str(path) for path in (stats_path, preset_path) if path]
        raise FileNotFoundError(
            "No Showdex/pkmn data file found. Tried: " + ", ".join(attempted)
        )

    @classmethod
    def from_randoms_stats(cls, response: Mapping[str, Any]) -> "ShowdexCatalog":
        entries: list[PokemonDistribution] = []
        for species, stats in sorted(response.items()):
            if not isinstance(stats, Mapping):
                continue
            root = dict(stats)
            roles = _mapping(root.get("roles"))
            if roles:
                for role_name, role_stats in sorted(roles.items()):
                    role = _mapping(role_stats)
                    entries.append(
                        PokemonDistribution(
                            species=species,
                            level=int(root.get("level") or 100),
                            abilities=_weighted_values(
                                role.get("abilities") or root.get("abilities")
                            ),
                            items=_weighted_values(role.get("items") or root.get("items")),
                            moves=_weighted_values(role.get("moves")),
                            tera_types=_weighted_values(role.get("teraTypes")),
                            evs=_stats_table(role.get("evs") or root.get("evs"), DEFAULT_RANDOMS_EVS),
                            ivs=_stats_table(role.get("ivs") or root.get("ivs"), DEFAULT_IVS),
                            role=str(role_name),
                            usage=float(role.get("weight") or 1.0),
                        )
                    )
            else:
                entries.append(
                    PokemonDistribution(
                        species=species,
                        level=int(root.get("level") or 100),
                        abilities=_weighted_values(root.get("abilities")),
                        items=_weighted_values(root.get("items")),
                        moves=_weighted_values(root.get("moves")),
                        tera_types=_weighted_values(root.get("teraTypes")),
                        evs=_stats_table(root.get("evs"), DEFAULT_RANDOMS_EVS),
                        ivs=_stats_table(root.get("ivs"), DEFAULT_IVS),
                    )
                )
        return cls(entries)

    @classmethod
    def from_randoms_presets(cls, response: Mapping[str, Any]) -> "ShowdexCatalog":
        entries: list[PokemonDistribution] = []
        for species, preset in sorted(response.items()):
            if not isinstance(preset, Mapping):
                continue
            root = dict(preset)
            roles = _mapping(root.get("roles"))
            if roles:
                for role_name, role_preset in sorted(roles.items()):
                    role = _mapping(role_preset)
                    entries.append(
                        PokemonDistribution(
                            species=species,
                            level=int(root.get("level") or 100),
                            abilities=_weighted_values(
                                role.get("abilities") or root.get("abilities")
                            ),
                            items=_weighted_values(role.get("items") or root.get("items")),
                            moves=_weighted_values(role.get("moves")),
                            tera_types=_weighted_values(role.get("teraTypes")),
                            evs=_stats_table(role.get("evs") or root.get("evs"), DEFAULT_RANDOMS_EVS),
                            ivs=_stats_table(role.get("ivs") or root.get("ivs"), DEFAULT_IVS),
                            role=str(role_name),
                        )
                    )
            else:
                entries.append(
                    PokemonDistribution(
                        species=species,
                        level=int(root.get("level") or 100),
                        abilities=_weighted_values(root.get("abilities")),
                        items=_weighted_values(root.get("items")),
                        moves=_weighted_values(root.get("moves")),
                        evs=_stats_table(root.get("evs"), DEFAULT_RANDOMS_EVS),
                        ivs=_stats_table(root.get("ivs"), DEFAULT_IVS),
                    )
                )
        return cls(entries)

    @classmethod
    def fixture(cls) -> "ShowdexCatalog":
        """Small deterministic catalog for offline smoke tests."""

        return cls(
            [
                PokemonDistribution(
                    species="charmander",
                    level=100,
                    abilities=(WeightedValue("blaze", 0.8), WeightedValue("solarpower", 0.2)),
                    items=(WeightedValue("charcoal", 0.7), WeightedValue("eviolite", 0.3)),
                    moves=(
                        WeightedValue("ember", 0.9),
                        WeightedValue("tackle", 0.8),
                        WeightedValue("quickattack", 0.6),
                        WeightedValue("leer", 0.3),
                    ),
                    tera_types=(WeightedValue("fire", 0.7), WeightedValue("normal", 0.3)),
                    usage=1.0,
                ),
                PokemonDistribution(
                    species="squirtle",
                    level=100,
                    abilities=(WeightedValue("torrent", 1.0),),
                    items=(WeightedValue("mysticwater", 0.7), WeightedValue("eviolite", 0.3)),
                    moves=(
                        WeightedValue("watergun", 0.9),
                        WeightedValue("tackle", 0.8),
                        WeightedValue("quickattack", 0.4),
                        WeightedValue("tailwhip", 0.3),
                    ),
                    tera_types=(WeightedValue("water", 1.0),),
                    usage=1.0,
                ),
                PokemonDistribution(
                    species="bulbasaur",
                    level=100,
                    abilities=(WeightedValue("overgrow", 1.0),),
                    items=(WeightedValue("miracleseed", 0.7), WeightedValue("eviolite", 0.3)),
                    moves=(
                        WeightedValue("vinewhip", 0.9),
                        WeightedValue("tackle", 0.8),
                        WeightedValue("growl", 0.3),
                        WeightedValue("leechseed", 0.3),
                    ),
                    tera_types=(WeightedValue("grass", 1.0),),
                    usage=1.0,
                ),
                PokemonDistribution(
                    species="pikachu",
                    level=100,
                    abilities=(WeightedValue("static", 1.0),),
                    items=(WeightedValue("lightball", 0.8), WeightedValue("none", 0.2)),
                    moves=(
                        WeightedValue("thunderbolt", 0.9),
                        WeightedValue("quickattack", 0.8),
                        WeightedValue("tackle", 0.4),
                        WeightedValue("growl", 0.2),
                    ),
                    tera_types=(WeightedValue("electric", 1.0),),
                    usage=1.0,
                ),
                PokemonDistribution(
                    species="pidgey",
                    level=100,
                    abilities=(WeightedValue("keeneye", 1.0),),
                    items=(WeightedValue("none", 1.0),),
                    moves=(
                        WeightedValue("gust", 0.9),
                        WeightedValue("quickattack", 0.8),
                        WeightedValue("tackle", 0.7),
                        WeightedValue("sandattack", 0.3),
                    ),
                    tera_types=(WeightedValue("flying", 1.0),),
                    usage=1.0,
                ),
                PokemonDistribution(
                    species="rattata",
                    level=100,
                    abilities=(WeightedValue("runaway", 0.5), WeightedValue("guts", 0.5)),
                    items=(WeightedValue("none", 1.0),),
                    moves=(
                        WeightedValue("quickattack", 0.9),
                        WeightedValue("tackle", 0.8),
                        WeightedValue("tailwhip", 0.3),
                        WeightedValue("bite", 0.3),
                    ),
                    tera_types=(WeightedValue("normal", 1.0),),
                    usage=1.0,
                ),
            ]
        )

    def sample_team_distributions(
        self,
        rng: random.Random,
        *,
        team_size: int = 6,
    ) -> list[PokemonDistribution]:
        selected: list[PokemonDistribution] = []
        seen_species: set[str] = set()
        attempts = 0
        while len(selected) < team_size and attempts < team_size * 50:
            attempts += 1
            entry = _weighted_choice(rng, self.entries, [max(0.001, e.usage) for e in self.entries])
            species_id = normalize_id(entry.species)
            if species_id in seen_species and len(self.entries) >= team_size:
                continue
            selected.append(entry)
            seen_species.add(species_id)

        if len(selected) < team_size:
            raise ValueError("Unable to sample a full team from Showdex catalog")
        return selected


def normalize_id(value: object) -> str:
    text = "" if value is None else str(value)
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def distribution_to_json(distribution: Mapping[str, float]) -> dict[str, float]:
    return {
        key: round(float(value), 8)
        for key, value in sorted(distribution.items())
        if key and float(value) > 0
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _weighted_values(value: object) -> tuple[WeightedValue, ...]:
    values: list[WeightedValue] = []
    if isinstance(value, Mapping):
        iterable = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        iterable = value
    else:
        iterable = []

    for item in iterable:
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            name = item[0] if item else ""
            weight = item[1] if len(item) > 1 else 1.0
        else:
            name = item
            weight = 1.0

        normalized = normalize_id(name)
        if not normalized or normalized == "nothing":
            continue
        try:
            parsed_weight = float(weight)
        except (TypeError, ValueError):
            parsed_weight = 1.0
        if parsed_weight <= 0:
            continue
        values.append(WeightedValue(normalized, parsed_weight))

    return tuple(values)


def _stats_table(value: object, defaults: Mapping[str, int]) -> dict[str, int]:
    output = dict(defaults)
    for raw_key, raw_value in _mapping(value).items():
        key = SHOWDOWN_STAT_KEYS.get(str(raw_key).lower(), str(raw_key).lower())
        if key not in output:
            continue
        try:
            output[key] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return output


def _normalized_weight_map(values: Iterable[WeightedValue]) -> dict[str, float]:
    pairs = [(item.value, max(0.0, float(item.weight))) for item in values if item.value]
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return {}
    return {value: weight / total for value, weight in pairs}


def _sample_one(
    rng: random.Random,
    values: Sequence[WeightedValue],
    *,
    default: str,
) -> str:
    if not values:
        return default
    return _weighted_choice(rng, values, [max(0.0, item.weight) for item in values]).value


def _weighted_sample_without_replacement(
    rng: random.Random,
    values: Sequence[WeightedValue],
    *,
    k: int,
) -> list[str]:
    remaining = list(values)
    sampled: list[str] = []
    while remaining and len(sampled) < k:
        item = _weighted_choice(rng, remaining, [max(0.0, value.weight) for value in remaining])
        sampled.append(item.value)
        remaining = [value for value in remaining if value.value != item.value]
    return sampled


def _weighted_choice(rng: random.Random, values: Sequence[Any], weights: Sequence[float]) -> Any:
    total = sum(weights)
    if total <= 0:
        return values[rng.randrange(len(values))]
    threshold = rng.random() * total
    running = 0.0
    for value, weight in zip(values, weights, strict=True):
        running += weight
        if running >= threshold:
            return value
    return values[-1]
