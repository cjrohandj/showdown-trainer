import unittest

from collect_trajectory_mcts import (
    PublicBattleTracker,
    SpeciesDex,
    SpeciesInfo,
    VisiblePokemon,
    _observed_state_from_request,
)


def _species_dex() -> SpeciesDex:
    return SpeciesDex(
        {
            "serperior": SpeciesInfo(
                id="serperior",
                name="Serperior",
                types=("grass", "typeless"),
                base_stats={
                    "hp": 75,
                    "attack": 75,
                    "defense": 95,
                    "special-attack": 75,
                    "special-defense": 95,
                    "speed": 113,
                },
                weight_kg=63.0,
            ),
            "alomomola": SpeciesInfo(
                id="alomomola",
                name="Alomomola",
                types=("water", "typeless"),
                base_stats={
                    "hp": 165,
                    "attack": 75,
                    "defense": 80,
                    "special-attack": 40,
                    "special-defense": 45,
                    "speed": 65,
                },
                weight_kg=31.6,
            ),
        }
    )


class ObservedStateBoostTests(unittest.TestCase):
    def test_user_active_boosts_are_merged_from_public_tracker(self) -> None:
        tracker = PublicBattleTracker(
            active={
                "p1": VisiblePokemon(
                    species="serperior",
                    level=82,
                    boosts={"special-attack": 2, "speed": 2},
                ),
                "p2": VisiblePokemon(species="alomomola", level=88),
            }
        )
        request = {
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {
                        "ident": "p1a: Serperior",
                        "details": "Serperior, L82",
                        "condition": "241/241",
                        "stats": {"atk": 180, "def": 200, "spa": 212, "spd": 200, "spe": 299},
                        "moves": ["leafstorm", "substitute", "glare", "terablast"],
                        "ability": "contrary",
                        "item": "leftovers",
                        "active": True,
                        "slot": 0,
                    }
                ],
            },
            "active": [{"moves": [{"id": "leafstorm", "pp": 8, "maxpp": 8}]}],
        }

        state = _observed_state_from_request(
            request=request,
            tracker=tracker,
            actor="p1",
            opponent="p2",
            battle_id="trajectory-test",
            pokemon_format="gen9randombattle",
            generation="gen9",
            species_dex=_species_dex(),
        )

        self.assertEqual(
            state["user"]["active"]["boosts"],
            {"special-attack": 2, "speed": 2},
        )

    def test_arceus_forme_still_merges_user_active_boosts(self) -> None:
        tracker = PublicBattleTracker(
            active={
                "p1": VisiblePokemon(
                    species="arceusnormal",
                    level=80,
                    boosts={"attack": 2},
                ),
                "p2": VisiblePokemon(species="alomomola", level=88),
            }
        )
        request = {
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {
                        "ident": "p1a: Arceus",
                        "details": "Arceus, L80",
                        "condition": "381/381",
                        "stats": {"atk": 276, "def": 276, "spa": 276, "spd": 276, "spe": 276},
                        "moves": ["swordsdance", "judgment", "recover", "earthquake"],
                        "ability": "multitype",
                        "item": "silkscarf",
                        "active": True,
                        "slot": 0,
                    }
                ],
            },
            "active": [{"moves": [{"id": "swordsdance", "pp": 32, "maxpp": 32}]}],
        }

        state = _observed_state_from_request(
            request=request,
            tracker=tracker,
            actor="p1",
            opponent="p2",
            battle_id="trajectory-test",
            pokemon_format="gen9customgame",
            generation="gen9",
            species_dex=_species_dex(),
        )

        self.assertEqual(state["user"]["active"]["boosts"], {"attack": 2})


if __name__ == "__main__":
    unittest.main()
