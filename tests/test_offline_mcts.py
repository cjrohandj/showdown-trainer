import unittest

from offline_mcts import make_backend
from map import aggregate_mcts_policy


class PokeEngineBoostEncodingTests(unittest.TestCase):
    def test_active_boosts_are_forwarded_to_side(self) -> None:
        try:
            backend = make_backend("poke-engine")
        except RuntimeError as exc:
            self.skipTest(str(exc))

        state = {
            "user": {
                "active": {
                    "name": "arceus",
                    "level": 80,
                    "types": ["normal", "typeless"],
                    "hp": 381,
                    "max_hp": 381,
                    "ability": "multitype",
                    "item": "silkscarf",
                    "nature": "serious",
                    "evs": [85, 85, 85, 85, 85, 85],
                    "stats": {
                        "attack": 276,
                        "defense": 276,
                        "special-attack": 276,
                        "special-defense": 276,
                        "speed": 276,
                    },
                    "status": "none",
                    "moves": [
                        {"name": "swordsdance", "current_pp": 32, "max_pp": 32},
                        {"name": "judgment", "current_pp": 16, "max_pp": 16},
                    ],
                    "boosts": {"attack": 2, "speed": 1},
                },
                "reserve": [],
                "side_conditions": {},
                "trapped": False,
                "baton_passing": False,
                "shed_tailing": False,
                "wish": [0, 0],
                "future_sight": [0, ""],
                "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
            },
            "opponent": {
                "active": {
                    "name": "alomomola",
                    "level": 88,
                    "types": ["water", "typeless"],
                    "hp": 534,
                    "max_hp": 534,
                    "ability": "regenerator",
                    "item": "leftovers",
                    "nature": "serious",
                    "evs": [85, 85, 85, 85, 85, 85],
                    "stats": {
                        "attack": 166,
                        "defense": 181,
                        "special-attack": 103,
                        "special-defense": 118,
                        "speed": 133,
                    },
                    "status": "none",
                    "moves": [{"name": "scald", "current_pp": 16, "max_pp": 16}],
                    "boosts": {},
                },
                "reserve": [],
                "side_conditions": {},
                "trapped": False,
                "baton_passing": False,
                "shed_tailing": False,
                "wish": [0, 0],
                "future_sight": [0, ""],
                "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
            },
            "weather": "none",
            "weather_turns_remaining": 0,
            "field": "none",
            "field_turns_remaining": 0,
            "trick_room": False,
            "trick_room_turns_remaining": 0,
            "team_preview": False,
        }

        engine_state = backend.to_engine_state(state)
        self.assertEqual(engine_state.side_one.attack_boost, 2)
        self.assertEqual(engine_state.side_one.speed_boost, 1)
        self.assertEqual(engine_state.side_one.special_attack_boost, 0)


class PokeEngineHazardChoiceTests(unittest.TestCase):
    def _battle_state(self) -> dict:
        return {
            "user": {
                "active": {
                    "name": "chansey",
                    "level": 84,
                    "types": ["normal", "typeless"],
                    "hp": 620,
                    "max_hp": 620,
                    "ability": "naturalcure",
                    "item": "eviolite",
                    "nature": "bold",
                    "evs": {
                        "hp": 85,
                        "attack": 0,
                        "defense": 85,
                        "special-attack": 0,
                        "special-defense": 85,
                        "speed": 0,
                    },
                    "stats": {
                        "attack": 76,
                        "defense": 119,
                        "special-attack": 96,
                        "special-defense": 246,
                        "speed": 126,
                    },
                    "status": "none",
                    "moves": [
                        {"name": "stealthrock", "current_pp": 32, "max_pp": 32},
                        {"name": "seismictoss", "current_pp": 32, "max_pp": 32},
                        {"name": "softboiled", "current_pp": 16, "max_pp": 16},
                        {"name": "thunderwave", "current_pp": 32, "max_pp": 32},
                    ],
                    "boosts": {},
                    "can_terastallize": False,
                },
                "reserve": [],
                "side_conditions": {},
                "trapped": False,
                "baton_passing": False,
                "shed_tailing": False,
                "wish": [0, 0],
                "future_sight": [0, ""],
                "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
            },
            "opponent": {
                "active": {
                    "name": "charizard",
                    "level": 84,
                    "types": ["fire", "flying"],
                    "hp": 266,
                    "max_hp": 266,
                    "ability": "blaze",
                    "item": "heavydutyboots",
                    "nature": "timid",
                    "evs": {
                        "hp": 0,
                        "attack": 0,
                        "defense": 0,
                        "special-attack": 85,
                        "special-defense": 0,
                        "speed": 85,
                    },
                    "stats": {
                        "attack": 156,
                        "defense": 192,
                        "special-attack": 254,
                        "special-defense": 206,
                        "speed": 266,
                    },
                    "status": "none",
                    "moves": [
                        {"name": "flamethrower", "current_pp": 24, "max_pp": 24},
                    ],
                    "boosts": {},
                    "can_terastallize": False,
                },
                "reserve": [],
                "side_conditions": {},
                "trapped": False,
                "baton_passing": False,
                "shed_tailing": False,
                "wish": [0, 0],
                "future_sight": [0, ""],
                "last_used_move": {"pokemon_name": "", "move": "", "turn": 0},
            },
            "weather": "none",
            "weather_turns_remaining": 0,
            "field": "none",
            "field_turns_remaining": 0,
            "trick_room": False,
            "trick_room_turns_remaining": 0,
            "team_preview": False,
        }

    def _top_decision(self, state: dict) -> str:
        backend = make_backend("poke-engine")
        result = backend.search(state, duration_ms=100, threads=1)
        policy = aggregate_mcts_policy([result])
        self.assertTrue(policy)
        return next(iter(policy))

    def test_chansey_prefers_seismic_toss_over_stealth_rock_into_charizard(self) -> None:
        try:
            initial_top = self._top_decision(self._battle_state())
        except RuntimeError as exc:
            self.skipTest(str(exc))

        follow_up_state = self._battle_state()
        follow_up_state["opponent"]["side_conditions"]["stealthrock"] = 1
        follow_up_top = self._top_decision(follow_up_state)

        self.assertEqual(initial_top, "seismictoss")
        self.assertEqual(follow_up_top, "seismictoss")


if __name__ == "__main__":
    unittest.main()
