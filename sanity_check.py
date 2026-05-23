"""End-to-end smoke checks for the MCTS -> MLP training pipeline."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

from dataset import TrainingDatasetWriter, attach_results
from encode import EncoderConfig, encode_battle_state, feature_size
from map import (
    aggregate_mcts_policy,
    decision_to_action_slot,
    legal_action_mask,
    mcts_policy_to_vector,
)
from serialize import battle_state_json, serialize_battle_state
from collect_offline_mcts import run_collection as run_offline_collection


def build_fake_battle():
    active = NS(
        name="greattusk",
        base_name="greattusk",
        nickname="Tusky",
        index=1,
        level=100,
        hp=210,
        max_hp=300,
        status=None,
        status_at_switch_in=None,
        hp_at_switch_in=300,
        types=["ground", "fighting"],
        ability="protosynthesis",
        original_ability=None,
        item="boosterenergy",
        removed_item=None,
        item_inferred=False,
        nature="jolly",
        evs=(0, 252, 0, 0, 4, 252),
        base_stats={
            "hp": 115,
            "attack": 131,
            "defense": 131,
            "special-attack": 53,
            "special-defense": 53,
            "speed": 87,
        },
        stats={
            "attack": 359,
            "defense": 299,
            "special-attack": 127,
            "special-defense": 142,
            "speed": 300,
        },
        boosts={"attack": 1},
        speed_range=NS(min=0, max=float("inf")),
        moves=[
            NS(name="earthquake", current_pp=16, max_pp=16, disabled=False, can_z=False),
            NS(name="protect", current_pp=16, max_pp=16, disabled=False, can_z=False),
            NS(name="rapidspin", current_pp=32, max_pp=64, disabled=True, can_z=False),
        ],
        moves_used_since_switch_in={"rapidspin"},
        volatile_statuses=["protosynthesisatk"],
        volatile_status_durations={},
        rest_turns=0,
        sleep_turns=0,
        substitute_hit=False,
        terastallized=False,
        tera_type="water",
        can_terastallize=True,
        can_mega_evo=False,
        can_ultra_burst=False,
        can_dynamax=False,
        is_mega=False,
        mega_name=None,
        knocked_off=False,
        unknown_forme=False,
        forme_changed=False,
        zoroark_disguised_as=None,
        can_have_choice_item=True,
        impossible_items={"choicescarf"},
        impossible_abilities=set(),
        hidden_power_possibilities=set(),
        fainted=False,
        reviving=False,
    )

    reserve = [
        NS(
            name="dragapult",
            base_name="dragapult",
            hp=100,
            max_hp=100,
            alive=True,
            fainted=False,
            moves=[],
            types=["dragon", "ghost"],
        ),
        NS(
            name="corviknight",
            base_name="corviknight",
            hp=0,
            max_hp=100,
            alive=False,
            fainted=True,
            moves=[],
            types=["flying", "steel"],
        ),
    ]

    user = NS(
        active=active,
        reserve=reserve,
        name="p1",
        account_name="bot",
        trapped=False,
        baton_passing=False,
        shed_tailing=False,
        wish=(0, 0),
        future_sight=(0, ""),
        side_conditions={"stealthrock": 1},
        last_selected_move=NS(pokemon_name="greattusk", move="earthquake", turn=3),
        last_used_move=NS(pokemon_name="greattusk", move="protect", turn=2),
        team_dict=None,
    )
    opponent = NS(
        active=NS(
            name="kingambit",
            base_name="kingambit",
            hp=150,
            max_hp=300,
            types=["dark", "steel"],
            moves=[NS(name="kowtowcleave", current_pp=16, max_pp=16)],
        ),
        reserve=[],
        name="p2",
        account_name="opponent",
        side_conditions={},
    )

    return NS(
        battle_tag="battle-gen9ou-1",
        pokemon_format="gen9ou",
        generation="gen9",
        battle_type="STANDARD_BATTLE",
        turn=4,
        started=True,
        team_preview=False,
        rqid=7,
        force_switch=False,
        wait=False,
        time_remaining=120,
        weather="sunnyday",
        weather_turns_remaining=3,
        weather_source="torkoal",
        field=None,
        field_turns_remaining=0,
        trick_room=False,
        trick_room_turns_remaining=0,
        gravity=False,
        user=user,
        opponent=opponent,
        request_json=None,
    )


def check_map_component(battle):
    assert decision_to_action_slot(battle, "earthquake") == 0
    assert decision_to_action_slot(battle, "protect") == 1
    assert decision_to_action_slot(battle, "switch dragapult") == 4
    assert decision_to_action_slot(battle, "earthquake-tera") == 9

    vector = mcts_policy_to_vector(
        battle,
        {"earthquake": 0.7, "protect": 0.2, "switch dragapult": 0.1},
    )
    expected_vector = [
        0.7,
        0.2,
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert all(
        math.isclose(actual, expected, abs_tol=1e-8)
        for actual, expected in zip(vector, expected_vector, strict=True)
    )

    result_one = NS(
        total_visits=100,
        side_one=[
            NS(move_choice="earthquake", visits=70),
            NS(move_choice="protect", visits=30),
        ],
    )
    result_two = NS(
        total_visits=50,
        side_one=[
            NS(move_choice="earthquake", visits=25),
            NS(move_choice="switch dragapult", visits=25),
        ],
    )
    policy = aggregate_mcts_policy([(result_one, 0.5, 0), (result_two, 0.5, 1)])
    assert math.isclose(sum(policy.values()), 1.0)
    assert policy["earthquake"] > policy["protect"]

    print("map: ok")


def check_serialize_component(battle):
    state = serialize_battle_state(battle, include_action_mask=True)
    assert state["user"]["active"]["name"] == "greattusk"
    assert state["user"]["active"]["hp_fraction"] == 0.7
    assert state["action_mask"] == [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    json.dumps(state, allow_nan=False)
    assert battle_state_json(battle)
    print("serialize: ok")
    return state


def check_encode_component(state):
    encoded = encode_battle_state(state)
    assert encoded.vector_size == feature_size()
    assert encoded.action_mask == state["action_mask"]
    assert sum(1 for value in encoded.features if value) > 20

    tiny_config = EncoderConfig(hash_buckets=128)
    tiny_encoded = encode_battle_state(state, config=tiny_config)
    assert tiny_encoded.vector_size == feature_size(tiny_config)
    assert tiny_encoded.vector_size < encoded.vector_size

    print("encode: ok")


def check_dataset_component(state):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "examples.jsonl"
        writer = TrainingDatasetWriter(path, run_id="sanity")
        record = writer.write_decision(
            state=state,
            chosen_action="earthquake",
            mcts_policy={
                "earthquake": 0.7,
                "protect": 0.2,
                "switch dragapult": 0.1,
            },
        )
        writer.write_decision(
            state=state,
            search_result=NS(
                chosen_action="protect",
                mcts_policy={
                    "earthquake": 0.2,
                    "protect": 0.7,
                    "switch dragapult": 0.1,
                },
                considered_policy={"protect": 0.7},
            ),
        )
        writer.write_result(battle_id=record.battle_id, winner="bot")

        rows = list(attach_results(path))
        assert len(rows) == 2
        assert rows[0]["winner"] == "bot"
        assert math.isclose(rows[0]["mcts_target"][0], 0.7, abs_tol=1e-8)
        assert math.isclose(rows[0]["mcts_target"][4], 0.1, abs_tol=1e-8)
        assert rows[0]["slot_to_decision"]["0"] == "earthquake"
        assert rows[1]["chosen_action_slot"] == 1
        assert rows[1]["considered_policy"] == {"protect": 0.7}
        assert path.read_text().count("\n") == 4

    print("dataset: ok")


def check_offline_mcts_component():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "offline.jsonl"
        summary = run_offline_collection(
            {
                "output_path": str(path),
                "run_id": "offline-sanity",
                "use_fixture": True,
                "backend": "toy",
                "allow_toy_backend": True,
                "positions": 3,
                "hypotheses": 2,
                "search_time_ms": 1,
                "pokemon_format": "gen9randombattle",
                "generation": "gen9",
                "include_state": True,
            }
        )
        assert summary["examples_written"] == 3
        rows = list(attach_results(path))
        assert len(rows) == 3
        assert rows[0]["metadata"]["source"] == "showdex-poke-engine-mcts"
        assert rows[0]["state"]["opponent"]["active"]["move_distribution"]
        assert math.isclose(sum(rows[0]["mcts_target"]), 1.0, rel_tol=1e-8)

    print("offline-mcts: ok")


def check_train_component(state):
    import train

    if train.TORCH_IMPORT_ERROR is not None:
        try:
            train.PolicyMLP(8)
        except RuntimeError:
            print("train: torch unavailable, import/failure path ok")
            return
        raise AssertionError("PolicyMLP should require torch when torch is unavailable")

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "examples.jsonl"
        checkpoint_path = Path(temp_dir) / "policy.pt"
        writer = TrainingDatasetWriter(path, run_id="sanity-train")

        policies = [
            {"earthquake": 0.8, "protect": 0.1, "switch dragapult": 0.1},
            {"earthquake": 0.2, "protect": 0.7, "switch dragapult": 0.1},
            {"earthquake": 0.4, "protect": 0.2, "switch dragapult": 0.4},
            {"earthquake": 0.6, "protect": 0.2, "switch dragapult": 0.2},
            {"earthquake": 0.1, "protect": 0.8, "switch dragapult": 0.1},
            {"earthquake": 0.3, "protect": 0.3, "switch dragapult": 0.4},
        ]
        for index, policy in enumerate(policies):
            row_state = dict(state)
            row_state["turn"] = index + 1
            chosen = max(policy, key=policy.get)
            record = writer.write_decision(
                state=row_state,
                chosen_action=chosen,
                mcts_policy=policy,
            )
            writer.write_result(battle_id=record.battle_id, winner="bot")

        config = train.TrainConfig(
            data_path=str(path),
            output_path=str(checkpoint_path),
            epochs=1,
            batch_size=2,
            validation_split=0.33,
            hidden_sizes=(32,),
            dropout=0.0,
            hash_buckets=128,
            device="cpu",
        )
        summary = train.train(config)
        assert checkpoint_path.exists()
        assert summary["examples"] == len(policies)

        resume_config = train.TrainConfig(
            data_path=str(path),
            output_path=str(checkpoint_path),
            resume_checkpoint_path=str(checkpoint_path),
            epochs=1,
            batch_size=2,
            validation_split=0.33,
            hidden_sizes=(32,),
            dropout=0.0,
            hash_buckets=128,
            device="cpu",
        )
        resume_summary = train.train(resume_config)
        assert resume_summary["examples"] == len(policies)

        model, checkpoint = train.load_policy_checkpoint(checkpoint_path)
        encoder_config = EncoderConfig(**checkpoint["encoder_config"])
        probs = train.predict_policy(model, state, encoder_config=encoder_config)
        assert len(probs) == 13
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
        for slot, allowed in enumerate(state["action_mask"]):
            if not allowed:
                assert probs[slot] < 1e-6

    print("train: ok")


def main():
    battle = build_fake_battle()
    check_map_component(battle)
    state = check_serialize_component(battle)
    check_encode_component(state)
    check_dataset_component(state)
    check_offline_mcts_component()
    check_train_component(state)
    print("full MCTS -> MLP pipeline: ok")


if __name__ == "__main__":
    main()
