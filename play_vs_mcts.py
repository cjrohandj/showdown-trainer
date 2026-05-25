#!/usr/bin/env python3
"""Play a local Showdown battle against an MCTS-controlled bot or policy checkpoint.

This starts an in-process Pokemon Showdown battle, shows the human player the
public state before each choice, and sends the opposing side's moves through the
same MCTS backend used by the collector.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import threading
import webbrowser

import collect_trajectory_mcts as ctm
import train as policy_train
from encode import EncoderConfig
from map import action_slot_to_decision, legal_action_mask
from trainer_config import collection_config_from_yaml


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--pokemon-format")
    parser.add_argument("--generation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("poke-engine", "toy"))
    parser.add_argument("--search-time-ms", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--hypotheses", type=int)
    parser.add_argument("--bridge-script", default="scripts/showdown_bridge.js")
    parser.add_argument("--species-data-path", default=ctm.DEFAULT_SPECIES_DATA_PATH)
    parser.add_argument("--randoms-preset-path")
    parser.add_argument("--randoms-stats-path")
    parser.add_argument("--allow-toy-backend", action="store_true")
    parser.add_argument("--use-fixture", action="store_true")
    parser.add_argument("--human-side", choices=("p1", "p2"), default="p1")
    parser.add_argument("--human-name", default="human")
    parser.add_argument("--bot-name", default="mcts")
    parser.add_argument(
        "--bot-mode",
        choices=("mcts", "policy"),
        default="mcts",
        help="Use the existing MCTS bot or a trained policy checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Path to a .pt checkpoint used when --bot-mode policy is selected.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve a local browser UI for the battle instead of using the terminal.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local browser UI automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = collection_config_from_yaml(args.config)
    _apply_overrides(settings, args)
    if args.web:
        run_browser_server(settings, args)
    else:
        run_match(settings, args)


def run_match(settings: dict[str, Any], args: argparse.Namespace) -> None:
    showdex = _mapping(settings.get("showdex"))
    mcts = _mapping(settings.get("mcts"))

    seed = _resolve_run_seed(settings, args, mcts)
    settings["seed"] = seed
    rng = random.Random(seed)
    logger.info("Using run seed %s", seed)
    pokemon_format = str(
        settings.get("pokemon_format")
        or showdex.get("pokemon_format")
        or "gen9randombattle"
    )
    generation = str(settings.get("generation") or _generation_from_format(pokemon_format))
    search_time_ms = _int(
        settings.get("search_time_ms", mcts.get("search_time_ms")),
        75,
    )
    threads = _int(settings.get("threads", mcts.get("threads")), 1)
    hypotheses = _int(
        settings.get("hypotheses", mcts.get("hypotheses_per_position")),
        4,
    )
    backend_name = str(settings.get("backend") or mcts.get("backend") or "poke-engine")
    allow_toy_backend = _bool(
        settings.get("allow_toy_backend", mcts.get("allow_toy_backend")),
        False,
    )

    catalog = ctm._load_catalog(settings)
    catalog_by_species = ctm._catalog_by_species(catalog)
    species_data_path = str(settings.get("species_data_path") or args.species_data_path)
    species_dex = ctm.SpeciesDex.from_path(species_data_path)
    policy_bot = _load_policy_bot(args)
    bot_display_name = policy_bot["bot_name"] if policy_bot else args.bot_name
    backend = None
    if policy_bot is None:
        backend = ctm.make_backend(backend_name, allow_toy_backend=allow_toy_backend)

    human_side = args.human_side
    bot_side = ctm._opponent(human_side)

    p1_team = ctm._sample_team(catalog, rng)
    p2_team = ctm._sample_team(catalog, rng)
    showdown_rng = random.SystemRandom()

    with ctm.ShowdownBridge(Path(args.bridge_script)) as bridge:
        bridge.send(
            {
                "type": "start",
                "formatid": pokemon_format,
                "seed": [showdown_rng.randrange(1, 0x10000) for _ in range(4)],
                "p1": {
                    "name": args.human_name if human_side == "p1" else bot_display_name,
                    "team": ctm._showdown_team(p1_team),
                },
                "p2": {
                    "name": args.human_name if human_side == "p2" else bot_display_name,
                    "team": ctm._showdown_team(p2_team),
                },
            }
        )

        tracker = ctm.PublicBattleTracker()
        turn_log = TurnTranscript(human_side=human_side)
        requests: dict[str, dict[str, Any]] = {}
        pending_human_choice: str | None = None
        pending_bot_choice: str | None = None
        battle_over = False

        while not battle_over:
            event = bridge.read_event()
            event_type = event.get("type")
            if event_type == "update":
                messages = event.get("messages") or []
                tracker.apply_update(messages)
                turn_log.consume(list(messages))
            elif event_type == "request":
                player = str(event.get("player"))
                request = _mapping(event.get("request"))
                if request:
                    turn_log.flush_pending()
                    requests[player] = request
            elif event_type == "end":
                turn_log.finalize()
                data = _mapping(event.get("data"))
                winner = ctm._winner_to_player(data.get("winner")) or tracker.winner
                print()
                print(f"Battle ended. Winner: {winner or 'unknown'}")
                return

            if human_side not in requests or bot_side not in requests:
                continue

            human_request = requests[human_side]
            bot_request = requests[bot_side]

            if _request_type(human_request) == "teampreview" or _request_type(bot_request) == "teampreview":
                _handle_team_preview(
                    bridge=bridge,
                    human_side=human_side,
                    bot_side=bot_side,
                    human_request=human_request,
                    bot_request=bot_request,
                    human_name=args.human_name,
                    bot_name=bot_display_name,
                )
                requests.clear()
                continue

            human_state = ctm._observed_state_from_request(
                request=human_request,
                tracker=tracker,
                actor=human_side,
                opponent=bot_side,
                battle_id=str(human_request.get("battle_tag") or "battle"),
                pokemon_format=pokemon_format,
                generation=generation,
                species_dex=species_dex,
            )
            human_state["action_mask"] = ctm._action_mask_from_request(human_request, human_state)

            bot_state = ctm._observed_state_from_request(
                request=bot_request,
                tracker=tracker,
                actor=bot_side,
                opponent=human_side,
                battle_id=str(bot_request.get("battle_tag") or "battle"),
                pokemon_format=pokemon_format,
                generation=generation,
                species_dex=species_dex,
            )
            bot_state["action_mask"] = ctm._action_mask_from_request(bot_request, bot_state)

            # Wait for the initial switch updates so the public opponent active
            # Pokémon is visible before the first turn decision.
            if not ctm._public_opponents_ready(tracker, (human_side, bot_side)):
                continue

            if pending_bot_choice is None:
                if policy_bot is None:
                    if backend is None:
                        raise RuntimeError("MCTS backend was not initialized")
                    pending_bot_choice = _choose_mcts_bot_action(
                        backend=backend,
                        rng=rng,
                        catalog=catalog,
                        catalog_by_species=catalog_by_species,
                        species_dex=species_dex,
                        observed_state=bot_state,
                        actor_request=bot_request,
                        actor=bot_side,
                        duration_ms=search_time_ms,
                        threads=threads,
                        hypotheses=hypotheses,
                        human_side=human_side,
                        bot_side=bot_side,
                    )
                else:
                    pending_bot_choice = _choose_policy_bot_action(
                        policy_bot=policy_bot,
                        observed_state=bot_state,
                        actor_request=bot_request,
                    )

            if pending_human_choice is None:
                pending_human_choice = _prompt_human_choice(
                    human_state=human_state,
                    human_request=human_request,
                    human_side=human_side,
                    human_name=args.human_name,
                    bot_name=bot_display_name,
                )

            bridge.send({"type": "choice", "player": bot_side, "choice": pending_bot_choice})
            bridge.send({"type": "choice", "player": human_side, "choice": pending_human_choice})
            requests.clear()
            pending_human_choice = None
            pending_bot_choice = None

    raise RuntimeError("Battle bridge closed unexpectedly")


def run_browser_server(settings: dict[str, Any], args: argparse.Namespace) -> None:
    session = BrowserBattleSession(settings=settings, args=args)
    server = _BrowserHTTPServer((args.host, args.port), _BrowserRequestHandler, session)
    url = f"http://{args.host}:{args.port}/"

    print(f"Browser battle server listening on {url}")
    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        server.shutdown()
        server.server_close()


class BrowserBattleSession:
    def __init__(self, *, settings: dict[str, Any], args: argparse.Namespace):
        self.settings = settings
        self.args = args
        self.lock = threading.Lock()
        self.choice_cond = threading.Condition(self.lock)
        self.pending_choice: str | None = None
        self.choice_error: str | None = None
        self.stopped = False
        self.state: dict[str, Any] = {
            "phase": "starting",
            "request_type": None,
            "prompt": "Starting battle...",
            "public_state_text": "",
            "recent_messages": [],
            "turn_summaries": [],
            "legal_choices": [],
            "battle_over": False,
            "winner": None,
            "human_name": args.human_name,
            "bot_name": args.bot_name,
            "run_seed": None,
            "last_submitted_choice": None,
            "last_error": None,
        }
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.choice_cond:
            self.stopped = True
            self.choice_cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.state)

    def submit_choice(self, choice: str) -> None:
        choice = str(choice or "").strip()
        with self.choice_cond:
            self.pending_choice = choice
            self.state["last_submitted_choice"] = choice
            self.state["choice_error"] = None
            if not self.state.get("battle_over"):
                self.state["prompt"] = f"Submitted {choice}. Waiting for the turn to resolve..."
            self.choice_cond.notify_all()

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _append_turn_summary(self, lines: list[str]) -> None:
        summary = "\n".join(lines)
        with self.lock:
            summaries = list(self.state.get("turn_summaries") or [])
            summaries.append(summary)
            self.state["turn_summaries"] = summaries[-24:]

    def _run(self) -> None:
        showdex = _mapping(self.settings.get("showdex"))
        mcts = _mapping(self.settings.get("mcts"))

        seed = _resolve_run_seed(self.settings, self.args, mcts)
        self.settings["seed"] = seed
        rng = random.Random(seed)
        logger.info("Using run seed %s", seed)
        pokemon_format = str(
            self.settings.get("pokemon_format")
            or showdex.get("pokemon_format")
            or "gen9randombattle"
        )
        generation = str(self.settings.get("generation") or _generation_from_format(pokemon_format))
        search_time_ms = _int(
            self.settings.get("search_time_ms", mcts.get("search_time_ms")),
            75,
        )
        threads = _int(self.settings.get("threads", mcts.get("threads")), 1)
        hypotheses = _int(
            self.settings.get("hypotheses", mcts.get("hypotheses_per_position")),
            4,
        )
        backend_name = str(self.settings.get("backend") or mcts.get("backend") or "poke-engine")
        allow_toy_backend = _bool(
            self.settings.get("allow_toy_backend", mcts.get("allow_toy_backend")),
            False,
        )

        catalog = ctm._load_catalog(self.settings)
        catalog_by_species = ctm._catalog_by_species(catalog)
        species_data_path = str(self.settings.get("species_data_path") or self.args.species_data_path)
        species_dex = ctm.SpeciesDex.from_path(species_data_path)
        policy_bot = _load_policy_bot(self.args)
        bot_display_name = policy_bot["bot_name"] if policy_bot else self.args.bot_name
        backend = None
        if policy_bot is None:
            backend = ctm.make_backend(backend_name, allow_toy_backend=allow_toy_backend)

        human_side = self.args.human_side
        bot_side = ctm._opponent(human_side)

        p1_team = ctm._sample_team(catalog, rng)
        p2_team = ctm._sample_team(catalog, rng)
        showdown_rng = random.SystemRandom()

        with ctm.ShowdownBridge(Path(self.args.bridge_script)) as bridge:
            bridge.send(
                {
                    "type": "start",
                    "formatid": pokemon_format,
                    "seed": [showdown_rng.randrange(1, 0x10000) for _ in range(4)],
                    "p1": {
                        "name": self.args.human_name if human_side == "p1" else bot_display_name,
                        "team": ctm._showdown_team(p1_team),
                    },
                    "p2": {
                        "name": self.args.human_name if human_side == "p2" else bot_display_name,
                        "team": ctm._showdown_team(p2_team),
                    },
                }
            )

            tracker = ctm.PublicBattleTracker()
            turn_log = TurnTranscript(
                human_side=human_side,
                human_label=self.args.human_name,
                opponent_label=bot_display_name,
                on_summary=self._append_turn_summary,
            )
            requests: dict[str, dict[str, Any]] = {}
            pending_bot_choice: str | None = None

            self._set_state(
                phase="waiting",
                prompt="Waiting for battle request...",
                human_name=self.args.human_name,
                bot_name=bot_display_name,
                run_seed=seed,
                battle_over=False,
                winner=None,
            )

            while True:
                with self.choice_cond:
                    if self.stopped:
                        return

                event = bridge.read_event()
                event_type = event.get("type")
                if event_type == "update":
                    messages = event.get("messages") or []
                    tracker.apply_update(messages)
                    turn_log.consume(list(messages))
                    self._set_state(recent_messages=list(messages)[-40:])
                elif event_type == "request":
                    player = str(event.get("player"))
                    request = _mapping(event.get("request"))
                    if request:
                        turn_log.flush_pending()
                        requests[player] = request
                elif event_type == "end":
                    turn_log.finalize()
                    data = _mapping(event.get("data"))
                    winner = ctm._winner_to_player(data.get("winner")) or tracker.winner
                    self._set_state(
                        phase="battle_over",
                        prompt="Battle ended.",
                        battle_over=True,
                        winner=winner or "unknown",
                        public_state_text="Battle complete.",
                        legal_choices=[],
                        request_type=None,
                    )
                    return

                ready_players = ctm._ready_players(requests)
                if not ready_players:
                    continue

                human_request = requests[human_side]
                bot_request = requests[bot_side]

                if _request_type(human_request) == "teampreview" or _request_type(bot_request) == "teampreview":
                    public_state_text = self._public_state_text(
                        tracker,
                        human_side=human_side,
                        bot_display_name=bot_display_name,
                        pokemon_format=pokemon_format,
                        generation=generation,
                        human_request=human_request,
                        bot_request=bot_request,
                        species_dex=species_dex,
                    )
                    self._set_state(
                        phase="team_preview",
                        request_type="teampreview",
                        public_state_text=public_state_text,
                        prompt="Enter a team order such as 123456 and click Submit.",
                        legal_choices=[],
                        choice_error=None,
                    )
                    human_choice = self._await_choice("teampreview")
                    bridge.send({"type": "choice", "player": human_side, "choice": human_choice})
                    bridge.send({"type": "choice", "player": bot_side, "choice": "team 123456"})
                    requests.clear()
                    continue

                if not ctm._public_opponents_ready(tracker, (human_side, bot_side)):
                    continue

                if ready_players == [bot_side]:
                    bot_state = ctm._observed_state_from_request(
                        request=bot_request,
                        tracker=tracker,
                        actor=bot_side,
                        opponent=human_side,
                        battle_id=str(bot_request.get("battle_tag") or "battle"),
                        pokemon_format=pokemon_format,
                        generation=generation,
                        species_dex=species_dex,
                    )
                    bot_state["action_mask"] = ctm._action_mask_from_request(bot_request, bot_state)
                    if pending_bot_choice is None:
                        if policy_bot is None:
                            if backend is None:
                                raise RuntimeError("MCTS backend was not initialized")
                            pending_bot_choice = _choose_mcts_bot_action(
                                backend=backend,
                                rng=rng,
                                catalog=catalog,
                                catalog_by_species=catalog_by_species,
                                species_dex=species_dex,
                                observed_state=bot_state,
                                actor_request=bot_request,
                                actor=bot_side,
                                duration_ms=search_time_ms,
                                threads=threads,
                                hypotheses=hypotheses,
                                human_side=human_side,
                                bot_side=bot_side,
                            )
                        else:
                            pending_bot_choice = _choose_policy_bot_action(
                                policy_bot=policy_bot,
                                observed_state=bot_state,
                                actor_request=bot_request,
                            )
                    bridge.send({"type": "choice", "player": bot_side, "choice": pending_bot_choice})
                    requests.pop(bot_side, None)
                    pending_bot_choice = None
                    self._set_state(prompt="Waiting for the next request...", phase="waiting")
                    continue

                human_state = ctm._observed_state_from_request(
                    request=human_request,
                    tracker=tracker,
                    actor=human_side,
                    opponent=bot_side,
                    battle_id=str(human_request.get("battle_tag") or "battle"),
                    pokemon_format=pokemon_format,
                    generation=generation,
                    species_dex=species_dex,
                )
                human_state["action_mask"] = ctm._action_mask_from_request(human_request, human_state)

                bot_state = ctm._observed_state_from_request(
                    request=bot_request,
                    tracker=tracker,
                    actor=bot_side,
                    opponent=human_side,
                    battle_id=str(bot_request.get("battle_tag") or "battle"),
                    pokemon_format=pokemon_format,
                    generation=generation,
                    species_dex=species_dex,
                )
                bot_state["action_mask"] = ctm._action_mask_from_request(bot_request, bot_state)

                if pending_bot_choice is None:
                    if policy_bot is None:
                        if backend is None:
                            raise RuntimeError("MCTS backend was not initialized")
                        pending_bot_choice = _choose_mcts_bot_action(
                            backend=backend,
                            rng=rng,
                            catalog=catalog,
                            catalog_by_species=catalog_by_species,
                            species_dex=species_dex,
                            observed_state=bot_state,
                            actor_request=bot_request,
                            actor=bot_side,
                            duration_ms=search_time_ms,
                            threads=threads,
                            hypotheses=hypotheses,
                            human_side=human_side,
                            bot_side=bot_side,
                        )
                    else:
                        pending_bot_choice = _choose_policy_bot_action(
                            policy_bot=policy_bot,
                            observed_state=bot_state,
                            actor_request=bot_request,
                        )

                public_state_text = render_public_state(
                    human_state,
                    human_side=human_side,
                    human_name=self.args.human_name,
                    bot_name=bot_display_name,
                )
                human_choices = _build_human_choices(
                    human_state=human_state,
                    human_request=human_request,
                )
                self._set_state(
                    phase="choose_action",
                    request_type=_request_type(human_request),
                    public_state_text=public_state_text,
                    prompt="Choose an action from the buttons below.",
                    legal_choices=human_choices,
                    choice_error=None,
                    battle_over=False,
                    winner=None,
                )
                human_choice = self._await_choice(_request_type(human_request))
                bridge.send({"type": "choice", "player": bot_side, "choice": pending_bot_choice})
                bridge.send({"type": "choice", "player": human_side, "choice": human_choice})
                requests.clear()
                pending_bot_choice = None
                self._set_state(legal_choices=[], prompt="Waiting for the next request...", phase="waiting")

    def _await_choice(self, request_type: str) -> str:
        with self.choice_cond:
            self.pending_choice = None
            while True:
                if self.stopped:
                    raise RuntimeError("Browser battle session was stopped")
                if self.pending_choice is not None:
                    raw = self.pending_choice
                    self.pending_choice = None
                    try:
                        choice = self._normalize_choice(request_type, raw)
                        self.state["choice_error"] = None
                        return choice
                    except Exception as exc:
                        self.state["choice_error"] = str(exc)
                        self.pending_choice = None
                        self.choice_cond.notify_all()
                self.choice_cond.wait(timeout=0.25)

    def _normalize_choice(self, request_type: str, raw: str) -> str:
        raw = str(raw or "").strip()
        if request_type == "teampreview":
            if raw.lower().startswith("team "):
                raw = raw[5:].strip()
            if not raw:
                raw = "123456"
            if not _is_valid_team_order(raw):
                raise ValueError("Team order must be a permutation of 123456")
            return f"team {raw}"

        choices = list(self.state.get("legal_choices") or [])
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(choices):
                return str(choices[picked - 1]["command"])
        allowed = {str(choice.get("command")) for choice in choices}
        if raw in allowed:
            return raw
        raise ValueError("Choice is not one of the listed legal actions")

    def _public_state_text(
        self,
        tracker: ctm.PublicBattleTracker,
        *,
        human_side: str,
        bot_display_name: str,
        pokemon_format: str,
        generation: str,
        human_request: Mapping[str, Any],
        bot_request: Mapping[str, Any],
        species_dex: ctm.SpeciesDex,
    ) -> str:
        human_state = ctm._observed_state_from_request(
            request=human_request,
            tracker=tracker,
            actor=human_side,
            opponent=ctm._opponent(human_side),
            battle_id=str(human_request.get("battle_tag") or "battle"),
            pokemon_format=pokemon_format,
            generation=generation,
            species_dex=species_dex,
        )
        bot_state = ctm._observed_state_from_request(
            request=bot_request,
            tracker=tracker,
            actor=ctm._opponent(human_side),
            opponent=human_side,
            battle_id=str(bot_request.get("battle_tag") or "battle"),
            pokemon_format=pokemon_format,
            generation=generation,
            species_dex=species_dex,
        )
        return render_public_state(
            human_state,
            human_side=human_side,
            human_name=self.args.human_name,
            bot_name=bot_display_name,
        )


class _BrowserHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass, session: BrowserBattleSession):
        self.session = session
        super().__init__(server_address, RequestHandlerClass)


class _BrowserRequestHandler(BaseHTTPRequestHandler):
    server: _BrowserHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(BROWSER_HTML)
            return
        if parsed.path == "/api/state":
            self._send_json(self.server.session.snapshot())
            return
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/action":
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            choice = str(payload.get("choice") or "")
            self.server.session.submit_choice(choice)
            self._send_json({"ok": True})
            return
        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


BROWSER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local Showdown Battle</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101318;
      --panel: #171c24;
      --panel-2: #1d2330;
      --text: #eef3ff;
      --muted: #aab4c5;
      --accent: #8cc8ff;
      --accent-2: #7ce7b7;
      --danger: #ff7e7e;
      --border: rgba(255,255,255,0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(140,200,255,.12), transparent 36%),
        radial-gradient(circle at right 20%, rgba(124,231,183,.10), transparent 34%),
        var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 16px;
      grid-template-columns: 1.25fr 0.9fr;
    }
    .hero, .panel {
      background: rgba(23, 28, 36, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0,0,0,.25);
      backdrop-filter: blur(10px);
    }
    .hero { grid-column: 1 / -1; padding: 20px 22px; }
    .panel { padding: 18px; min-height: 200px; }
    h1, h2, h3 { margin: 0 0 12px; }
    h1 { font-size: 24px; }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #dbe6ff;
    }
    .stack { display: grid; gap: 12px; }
    .controls {
      display: grid;
      gap: 10px;
    }
    .choice-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    button, input {
      font: inherit;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px 14px;
    }
    button {
      cursor: pointer;
      text-align: left;
      transition: transform .12s ease, border-color .12s ease, background .12s ease;
    }
    button:hover { transform: translateY(-1px); border-color: rgba(140,200,255,.35); }
    button.primary { background: linear-gradient(135deg, rgba(140,200,255,.25), rgba(124,231,183,.18)); }
    button[disabled] { opacity: .4; cursor: not-allowed; transform: none; }
    .status { color: var(--muted); font-size: 14px; }
    .error { color: var(--danger); }
    .summary {
      display: grid;
      gap: 10px;
      max-height: 360px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .summary-item {
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(255,255,255,.04);
    }
    .section-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .small { font-size: 12px; color: var(--muted); }
    @media (max-width: 920px) {
      .wrap { grid-template-columns: 1fr; }
      .hero { grid-column: auto; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Local Showdown Battle</h1>
      <div class="meta">
        <span class="badge">Phase: <span id="phase">starting</span></span>
        <span class="badge">Winner: <span id="winner">-</span></span>
        <span class="badge">Request: <span id="request_type">-</span></span>
        <span class="badge">Bot: <span id="bot_name">bot</span></span>
        <span class="badge">Seed: <span id="run_seed">-</span></span>
      </div>
      <p class="status" id="prompt" style="margin-top: 12px;">Loading battle...</p>
      <p class="status error" id="error" style="margin: 4px 0 0;"></p>
    </section>

    <section class="panel stack">
      <div class="section-title">
        <h2>Battle View</h2>
        <span class="small">auto-refreshing</span>
      </div>
      <pre id="public_state">Loading...</pre>
    </section>

    <section class="panel stack">
      <div class="section-title">
        <h2>Actions</h2>
        <span class="small">click or type</span>
      </div>
      <div class="controls" id="controls">
        <div class="status">Waiting for the battle to request a choice.</div>
      </div>
    </section>

    <section class="panel stack">
      <div class="section-title">
        <h2>Turn Recap</h2>
        <span class="small">end-of-turn summary</span>
      </div>
      <div class="summary" id="turn_summaries"></div>
    </section>

    <section class="panel stack">
      <div class="section-title">
        <h2>Recent Messages</h2>
        <span class="small">raw simulator output</span>
      </div>
      <pre id="recent_messages"></pre>
    </section>
  </div>

  <script>
    const els = {
      phase: document.getElementById('phase'),
      winner: document.getElementById('winner'),
      requestType: document.getElementById('request_type'),
      botName: document.getElementById('bot_name'),
      runSeed: document.getElementById('run_seed'),
      prompt: document.getElementById('prompt'),
      error: document.getElementById('error'),
      publicState: document.getElementById('public_state'),
      controls: document.getElementById('controls'),
      turnSummaries: document.getElementById('turn_summaries'),
      recentMessages: document.getElementById('recent_messages'),
    };

    async function postChoice(choice) {
      await fetch('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({choice}),
      });
    }

    function renderControls(state) {
      els.controls.innerHTML = '';
      const choices = state.legal_choices || [];

      if (state.battle_over) {
        const done = document.createElement('div');
        done.className = 'status';
        done.textContent = 'Battle over. Reload the page to inspect the final state.';
        els.controls.appendChild(done);
        return;
      }

      if (state.request_type === 'teampreview') {
        const row = document.createElement('div');
        row.className = 'controls';
        const input = document.createElement('input');
        input.id = 'team-order';
        input.placeholder = '123456';
        input.value = '123456';
        const button = document.createElement('button');
        button.className = 'primary';
        button.textContent = 'Submit team order';
        button.onclick = () => postChoice(input.value.trim());
        input.addEventListener('keydown', (event) => {
          if (event.key === 'Enter') button.click();
        });
        row.appendChild(input);
        row.appendChild(button);
        els.controls.appendChild(row);
        return;
      }

      if (!choices.length) {
        const idle = document.createElement('div');
        idle.className = 'status';
        idle.textContent = 'Waiting for the next action request...';
        els.controls.appendChild(idle);
        return;
      }

      const grid = document.createElement('div');
      grid.className = 'choice-grid';
      choices.forEach((choice, index) => {
        const button = document.createElement('button');
        button.className = index === 0 ? 'primary' : '';
        button.textContent = choice.label + '\\n' + choice.command;
        button.onclick = () => postChoice(choice.command);
        grid.appendChild(button);
      });
      els.controls.appendChild(grid);
    }

    function renderTurnSummaries(items) {
      els.turnSummaries.innerHTML = '';
      if (!items || !items.length) {
        const empty = document.createElement('div');
        empty.className = 'status';
        empty.textContent = 'No turns have been summarized yet.';
        els.turnSummaries.appendChild(empty);
        return;
      }
      items.forEach((text) => {
        const item = document.createElement('div');
        item.className = 'summary-item';
        const pre = document.createElement('pre');
        pre.textContent = text;
        item.appendChild(pre);
        els.turnSummaries.appendChild(item);
      });
      els.turnSummaries.scrollTop = els.turnSummaries.scrollHeight;
    }

    async function refresh() {
      const response = await fetch('/api/state');
      const state = await response.json();
      els.phase.textContent = state.phase || '-';
      els.winner.textContent = state.winner || '-';
      els.requestType.textContent = state.request_type || '-';
      els.botName.textContent = state.bot_name || 'bot';
      els.runSeed.textContent = state.run_seed == null ? '-' : String(state.run_seed);
      const submitted = state.last_submitted_choice ? ` Last submitted: ${state.last_submitted_choice}.` : '';
      els.prompt.textContent = (state.prompt || '') + submitted;
      els.error.textContent = state.choice_error || state.last_error || '';
      els.publicState.textContent = state.public_state_text || '';
      els.recentMessages.textContent = (state.recent_messages || []).join('\\n');
      renderControls(state);
      renderTurnSummaries(state.turn_summaries || []);
    }

    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""

def _load_policy_bot(args: argparse.Namespace) -> dict[str, Any] | None:
    bot_mode = str(args.bot_mode or "mcts")
    checkpoint_path = str(args.checkpoint_path or "").strip()

    if checkpoint_path and bot_mode == "mcts":
        bot_mode = "policy"

    if bot_mode != "policy":
        return None

    if not checkpoint_path:
        raise ValueError("A checkpoint path is required when --bot-mode policy is selected")

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model, checkpoint = policy_train.load_policy_checkpoint(path)
    encoder_config = EncoderConfig(**_mapping(checkpoint.get("encoder_config")))
    bot_name = args.bot_name if args.bot_name != "mcts" else path.stem
    return {
        "checkpoint_path": path,
        "model": model,
        "encoder_config": encoder_config,
        "bot_name": bot_name,
    }


def _handle_team_preview(
    *,
    bridge: ctm.ShowdownBridge,
    human_side: str,
    bot_side: str,
    human_request: Mapping[str, Any],
    bot_request: Mapping[str, Any],
    human_name: str,
    bot_name: str,
) -> None:
    print()
    print("Team preview")
    print(f"{human_name} ({human_side}): {_preview_team(human_request)}")
    print(f"{bot_name} ({bot_side}): {_preview_team(bot_request)}")
    order = input("Enter your team order as a 6-digit permutation (default 123456): ").strip()
    if not order:
        order = "123456"
    if not _is_valid_team_order(order):
        raise ValueError("Team order must be a permutation of 123456")
    bridge.send({"type": "choice", "player": human_side, "choice": f"team {order}"})
    bridge.send({"type": "choice", "player": bot_side, "choice": "team 123456"})


def _choose_mcts_bot_action(
    *,
    backend: Any,
    rng: random.Random,
    catalog: Any,
    catalog_by_species: Any,
    species_dex: ctm.SpeciesDex,
    observed_state: Mapping[str, Any],
    actor_request: Mapping[str, Any],
    actor: str,
    duration_ms: int,
    threads: int,
    hypotheses: int,
    human_side: str,
    bot_side: str,
) -> str:
    policy = ctm._search_policy(
        backend=backend,
        rng=rng,
        catalog=catalog,
        catalog_by_species=catalog_by_species,
        species_dex=species_dex,
        observed_state=observed_state,
        actor_request=actor_request,
        actor=actor,
        duration_ms=duration_ms,
        threads=threads,
        hypotheses=hypotheses,
    )
    legal_policy = ctm._filter_policy_to_mask(observed_state, policy)
    chosen_action = max(legal_policy, key=legal_policy.get)
    return ctm._decision_to_showdown_choice(
        observed_state,
        actor_request,
        chosen_action,
    )


def _choose_policy_bot_action(
    *,
    policy_bot: Mapping[str, Any],
    observed_state: Mapping[str, Any],
    actor_request: Mapping[str, Any],
) -> str:
    model = policy_bot["model"]
    encoder_config = policy_bot["encoder_config"]
    policy = policy_train.predict_policy(
        model,
        observed_state,
        encoder_config=encoder_config,
    )
    legal_slots = [
        slot
        for slot, allowed in enumerate(observed_state.get("action_mask") or [])
        if allowed
    ]
    if not legal_slots:
        raise RuntimeError("No legal actions available for the policy bot")

    chosen_slot = max(
        legal_slots,
        key=lambda slot: policy[slot] if slot < len(policy) else float("-inf"),
    )
    decision = action_slot_to_decision(observed_state, chosen_slot)
    return ctm._decision_to_showdown_choice(
        observed_state,
        actor_request,
        decision,
    )


def _build_human_choices(
    *,
    human_state: Mapping[str, Any],
    human_request: Mapping[str, Any],
) -> list[dict[str, str]]:
    request_type = _request_type(human_request)
    if request_type == "teampreview":
        return []

    legal_slots = [
        slot
        for slot, allowed in enumerate(legal_action_mask(human_state))
        if allowed
    ]
    choices: list[dict[str, str]] = []
    for slot in legal_slots:
        decision = action_slot_to_decision(human_state, slot)
        command = ctm._decision_to_showdown_choice(human_state, human_request, decision)
        label = _choice_label_from_slot(human_state, slot)
        choices.append({"command": command, "label": label})
    return choices


@dataclass
class TurnActionSummary:
    side: str
    kind: str = "move"
    action: str = ""
    target: str = ""
    outcome: str = "hit"
    secondary_effects: list[str] = field(default_factory=list)


class TurnTranscript:
    def __init__(
        self,
        *,
        human_side: str,
        human_label: str = "You",
        opponent_label: str = "Opponent",
        on_summary: Any | None = None,
    ):
        self.human_side = human_side
        self.opponent_side = ctm._opponent(human_side)
        self.human_label = human_label
        self.opponent_label = opponent_label
        self.on_summary = on_summary
        self.current_turn: int | None = None
        self.actions: dict[str, TurnActionSummary] = {}
        self.last_actor: str | None = None

    def consume(self, messages: list[str]) -> None:
        for line in messages:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            tag = parts[1] if len(parts) > 1 else ""
            if tag == "turn" and len(parts) > 2:
                next_turn = _int(parts[2], self.current_turn or 0)
                if self.current_turn is not None and next_turn != self.current_turn:
                    self._print_summary()
                    self.actions.clear()
                    self.last_actor = None
                self.current_turn = next_turn
                continue
            if self.current_turn is None:
                continue
            self._consume_line(parts)

    def finalize(self) -> None:
        self._print_summary()
        self.actions.clear()
        self.last_actor = None

    def flush_pending(self) -> None:
        if self.current_turn is None or not self.actions:
            return
        self._print_summary()
        self.actions.clear()
        self.last_actor = None

    def _consume_line(self, parts: list[str]) -> None:
        tag = parts[1] if len(parts) > 1 else ""
        if tag in {"move", "switch", "drag", "replace"} and len(parts) >= 3:
            side = ctm._player_from_ident(parts[2])
            if not side:
                return
            action = self.actions.setdefault(side, TurnActionSummary(side=side))
            self.last_actor = side
            if tag == "move":
                action.kind = "move"
                action.action = _pretty_move(str(parts[3] if len(parts) > 3 else ""))
                action.target = _pretty_target(parts[4]) if len(parts) > 4 else ""
                action.outcome = "hit"
            else:
                action.kind = "switch"
                action.action = _pretty_target(parts[3]) if len(parts) > 3 else ""
                action.target = ""
                action.outcome = "switched"
            return

        if tag == "-miss":
            self._mark_last_actor("miss")
            return

        if tag == "-immune":
            self._mark_last_actor("immune")
            return

        if tag == "cant" and len(parts) >= 4:
            side = ctm._player_from_ident(parts[2]) or self.last_actor or ""
            if side:
                action = self.actions.setdefault(side, TurnActionSummary(side=side))
                reason = _pretty_move(str(parts[3]))
                action.outcome = f"failed ({reason})" if reason else "failed"
            return

        if tag == "-status" and len(parts) >= 4:
            self._add_secondary_effect(self.last_actor, f"{_status_name(parts[3])} on {_pretty_target(parts[2])}")
            return

        if tag == "-flinch" and len(parts) >= 3:
            self._add_secondary_effect(self.last_actor, f"flinched {_pretty_target(parts[2])}")
            return

        if tag == "-crit":
            self._add_secondary_effect(self.last_actor, "critical hit")
            return

        if tag == "-supereffective":
            self._add_secondary_effect(self.last_actor, "super effective")
            return

        if tag == "-resisted":
            self._add_secondary_effect(self.last_actor, "not very effective")
            return

    def _mark_last_actor(self, outcome: str) -> None:
        if not self.last_actor:
            return
        action = self.actions.setdefault(self.last_actor, TurnActionSummary(side=self.last_actor))
        action.outcome = outcome

    def _add_secondary_effect(self, side: str | None, effect: str) -> None:
        if not side:
            return
        action = self.actions.setdefault(side, TurnActionSummary(side=side))
        if effect not in action.secondary_effects:
            action.secondary_effects.append(effect)

    def _print_summary(self) -> None:
        if self.current_turn is None or not self.actions:
            return
        lines = [""]
        lines.append(f"End of turn {self.current_turn}")
        for side in (self.human_side, self.opponent_side):
            action = self.actions.get(side)
            label = self.human_label if side == self.human_side else self.opponent_label
            if action is None:
                lines.append(f"  {label}: no action logged")
                continue
            lines.append(f"  {label}: {self._format_action(action)}")
        if self.on_summary is not None:
            self.on_summary(lines)
        else:
            for line in lines:
                print(line)

    def _format_action(self, action: TurnActionSummary) -> str:
        if action.kind == "switch":
            detail = f"switch to {action.action or 'unknown'}"
        else:
            detail = action.action or "unknown move"
            if action.target:
                detail = f"{detail} -> {action.target}"
        if action.secondary_effects:
            effects = ", ".join(action.secondary_effects)
        else:
            effects = "none"
        return f"{detail} | {action.outcome} | secondary: {effects}"


def _prompt_human_choice(
    *,
    human_state: Mapping[str, Any],
    human_request: Mapping[str, Any],
    human_side: str,
    human_name: str,
    bot_name: str,
) -> str:
    legal_slots = [
        slot
        for slot, allowed in enumerate(legal_action_mask(human_state))
        if allowed
    ]
    if not legal_slots:
        raise RuntimeError("No legal actions available for the human player")

    options: list[tuple[int, str]] = []
    for index, slot in enumerate(legal_slots, start=1):
        decision = action_slot_to_decision(human_state, slot)
        choice = ctm._decision_to_showdown_choice(human_state, human_request, decision)
        options.append((index, choice))

    print(render_public_state(human_state, human_side=human_side, human_name=human_name, bot_name=bot_name))
    print("Legal actions:")
    for index, choice in options:
        print(f"  {index}. {choice}")

    while True:
        raw = input("Choose an action by number or type a legal Showdown command: ").strip()
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(options):
                return options[picked - 1][1]
        for _, choice in options:
            if raw == choice:
                return choice
        print("Invalid choice. Try one of the listed numbers or commands.")


def render_public_state(
    state: Mapping[str, Any],
    *,
    human_side: str,
    human_name: str,
    bot_name: str,
) -> str:
    user_side = _mapping(state.get("user"))
    opp_side = _mapping(state.get("opponent"))
    lines: list[str] = []
    lines.append("Public battle state")
    lines.append(f"Turn: {state.get('turn')}")
    lines.append(
        f"Weather: {state.get('weather', 'none')} | Field: {state.get('field', 'none')} | "
        f"Trick room: {bool(state.get('trick_room'))} | Gravity: {bool(state.get('gravity'))}"
    )
    lines.append("")
    lines.append(f"{human_name} ({human_side})")
    lines.append(f"  Active: {format_pokemon(_mapping(user_side.get('active')))}")
    lines.append(f"  Side conditions: {format_side_conditions(user_side.get('side_conditions'))}")
    lines.append("  Reserve:")
    for pokemon in _sequence(user_side.get("reserve"), 5):
        if pokemon:
            lines.append(f"    - {format_pokemon(_mapping(pokemon))}")
    lines.append("")
    lines.append(f"{bot_name} ({ctm._opponent(human_side)})")
    lines.append(f"  Active: {format_pokemon(_mapping(opp_side.get('active')))}")
    lines.append(f"  Side conditions: {format_side_conditions(opp_side.get('side_conditions'))}")
    lines.append("  Reserve:")
    for pokemon in _sequence(opp_side.get("reserve"), 5):
        if pokemon:
            lines.append(f"    - {format_pokemon(_mapping(pokemon))}")
    return "\n".join(lines)


def format_pokemon(pokemon: Mapping[str, Any]) -> str:
    if not pokemon:
        return "-"
    name = _pretty_name(pokemon.get("name") or pokemon.get("base_name"))
    hp = _int(pokemon.get("hp"), 0)
    max_hp = _int(pokemon.get("max_hp"), 0)
    hp_text = "0% fnt" if _is_fainted(pokemon) else (f"{hp}/{max_hp}" if max_hp else str(hp))
    parts = [name, hp_text]
    status = _text(pokemon.get("status"))
    if status and status != "none" and status != "fnt":
        parts.append(status)
    boosts = format_boosts(_mapping(pokemon.get("boosts")))
    if boosts:
        parts.append(f"boosts {boosts}")
    ability = _text(pokemon.get("ability"))
    item = _text(pokemon.get("item"))
    if ability:
        parts.append(f"ability {ability}")
    if item:
        parts.append(f"item {item}")
    moves = [move.get("name") for move in _sequence(pokemon.get("moves"), 4) if _mapping(move).get("name")]
    if moves:
        parts.append("moves " + ", ".join(_pretty_move(move) for move in moves))
    return " | ".join(parts)


def format_side_conditions(conditions: object) -> str:
    mapping = _mapping(conditions)
    if not mapping:
        return "none"
    pieces = []
    for key, value in sorted(mapping.items()):
        if value:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces) if pieces else "none"


def format_boosts(boosts: Mapping[str, Any]) -> str:
    labels = {
        "attack": "atk",
        "defense": "def",
        "special-attack": "spa",
        "special-defense": "spd",
        "speed": "spe",
        "accuracy": "acc",
        "evasion": "eva",
    }
    pieces: list[str] = []
    for stat, label in labels.items():
        value = _int(boosts.get(stat), 0)
        if value:
            sign = "+" if value > 0 else ""
            pieces.append(f"{label}{sign}{value}")
    return ", ".join(pieces)


def _is_fainted(pokemon: Mapping[str, Any]) -> bool:
    if pokemon.get("fainted") is not None:
        return bool(pokemon.get("fainted"))
    if pokemon.get("alive") is not None:
        return not bool(pokemon.get("alive"))
    status = _text(pokemon.get("status"))
    if status == "fnt":
        return True
    hp_fraction = pokemon.get("hp_fraction")
    try:
        return hp_fraction is not None and float(hp_fraction) <= 0.0
    except (TypeError, ValueError):
        return False


def _preview_team(request: Mapping[str, Any]) -> str:
    side = _mapping(request.get("side"))
    names = []
    for pokemon in _sequence(side.get("pokemon"), 6):
        pokemon_map = _mapping(pokemon)
        detail = str(pokemon_map.get("details") or "")
        name = detail.split(",")[0].strip() if detail else str(pokemon_map.get("ident") or "unknown")
        names.append(_pretty_name(name))
    return ", ".join(name for name in names if name)


def _is_valid_team_order(order: str) -> bool:
    return len(order) == 6 and sorted(order) == list("123456")


def _request_type(request: Mapping[str, Any]) -> str:
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        return "teampreview"
    if request.get("forceSwitch"):
        return "switch"
    return "move"


def _pretty_move(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def _choice_label_from_slot(state: Mapping[str, Any], slot: int) -> str:
    user = _mapping(state.get("user"))
    active = _mapping(user.get("active"))
    moves = _sequence(active.get("moves"), 4)
    if 0 <= slot < 4:
        move = _mapping(moves[slot])
        return f"Move {slot + 1}: {_pretty_move(move.get('name') or move.get('move') or 'unknown')}"
    if 4 <= slot < 9:
        reserve = _sequence(user.get("reserve"), 5)
        pokemon = _mapping(reserve[slot - 4])
        return f"Switch to {_pretty_name(pokemon.get('name') or pokemon.get('base_name') or 'unknown')}"
    move = _mapping(moves[slot - 9])
    return f"Tera move {slot - 8}: {_pretty_move(move.get('name') or move.get('move') or 'unknown')}"


def _pretty_target(value: object) -> str:
    text = str(value or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return _pretty_name(text)


def _status_name(value: object) -> str:
    mapping = {
        "brn": "burn",
        "par": "paralysis",
        "psn": "poison",
        "tox": "toxic poison",
        "slp": "sleep",
        "frz": "freeze",
    }
    text = _text(value)
    return mapping.get(text, _pretty_move(text)) if text else ""


def _pretty_name(value: object) -> str:
    text = str(value or "").replace("_", "-").strip()
    if not text:
        return ""
    return "-".join(part[:1].upper() + part[1:] for part in text.split("-") if part)


def _text(value: object) -> str:
    return "" if value in (None, "", "none") else str(value)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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


def _bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _generation_from_format(pokemon_format: str) -> str:
    text = str(pokemon_format)
    if text.startswith("gen") and len(text) >= 4 and text[3].isdigit():
        return f"gen{text[3]}"
    return "gen9"


def _resolve_run_seed(
    settings: Mapping[str, Any],
    args: argparse.Namespace,
    mcts: Mapping[str, Any],
) -> int:
    if args.seed is not None:
        return int(args.seed)
    value = settings.get("seed")
    if value is not None and value != 1337:
        return _int(value, 1337)
    value = mcts.get("seed")
    if value is not None and value != 1337:
        return _int(value, 1337)
    return random.SystemRandom().randrange(1, 2_147_483_647)


def _apply_overrides(settings: dict[str, Any], args: argparse.Namespace) -> None:
    showdex = _mapping(settings.get("showdex"))
    mcts = _mapping(settings.get("mcts"))
    settings["showdex"] = showdex
    settings["mcts"] = mcts

    for arg_name, key in (
        ("pokemon_format", "pokemon_format"),
        ("generation", "generation"),
        ("seed", "seed"),
        ("backend", "backend"),
        ("search_time_ms", "search_time_ms"),
        ("threads", "threads"),
        ("hypotheses", "hypotheses"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            settings[key] = value

    if args.randoms_preset_path:
        showdex["randoms_preset_path"] = args.randoms_preset_path
    if args.randoms_stats_path:
        showdex["randoms_stats_path"] = args.randoms_stats_path
    if args.use_fixture:
        showdex["use_embedded_fixture"] = True
    if args.allow_toy_backend:
        settings["allow_toy_backend"] = True
    settings["species_data_path"] = args.species_data_path


if __name__ == "__main__":
    main()
