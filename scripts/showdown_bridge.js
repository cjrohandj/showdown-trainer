#!/usr/bin/env node
"use strict";

let Sim;
try {
  Sim = require("pokemon-showdown");
} catch (error) {
  process.stdout.write(JSON.stringify({
    type: "error",
    message: "Missing Node package 'pokemon-showdown'. Run `npm install` in the repo first.",
    detail: String(error && error.message || error),
  }) + "\n");
  process.exit(1);
}

const readline = require("node:readline");

let stream = null;
let outputLoopStarted = false;

function emit(event) {
  process.stdout.write(JSON.stringify(event) + "\n");
}

function writeBattle(message) {
  if (!stream) {
    throw new Error("battle stream has not been started");
  }
  stream.write(message);
}

function parseOutput(chunk) {
  const lines = String(chunk).split("\n");
  const kind = lines.shift();

  if (kind === "sideupdate") {
    const player = lines.shift();
    const messages = lines;
    for (const line of messages) {
      if (line.startsWith("|request|")) {
        const raw = line.slice("|request|".length);
        if (raw) {
          try {
            emit({type: "request", player, request: JSON.parse(raw), messages});
          } catch (error) {
            emit({type: "error", message: "failed to parse request", detail: String(error)});
          }
        }
      }
    }
    emit({type: "sideupdate", player, messages});
    return;
  }

  if (kind === "update") {
    emit({type: "update", messages: lines});
    return;
  }

  if (kind === "end") {
    let data = {};
    const raw = lines.join("\n").trim();
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch {
        data = {raw};
      }
    }
    emit({type: "end", data});
    return;
  }

  emit({type: "raw", kind, messages: lines});
}

function startOutputLoop() {
  if (outputLoopStarted || !stream) return;
  outputLoopStarted = true;
  (async () => {
    try {
      for await (const output of stream) {
        parseOutput(output);
      }
    } catch (error) {
      emit({type: "error", message: "battle stream failed", detail: String(error)});
    }
  })();
}

function startBattle(command) {
  stream = new Sim.BattleStream();
  outputLoopStarted = false;
  startOutputLoop();

  const options = {
    formatid: command.formatid || "gen9randombattle",
  };
  if (Array.isArray(command.seed)) {
    options.seed = command.seed;
  }

  writeBattle(`>start ${JSON.stringify(options)}`);
  writeBattle(`>player p1 ${JSON.stringify(command.p1 || {name: "p1"})}`);
  writeBattle(`>player p2 ${JSON.stringify(command.p2 || {name: "p2"})}`);
  emit({type: "started"});
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", line => {
  if (!line.trim()) return;
  let command;
  try {
    command = JSON.parse(line);
  } catch (error) {
    emit({type: "error", message: "invalid JSON command", detail: String(error)});
    return;
  }

  try {
    if (command.type === "start") {
      startBattle(command);
    } else if (command.type === "choice") {
      writeBattle(`>${command.player} ${command.choice}`);
    } else if (command.type === "stop") {
      rl.close();
    } else {
      emit({type: "error", message: `unknown command type: ${command.type}`});
    }
  } catch (error) {
    emit({type: "error", message: "command failed", detail: String(error)});
  }
});
