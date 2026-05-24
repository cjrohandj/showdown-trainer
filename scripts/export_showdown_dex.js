#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

let Sim;
try {
  Sim = require("pokemon-showdown");
} catch (error) {
  process.stderr.write(
    `Missing Node package 'pokemon-showdown'. Run npm install first.\n${String(error && error.message || error)}\n`
  );
  process.exit(1);
}

const outputPath = process.argv[2] || "showdex_cache/showdown_species.json";
const Dex = Sim.Dex;

function normalizeStats(baseStats) {
  return {
    hp: Number(baseStats.hp || 100),
    attack: Number(baseStats.atk || 100),
    defense: Number(baseStats.def || 100),
    "special-attack": Number(baseStats.spa || 100),
    "special-defense": Number(baseStats.spd || 100),
    speed: Number(baseStats.spe || 100),
  };
}

const speciesTable = {};
for (const species of Dex.species.all()) {
  if (!species.exists || species.isNonstandard === "CAP") continue;
  speciesTable[species.id] = {
    id: species.id,
    name: species.name,
    base_species: species.baseSpecies || species.name,
    forme: species.forme || "",
    types: species.types || [],
    base_stats: normalizeStats(species.baseStats || {}),
    weight_kg: Number(species.weightkg || 0),
    abilities: species.abilities || {},
  };
}

fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, `${JSON.stringify(speciesTable, null, 2)}\n`);
process.stdout.write(`Wrote ${Object.keys(speciesTable).length} species to ${outputPath}\n`);
