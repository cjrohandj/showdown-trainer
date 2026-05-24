# Mac Data Collection Setup

This is the lightest way to run the full-game Showdown + MCTS trajectory
collector on other Macs. It does not install PyTorch or train the model. Each
machine writes independent JSONL shards that can be sent back and merged later.

## Make A Friend-Friendly Bundle

From your repo checkout:

```bash
./scripts/make_mac_collector_bundle.sh
```

Send your friend `dist/showdown-trainer-mac-collector.tar.gz`. They can unpack
it anywhere:

```bash
tar -xzf showdown-trainer-mac-collector.tar.gz
cd showdown-trainer-mac-collector
```

## What Your Friend Needs

- macOS
- Python 3.10 through 3.13
- Node.js and npm
- Internet access for first run

If they do not have Python 3.10-3.13, install Python 3.13 from
Homebrew or the official Python.org macOS package. `poke-engine` does not
currently build cleanly on Python 3.14.

If they do not have Node.js/npm, install Node.js from Homebrew or
https://nodejs.org/. The setup script uses npm to install the local
`pokemon-showdown` simulator package.

## Run A Data Shard

From the repo folder:

```bash
./scripts/collect_mac_shard.sh
```

On first run, the script installs Python 3.13 if needed, creates
`.venv-collector`, installs the Python collector dependencies, installs the
Node Pokemon Showdown simulator dependency, exports Pokemon Showdown species
metadata into `showdex_cache/showdown_species.json`, downloads the Showdex/pkmn
random battle data into `showdex_cache/`, runs a tiny full-game trajectory smoke
test, and then starts the real shard.

The default shard is 100 full games. Useful longer run:

```bash
GAMES=1000 SEARCH_TIME_MS=75 HYPOTHESES=4 THREADS=1 ./scripts/collect_mac_shard.sh
```

The runner writes:

- `training_data/shards/*.jsonl`
- `training_data/shards/*.jsonl.gz`
- `training_data/shards/*.summary.json`

Have your friend send back the `.jsonl.gz` file.

## Tuning

Start with `THREADS=1`. On larger Macs, running multiple terminal windows with
different `COLLECTOR_ID` values can give better throughput than one run with a
large thread count:

```bash
COLLECTOR_ID=alice-mac-a GAMES=500 THREADS=1 ./scripts/collect_mac_shard.sh
COLLECTOR_ID=alice-mac-b GAMES=500 THREADS=1 ./scripts/collect_mac_shard.sh
```

Main knobs:

- `GAMES`: more full games and more runtime
- `MAX_TURNS`: safety cutoff for unusually long games
- `SEARCH_TIME_MS`: stronger MCTS targets and more runtime
- `HYPOTHESES`: better hidden-information averaging and more runtime
- `THREADS`: threads inside each poke-engine search

## Merge Returned Shards

Put the returned `.jsonl.gz` files in one folder, then from this repo:

```bash
mkdir -p training_data/merged
gzip -dc returned_shards/*.jsonl.gz > training_data/merged/mcts_all.jsonl
```

Train against the merged file:

```bash
python train.py --config configs/student.yaml training_data/merged/mcts_all.jsonl
```

## Resume Or Restart

Each runner invocation creates a new timestamped shard. If a run stops halfway,
keep the partial `.jsonl` only if the summary says examples were written, then
start a new shard with the same command.
