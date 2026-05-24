# Offline Showdex-Guided MCTS Training Pipeline

## Offline Pipeline

The collection path is:

1. `collect_offline_mcts.py` loads local Showdex-compatible pkmn data from
   `showdex_cache/`.
2. `showdex_distributions.py` turns the randoms preset or usage JSON into
   weighted distributions over hidden variables: role, ability, item, moves,
   EVs/IVs, and tera type.
3. `offline_mcts.py` samples an observed position and several concrete hidden
   hypotheses. The observed training state keeps opponent unknowns as
   distributions, while each hypothesis samples concrete values for
   `poke-engine`.
4. The `PokeEngineBackend` builds `poke_engine.State` objects and calls
   `poke_engine.monte_carlo_tree_search(...)`.
5. MCTS visit policies are averaged across hypotheses and written through the
   existing `TrainingDatasetWriter`.
6. `train.py` trains the same masked 13-action policy network on those MCTS
   target distributions.

The important change is that Showdex guides search before the policy target is
created. The network is no longer trained on one concrete guess for an unknown
opponent set. It is trained on the action distribution that comes from averaging
poke-engine MCTS over Showdex/pkmn hidden-variable samples.

## Offline Capability

Once `showdex_cache/gen9randombattle.json` and
`showdex_cache/gen9randombattle-stats.json` exist, collection does not require
Pokemon Showdown, browser extension runtime state, or network access. The
notebook downloads those files once from the same pkmn data paths Showdex uses,
then collection reads from disk.

For a fully air-gapped run:

1. Pre-place the two JSON files in `showdex_cache/`.
2. Install the Python dependencies from `requirements.txt` or `environment.yml`.
3. Run `python collect_offline_mcts.py --config configs/student.yaml`.
4. Run `python train.py --config configs/student.yaml`.

## Colab Defaults

`configs/student.yaml` is tuned to keep first Colab runs short:

- `positions: 5000`
- `hypotheses: 4`
- `search_time_ms: 75`
- `training.updates: 400`
- `model.hidden_sizes: [512, 256]`
- `model.hash_buckets: 2048`
- checkpoint output at `checkpoints/policy_mlp.pt`

Use `--positions`, `--hypotheses`, and `--search-time-ms` to scale collection
up. Use `training.resume_checkpoint_path` or `--resume-checkpoint-path` to keep
continuing from a saved checkpoint.

## Notes And Limits

`poke-engine` is an intentionally compact battle engine rather than a full
Pokemon Showdown clone. The upstream README describes it as singles-focused and
not as complete as Pokemon Showdown. This repo therefore treats the collector as
an offline policy-target generator, not an exact simulator replacement for all
formats.

The current offline state builder focuses on the variables Showdex/pkmn exposes
directly. If you want higher-fidelity damage and typing, the next useful
extension is a local species metadata table for exact types, weights, and
calculated stats. The collector is structured so that can be added without
changing the JSONL schema or trainer.
