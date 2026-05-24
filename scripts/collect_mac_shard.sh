#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv-collector/bin/python"

cd "${ROOT_DIR}"

collector_env_ok() {
  if ! "${VENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

ok = (3, 10) <= sys.version_info < (3, 14)
ok = ok and importlib.util.find_spec("yaml") is not None
ok = ok and importlib.util.find_spec("poke_engine") is not None
raise SystemExit(0 if ok else 1)
PY
  then
    return 1
  fi
  command -v node >/dev/null 2>&1 \
    && node -e 'require("pokemon-showdown")' >/dev/null 2>&1
}

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "==> Collector environment is missing; running first-time setup"
  ./scripts/mac_setup.sh
elif ! collector_env_ok
then
  echo "==> Collector environment needs repair; running setup"
  ./scripts/mac_setup.sh
fi

GAMES="${GAMES:-10000}"
MAX_TURNS="${MAX_TURNS:-100}"
SEARCH_TIME_MS="${SEARCH_TIME_MS:-100}"
HYPOTHESES="${HYPOTHESES:-4}"
THREADS="${THREADS:-6}"
POKEMON_FORMAT="${POKEMON_FORMAT:-gen9randombattle}"
GENERATION="${GENERATION:-gen9}"
COLLECTOR_ID="${COLLECTOR_ID:-$(whoami)-$(hostname -s 2>/dev/null || hostname)}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SEED="${SEED:-$("${VENV_PYTHON}" - <<'PY'
import random
print(random.SystemRandom().randint(1, 2_147_483_647))
PY
)}"

SAFE_COLLECTOR_ID="$(printf '%s' "${COLLECTOR_ID}" | tr -cs 'A-Za-z0-9_.-' '_')"
SHARD_DIR="${SHARD_DIR:-training_data/shards}"
OUTPUT_PATH="${OUTPUT_PATH:-${SHARD_DIR}/trajectory_mcts_${SAFE_COLLECTOR_ID}_${TIMESTAMP}_seed${SEED}.jsonl}"
SUMMARY_PATH="${OUTPUT_PATH%.jsonl}.summary.json"
SUMMARY_MD_PATH="${OUTPUT_PATH%.jsonl}.summary.md"
ARCHIVE_PATH="${OUTPUT_PATH}.gz"

mkdir -p "${SHARD_DIR}"

echo "==> Starting MCTS data shard"
echo "collector_id=${COLLECTOR_ID}"
echo "games=${GAMES}"
echo "max_turns=${MAX_TURNS}"
echo "search_time_ms=${SEARCH_TIME_MS}"
echo "hypotheses=${HYPOTHESES}"
echo "threads=${THREADS}"
echo "seed=${SEED}"
echo "output_path=${OUTPUT_PATH}"

"${VENV_PYTHON}" collect_trajectory_mcts.py \
  --config configs/student.yaml \
  --output-path "${OUTPUT_PATH}" \
  --games "${GAMES}" \
  --max-turns "${MAX_TURNS}" \
  --search-time-ms "${SEARCH_TIME_MS}" \
  --hypotheses "${HYPOTHESES}" \
  --threads "${THREADS}" \
  --pokemon-format "${POKEMON_FORMAT}" \
  --generation "${GENERATION}" \
  --seed "${SEED}" | tee "${SUMMARY_PATH}"

gzip -c "${OUTPUT_PATH}" > "${ARCHIVE_PATH}"

"${VENV_PYTHON}" scripts/summarize_shard.py "${OUTPUT_PATH}" --output "${SUMMARY_MD_PATH}"

echo
echo "Shard complete."
echo "Send this compressed file back:"
echo "  ${ARCHIVE_PATH}"
echo "Summary:"
echo "  ${SUMMARY_PATH}"
echo "Human-readable summary:"
echo "  ${SUMMARY_MD_PATH}"
