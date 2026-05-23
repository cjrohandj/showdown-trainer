#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv-collector/bin/python"

cd "${ROOT_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "==> Collector environment is missing; running first-time setup"
  ./scripts/mac_setup.sh
elif ! "${VENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

ok = (3, 10) <= sys.version_info < (3, 14)
ok = ok and importlib.util.find_spec("yaml") is not None
ok = ok and importlib.util.find_spec("poke_engine") is not None
raise SystemExit(0 if ok else 1)
PY
then
  echo "==> Collector environment needs repair; running setup"
  ./scripts/mac_setup.sh
fi

POSITIONS="${POSITIONS:-5000}"
SEARCH_TIME_MS="${SEARCH_TIME_MS:-75}"
HYPOTHESES="${HYPOTHESES:-4}"
THREADS="${THREADS:-1}"
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
OUTPUT_PATH="${OUTPUT_PATH:-${SHARD_DIR}/mcts_${SAFE_COLLECTOR_ID}_${TIMESTAMP}_seed${SEED}.jsonl}"
SUMMARY_PATH="${OUTPUT_PATH%.jsonl}.summary.json"
ARCHIVE_PATH="${OUTPUT_PATH}.gz"

mkdir -p "${SHARD_DIR}"

echo "==> Starting MCTS data shard"
echo "collector_id=${COLLECTOR_ID}"
echo "positions=${POSITIONS}"
echo "search_time_ms=${SEARCH_TIME_MS}"
echo "hypotheses=${HYPOTHESES}"
echo "threads=${THREADS}"
echo "seed=${SEED}"
echo "output_path=${OUTPUT_PATH}"

"${VENV_PYTHON}" collect_offline_mcts.py \
  --config configs/student.yaml \
  --output-path "${OUTPUT_PATH}" \
  --positions "${POSITIONS}" \
  --search-time-ms "${SEARCH_TIME_MS}" \
  --hypotheses "${HYPOTHESES}" \
  --threads "${THREADS}" \
  --pokemon-format "${POKEMON_FORMAT}" \
  --generation "${GENERATION}" \
  --seed "${SEED}" | tee "${SUMMARY_PATH}"

gzip -c "${OUTPUT_PATH}" > "${ARCHIVE_PATH}"

echo
echo "Shard complete."
echo "Send this compressed file back:"
echo "  ${ARCHIVE_PATH}"
echo "Summary:"
echo "  ${SUMMARY_PATH}"
