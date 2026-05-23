#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-collector"
PYTHON_INSTALL_VERSION="${PYTHON_INSTALL_VERSION:-3.13.7}"
PYTHON_PKG_URL="${PYTHON_PKG_URL:-https://www.python.org/ftp/python/${PYTHON_INSTALL_VERSION}/python-${PYTHON_INSTALL_VERSION}-macos11.pkg}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=("${PYTHON_BIN}")
else
  PYTHON_CANDIDATES=(
    python3.13
    python3.12
    python3.11
    python3.10
    /opt/homebrew/opt/python@3.13/bin/python3.13
    /usr/local/opt/python@3.13/bin/python3.13
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
    python3
  )
fi

cd "${ROOT_DIR}"

find_compatible_python() {
  local candidate
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ "${candidate}" == */* ]]; then
      [[ -x "${candidate}" ]] || continue
    elif ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

install_python() {
  echo "==> Python 3.10 through 3.13 not found; installing Python ${PYTHON_INSTALL_VERSION}"
  if command -v brew >/dev/null 2>&1; then
    brew install python@3.13
    PYTHON_CANDIDATES=(
      python3.13
      /opt/homebrew/opt/python@3.13/bin/python3.13
      /usr/local/opt/python@3.13/bin/python3.13
      "${PYTHON_CANDIDATES[@]}"
    )
    return 0
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Automatic Python install is only supported on macOS." >&2
    return 1
  fi

  local tmp_dir
  local pkg_path
  tmp_dir="$(mktemp -d)"
  pkg_path="${tmp_dir}/python-${PYTHON_INSTALL_VERSION}.pkg"
  echo "Downloading ${PYTHON_PKG_URL}"
  curl -L "${PYTHON_PKG_URL}" -o "${pkg_path}"
  echo "Installing Python ${PYTHON_INSTALL_VERSION}; macOS may ask for your password."
  sudo installer -pkg "${pkg_path}" -target /
  rm -rf "${tmp_dir}"
  PYTHON_CANDIDATES=(
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
    python3.13
    "${PYTHON_CANDIDATES[@]}"
  )
}

echo "==> Checking Python"
SELECTED_PYTHON="$(find_compatible_python || true)"

if [[ -z "${SELECTED_PYTHON}" ]]; then
  install_python
  SELECTED_PYTHON="$(find_compatible_python || true)"
  if [[ -z "${SELECTED_PYTHON}" ]]; then
    echo "Python install finished, but no compatible Python was found." >&2
    echo "Set PYTHON_BIN=/path/to/python3.13 and rerun ./scripts/collect_mac_shard.sh." >&2
    exit 1
  fi
fi

"${SELECTED_PYTHON}" - <<'PY'
import sys

print(f"Using Python {sys.version.split()[0]} at {sys.executable}")
PY

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  if ! "${VENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)
PY
  then
    echo "==> Recreating collector virtual environment with a poke-engine-compatible Python"
    rm -rf "${VENV_DIR}"
  fi
fi

echo "==> Creating collector virtual environment"
"${SELECTED_PYTHON}" -m venv "${VENV_DIR}"

echo "==> Installing collector dependencies"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements-collector.txt

echo "==> Downloading Showdex/pkmn random battle data"
mkdir -p showdex_cache
"${VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    Path("showdex_cache/gen9randombattle.json"): "https://pkmn.github.io/randbats/data/gen9randombattle.json",
    Path("showdex_cache/gen9randombattle-stats.json"): "https://pkmn.github.io/randbats/data/stats/gen9randombattle-stats.json",
}

for path, url in FILES.items():
    if path.exists() and path.stat().st_size > 0:
        print(f"Already have {path}")
        continue
    print(f"Downloading {url}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    urlretrieve(url, tmp_path)
    tmp_path.replace(path)
PY

echo "==> Verifying poke-engine collector with a tiny smoke run"
"${VENV_DIR}/bin/python" collect_offline_mcts.py \
  --config configs/student.yaml \
  --output-path training_data/smoke_mac_setup.jsonl \
  --positions 2 \
  --search-time-ms 10 \
  --hypotheses 1 \
  --threads 1

echo
echo "Setup complete."
echo "Run data collection with:"
echo "  ./scripts/collect_mac_shard.sh"
