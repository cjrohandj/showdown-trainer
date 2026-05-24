#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_NAME="${BUNDLE_NAME:-showdown-trainer-mac-collector}"
DIST_DIR="${ROOT_DIR}/dist"
BUILD_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.tar.gz"

cd "${ROOT_DIR}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/configs" "${BUILD_DIR}/docs" "${BUILD_DIR}/scripts"

cp \
  collect_offline_mcts.py \
  dataset.py \
  encode.py \
  map.py \
  offline_mcts.py \
  requirements-collector.txt \
  showdex_distributions.py \
  trainer_config.py \
  "${BUILD_DIR}/"

cp configs/student.yaml "${BUILD_DIR}/configs/student.yaml"
cp docs/mac_data_collection.md "${BUILD_DIR}/docs/mac_data_collection.md"
cp scripts/mac_setup.sh "${BUILD_DIR}/scripts/mac_setup.sh"
cp scripts/collect_mac_shard.sh "${BUILD_DIR}/scripts/collect_mac_shard.sh"

chmod +x "${BUILD_DIR}/scripts/mac_setup.sh" "${BUILD_DIR}/scripts/collect_mac_shard.sh"

LC_ALL=C tar -C "${DIST_DIR}" -czf "${ARCHIVE_PATH}" "${BUNDLE_NAME}"

echo "Created ${ARCHIVE_PATH}"
