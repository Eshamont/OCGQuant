#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-32}"
python "${SCRIPT_DIR}/main.py"
