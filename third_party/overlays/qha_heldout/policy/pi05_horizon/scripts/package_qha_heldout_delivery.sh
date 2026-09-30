#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QHA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${QHA_ROOT}/../.." && pwd)"

MODEL=""
STEP="10000"
ROOT_TAG="pi05_base_aloha_robotwin_full_qha_heldout"
OUTPUT=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/package_qha_heldout_delivery.sh --model FORMAL_RUN_NAME [options]

Build a portable QHA held-out delivery archive only after the recorded final
QHA-only checkpoint exists. The archive contains the head, audit, frozen
protocol/config files, and the runtime files required for manifest evaluation.

Options:
  --model NAME       Formal run name under the QHA checkpoint root (required)
  --exp-name NAME    Alias for --model; used by the formal training finalizer
  --step N           QHA-only checkpoint step (default: 10000)
  --root-tag TAG     QHA checkpoint root tag
  --output PATH      Archive destination; default is deliveries/ under QHA root
USAGE
}

die() { echo "ERROR: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--exp-name) MODEL="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --root-tag) ROOT_TAG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -n "${MODEL}" ]] || die "--model is required"
[[ "${STEP}" =~ ^[1-9][0-9]*$ ]] || die "--step must be a positive integer"

RUN_REL="policy/pi05_horizon/checkpoints/${ROOT_TAG}/${MODEL}"
HEAD_REL="${RUN_REL}/${STEP}"
AUDIT_REL="${RUN_REL}/audit"
HEAD_DIR="${REPO_ROOT}/${HEAD_REL}"
AUDIT_DIR="${REPO_ROOT}/${AUDIT_REL}"

[[ -f "${HEAD_DIR}/assets/_QHA_ONLY" ]] || die "Not a finalized QHA-only checkpoint: ${HEAD_DIR}"
[[ -f "${AUDIT_DIR}/training_audit.json" ]] || die "Missing training audit: ${AUDIT_DIR}"
[[ -f "${AUDIT_DIR}/checkpoints.jsonl" ]] || die "Missing checkpoint audit: ${AUDIT_DIR}"
python "${QHA_ROOT}/scripts/verify_qha_heldout_run.py" \
  --run-dir "${REPO_ROOT}/${RUN_REL}" --step "${STEP}" --require-final --repo-root "${REPO_ROOT}"

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${QHA_ROOT}/deliveries/${MODEL}_step${STEP}.tar.gz"
fi
OUTPUT="$(realpath -m "${OUTPUT}")"
mkdir -p "$(dirname "${OUTPUT}")"
[[ ! -e "${OUTPUT}" ]] || die "Refusing to overwrite existing archive: ${OUTPUT}"

SOURCE_MANIFEST="${AUDIT_DIR}/training_runtime_source_manifest.json"
[[ -f "${SOURCE_MANIFEST}" ]] || die "Missing runtime source manifest: ${SOURCE_MANIFEST}"
mapfile -t SOURCE_FILES < <(python - "${SOURCE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("runtime source manifest has no files")
for path in files:
    print(path)
PY
)

FILES=(
  "${HEAD_REL}"
  "${AUDIT_REL}"
  "${SOURCE_FILES[@]}"
  "policy/pi05_horizon/scripts/package_qha_heldout_delivery.sh"
  "policy/pi05_horizon/UBUNTU_HELDOUT_EVAL.md"
  "policy/pi05_horizon/UBUNTU_HELDOUT_SMOKE_EVAL.md"
  "policy/pi05_horizon/UBUNTU_HELDOUT_LOCAL_RUNBOOK.md"
  "policy/pi05_horizon/UBUNTU_CODEX_HELDOUT_PROMPT.md"
)

tar -C "${REPO_ROOT}" -czf "${OUTPUT}" "${FILES[@]}"
sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"
printf 'archive=%s\nsha256=%s\n' "${OUTPUT}" "$(awk '{print $1}' "${OUTPUT}.sha256")"
