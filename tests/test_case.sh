#!/usr/bin/env bash
set -euo pipefail

BIN_PATH="${1:-./ns_dg_solver}"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "ERROR: binary not found or not executable: $BIN_PATH" >&2
  exit 1
fi

run_and_check() {
  local p="$1"
  local outfile
  outfile="$(mktemp)"
  trap 'rm -f "$outfile"' RETURN

  "$BIN_PATH" "$p" >"$outfile"

  if ! grep -q '^Done\.' "$outfile"; then
    echo "ERROR: missing completion line for p=$p" >&2
    cat "$outfile" >&2
    exit 1
  fi

  python - <<'PY' "$outfile" "$p"
import math
import re
import sys

path = sys.argv[1]
p = sys.argv[2]
text = open(path, 'r', encoding='utf-8').read()
match = re.search(r'final_mass=([0-9eE+\-.]+)', text)
if not match:
    raise SystemExit(f"ERROR: final_mass not found for p={p}")
mass = float(match.group(1))
if not math.isfinite(mass):
    raise SystemExit(f"ERROR: non-finite mass for p={p}: {mass}")
if mass <= 0.0:
    raise SystemExit(f"ERROR: non-positive mass for p={p}: {mass}")
print(f"p={p} final_mass={mass:.6e}")
PY
}

# Simple regression-style smoke cases
run_and_check 0
run_and_check 2

echo "All simple DG solver tests passed."
