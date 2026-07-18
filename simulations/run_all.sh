#!/usr/bin/env bash
# Runs osiris.1e for every simulation subfolder in this directory.
# Each subfolder must contain a weibel.in deck; output/log stay inside that
# subfolder so runs don't clobber each other.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROCS=10

cd "$SCRIPT_DIR"

failed=()

for sim_dir in */; do
    sim_dir="${sim_dir%/}"
    deck="$sim_dir/weibel.in"

    if [[ ! -f "$deck" ]]; then
        continue
    fi

    echo "=== Running $sim_dir ==="
    (
        cd "$sim_dir" && mpirun -n "$NPROCS" /Users/joaobiu/Developer/osiris-boost/bin/osiris-1D.e weibel.in
    ) 2>&1 | tee "$sim_dir/run.log"

    # PIPESTATUS[0] is mpirun's exit code (tee always exits 0)
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "!!! $sim_dir FAILED (see $sim_dir/run.log)"
        failed+=("$sim_dir")
    fi
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All simulations completed successfully."
else
    echo "Failed simulations (${#failed[@]}):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
