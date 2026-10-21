#!/bin/bash

# Force the dot as decimal separator
export LC_ALL=C

# ---------------- Parameters ----------------
EXEC="./AreaMC_MPI"
POINTS="5000000000"
INPUT_FILE="input.txt"
OUTPUT_FILE="risultati_strong_scaling_mpi.csv"
NUM_RUNS=10
PROCS=(1 2 4 8 16 32 40 64 80)

# ---------------- Preliminary checks ----------------
if [ ! -x "$EXEC" ]; then
    echo "ERROR: executable '$EXEC' not found or not executable."
    echo "Compile with: mpicc -O2 -fopenmp-simd -o AreaMC_MPI_datatypes AreaMC_MPI_datatypes.c"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file '$INPUT_FILE' not found in $(pwd)."
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found in PATH."
    exit 1
fi

# ---------------- mpirun options depending on the implementation ----------------
MPI_VERSION=$(mpirun --version 2>&1)
MPI_OPTS=()

if echo "$MPI_VERSION" | grep -qi "open mpi\|open-mpi"; then
    MPI_OPTS+=(--oversubscribe --use-hwthread-cpus --bind-to core --map-by core)
    # Required if the script runs as root (Docker, WSL, containers)
    if [ "$(id -u)" -eq 0 ]; then
        MPI_OPTS+=(--allow-run-as-root)
    fi
else
    echo "WARNING: non-OpenMPI implementation detected, using mpirun without extra options."
fi

# ---------------- CSV writing (also in case of interruption) ----------------
declare -a ROW_DATA
for idx in "${!PROCS[@]}"; do
    ROW_DATA[$idx]="${PROCS[$idx]}"
done

write_csv() {
    local header="Processi"
    for (( i=1; i<=NUM_RUNS; i++ )); do
        header="$header,Run$i"
    done
    echo "$header" > "$OUTPUT_FILE"
    for idx in "${!PROCS[@]}"; do
        echo "${ROW_DATA[$idx]}" >> "$OUTPUT_FILE"
    done
    echo "Data saved in $OUTPUT_FILE."
}
# Write the CSV whenever the script exits (normal end, error or Ctrl+C)
trap write_csv EXIT

echo "==================================================="
echo " Starting MPI Strong Scaling test ($NUM_RUNS runs)"
echo " Executable : $EXEC"
echo " Points     : $POINTS"
echo " Input      : $INPUT_FILE"
echo " MPI options: ${MPI_OPTS[*]}"
echo "==================================================="

# ---------------- Main loop ----------------
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "--- Starting Run $run of $NUM_RUNS ---"

    for idx in "${!PROCS[@]}"; do
        P="${PROCS[$idx]}"
        echo -n "  Test with $P processes... "

        OUTPUT=$(mpirun "${MPI_OPTS[@]}" -n "$P" "$EXEC" "$POINTS" "$INPUT_FILE" 2>&1)

        # Extract the execution time from the "Execution time" line
        TIME_SEC=$(echo "$OUTPUT" | grep "Execution time" | awk -F ': ' '{print $2}' | awk '{print $1}')

        if [ -z "$TIME_SEC" ]; then
            echo "ERR"
            echo "    mpirun output (first lines):"
            echo "$OUTPUT" | head -n 8 | sed 's/^/    | /'
            TIME_SEC="ERR"
        else
            echo "$TIME_SEC s"
        fi

        ROW_DATA[$idx]="${ROW_DATA[$idx]},$TIME_SEC"
    done
done

echo "==================================================="
echo "Test completed!"
