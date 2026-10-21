#!/bin/bash

# Force the dot as decimal separator
export LC_ALL=C

# ---------------- MPI program parameters ----------------
EXEC="./AreaMC_MPI"
BASE_M=10000000        # Points PER SINGLE PROCESS (Weak Scaling)
INPUT_FILE="input.txt" # Input vertex file
OUTPUT_FILE="risultati_weak_scaling_mpi.csv"
NUM_RUNS=10

# Sequence of MPI processes
PROCS=(1 2 4 8 16 32 40 64 80)

# Estimated number of physical cores available on the node (e.g. 40 on DGX-1)
PHYSICAL_CORES=40

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

# ---------------- Base mpirun options ----------------
# Note: --bind-to and --map-by are handled dynamically based on $P in the loop
MPI_OPTS=(--oversubscribe --mca btl ^openib)

# Required if the script runs as root (e.g. in Docker containers / special systems)
if [ "$(id -u)" -eq 0 ]; then
    MPI_OPTS+=(--allow-run-as-root)
fi

# ---------------- CSV initialization ----------------
HEADER="Processi,Punti_Totali"
for (( i=1; i<=NUM_RUNS; i++ )); do
    HEADER="$HEADER,Run$i"
done
echo "$HEADER" > "$OUTPUT_FILE"

echo "==================================================="
echo " Starting MPI Weak Scaling test ($NUM_RUNS runs)"
echo " Executable         : $EXEC"
echo " Points per process : $BASE_M"
echo " Output             : $OUTPUT_FILE"
echo " Phys. cores limit  : $PHYSICAL_CORES"
echo "==================================================="

# Initialize the array that stores the result rows
declare -a ROW_DATA
for idx in "${!PROCS[@]}"; do
    P="${PROCS[$idx]}"
    CURRENT_M=$(( BASE_M * P ))
    ROW_DATA[$idx]="$P,$CURRENT_M"
done

# ---------------- Main loop ----------------
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "--- Starting Run $run of $NUM_RUNS ---"

    for idx in "${!PROCS[@]}"; do
        P="${PROCS[$idx]}"
        CURRENT_M=$(( BASE_M * P ))

        echo -n "  Test with $P processes ($CURRENT_M points)... "

        # Dynamic binding management based on the number of processes
        if [ "$P" -le "$PHYSICAL_CORES" ]; then
            # Within the physical cores: strict binding and mapping by core
            OUTPUT=$(mpirun "${MPI_OPTS[@]}" --bind-to core --map-by core -n "$P" "$EXEC" "$CURRENT_M" "$INPUT_FILE" 2>&1)
        else
            # Beyond the physical cores (over-subscription): disable strict binding
            OUTPUT=$(mpirun "${MPI_OPTS[@]}" --bind-to none -n "$P" "$EXEC" "$CURRENT_M" "$INPUT_FILE" 2>&1)
        fi

        # Robustly extract the execution time
        TIME_SEC=$(echo "$OUTPUT" | grep "Execution time" | awk -F ': ' '{print $2}' | awk '{print $1}')

        if [ -z "$TIME_SEC" ]; then
            echo "ERR"
            echo "    mpirun output (first lines):"
            echo "$OUTPUT" | head -n 8 | sed 's/^/    | /'
            TIME_SEC="ERR"
        else
            echo "$TIME_SEC s"
        fi

        # Append the extracted time to the corresponding row string
        ROW_DATA[$idx]="${ROW_DATA[$idx]},$TIME_SEC"
    done
done

echo "==================================================="
echo "Writing data to the CSV file..."

# Write all the accumulated rows to the CSV file
for idx in "${!PROCS[@]}"; do
    echo "${ROW_DATA[$idx]}" >> "$OUTPUT_FILE"
done

echo "Test completed! Data saved in $OUTPUT_FILE."
