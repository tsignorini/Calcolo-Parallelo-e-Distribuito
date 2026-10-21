#!/bin/bash

# Parameters
EXEC="./AreaMC 5000000000"
OUTPUT_FILE="risultati_strong_scaling_test.csv"
NUM_RUNS=10

# Sequence of thread counts
THREADS=(1 2 4 8 16 32 40 64 80)

# Dynamically build the CSV header (Threads,Run1,Run2,...,Run10)
HEADER="Threads"
for (( i=1; i<=NUM_RUNS; i++ )); do
    HEADER="$HEADER,Run$i"
done
echo "$HEADER" > $OUTPUT_FILE

echo "==================================================="
echo " Starting Strong Scaling test ($NUM_RUNS runs per column)"
echo " Executable : $EXEC"
echo " Output     : $OUTPUT_FILE"
echo "==================================================="

# Pin each OpenMP thread to a physical core
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Array holding the full CSV row of each thread configuration
declare -a ROW_DATA
for idx in "${!THREADS[@]}"; do
    ROW_DATA[$idx]="${THREADS[$idx]}"
done

# OUTER loop: runs (Run 1, Run 2, ..., Run 10)
# Looping over runs first spreads any load from other users of the machine
# across all thread configurations, instead of hitting just one of them
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "--- Starting Run $run of $NUM_RUNS ---"

    # INNER loop: number of threads (1, 2, 4, ...)
    for idx in "${!THREADS[@]}"; do
        T="${THREADS[$idx]}"
        export OMP_NUM_THREADS=$T

        echo -n "  Test with $T threads... "

        # Run the program
        OUTPUT=$($EXEC)

        # Extract the execution time: take the text after ':' on the
        # "Execution time" line and keep its first field (the number)
        TIME_SEC=$(echo "$OUTPUT" | grep "Execution time" | awk -F ':' '{print $2}' | awk '{print $1}')

        # Append the extracted time to the row of this thread count
        ROW_DATA[$idx]="${ROW_DATA[$idx]},$TIME_SEC"

        echo "$TIME_SEC s"
    done
done

echo "==================================================="
echo "Writing data to the CSV file..."

# Write all the rows stored in the array to the CSV file
for idx in "${!THREADS[@]}"; do
    echo "${ROW_DATA[$idx]}" >> $OUTPUT_FILE
done

echo "Test completed! Data saved in $OUTPUT_FILE."
