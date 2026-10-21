#!/bin/bash

EXEC="./AreaMC"
OUTPUT_FILE="risultati_weak_scaling_test.csv"

# Workload per SINGLE thread (e.g. 5 million points)
BASE_M=5000000
NUM_RUNS=10

# Dynamically build the CSV header (Threads,Punti_Totali,Run1,Run2,...,Run10)
HEADER="Threads,Punti_Totali"
for (( i=1; i<=NUM_RUNS; i++ )); do
    HEADER="$HEADER,Run$i"
done
echo "$HEADER" > $OUTPUT_FILE

THREADS=(1 2 4 8 16 32 40 64 80)

echo "==================================================="
echo " Starting Weak Scaling test ($NUM_RUNS separate runs)"
echo " Points per thread : $BASE_M"
echo " Output            : $OUTPUT_FILE"
echo "==================================================="

# Pin each OpenMP thread to a physical core
export OMP_PROC_BIND=true
export OMP_PLACES=cores

for T in "${THREADS[@]}"; do

    # Compute the total number of points for this step (T * BASE_M)
    CURRENT_M=$(($BASE_M * $T))

    echo -n "Test with $T threads ($CURRENT_M points, $NUM_RUNS runs)... "

    export OMP_NUM_THREADS=$T

    # Initialize the row string with the thread count and the total points
    ROW_DATA="$T,$CURRENT_M"

    # Run the program NUM_RUNS times and append the times to the same row
    for (( i=1; i<=NUM_RUNS; i++ )); do

        # Run the program, passing CURRENT_M as argument
        OUTPUT=$($EXEC $CURRENT_M)

        # Extract the execution time: take the text after ':' on the
        # "Execution time" line and keep its first field (the number)
        TIME_SEC=$(echo "$OUTPUT" | grep "Execution time" | awk -F ':' '{print $2}' | awk '{print $1}')

        # Append a comma and the extracted time to the row
        ROW_DATA="$ROW_DATA,$TIME_SEC"
    done

    # Write the whole row to the CSV file
    echo "$ROW_DATA" >> $OUTPUT_FILE

    echo "Done."
done

echo "==================================================="
echo "Test completed! Data saved in $OUTPUT_FILE."
