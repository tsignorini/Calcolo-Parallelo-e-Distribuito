#!/bin/bash

# Forza il punto come separatore decimale
export LC_ALL=C

# ---------------- Parametri ----------------
EXEC="./AreaMC_MPI"
POINTS="5000000000"
INPUT_FILE="input.txt"
OUTPUT_FILE="risultati_strong_scaling_mpi.csv"
NUM_RUNS=10
PROCS=(1 2 4 8 16 32 40 64 80)

# ---------------- Controlli preliminari ----------------
if [ ! -x "$EXEC" ]; then
    echo "ERRORE: eseguibile '$EXEC' non trovato o non eseguibile."
    echo "Compila con: mpicc -O2 -fopenmp-simd -o AreaMC_MPI_datatypes AreaMC_MPI_datatypes.c"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERRORE: file di input '$INPUT_FILE' non trovato in $(pwd)."
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERRORE: mpirun non trovato nel PATH."
    exit 1
fi

# ---------------- Opzioni mpirun in base all'implementazione ----------------
MPI_VERSION=$(mpirun --version 2>&1)
MPI_OPTS=()

if echo "$MPI_VERSION" | grep -qi "open mpi\|open-mpi"; then
    MPI_OPTS+=(--oversubscribe --use-hwthread-cpus --bind-to core --map-by core)
    # Necessario se lo script gira come root (Docker, WSL, container)
    if [ "$(id -u)" -eq 0 ]; then
        MPI_OPTS+=(--allow-run-as-root)
    fi
else
    echo "ATTENZIONE: MPI non-OpenMPI rilevato, uso mpirun senza opzioni extra."
fi

# ---------------- Scrittura CSV (anche in caso di interruzione) ----------------
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
    echo "Dati salvati in $OUTPUT_FILE."
}
trap write_csv EXIT

echo "==================================================="
echo " Avvio test di Strong Scaling MPI ($NUM_RUNS esecuzioni)"
echo " Eseguibile : $EXEC"
echo " Punti      : $POINTS"
echo " Input      : $INPUT_FILE"
echo " Opzioni MPI: ${MPI_OPTS[*]}"
echo "==================================================="

# ---------------- Ciclo principale ----------------
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "--- Inizio Run $run di $NUM_RUNS ---"

    for idx in "${!PROCS[@]}"; do
        P="${PROCS[$idx]}"
        echo -n "  Test con $P processi... "

        OUTPUT=$(mpirun "${MPI_OPTS[@]}" -n "$P" "$EXEC" "$POINTS" "$INPUT_FILE" 2>&1)

        TIME_SEC=$(echo "$OUTPUT" | grep "Tempo di esecuzione" | awk -F ': ' '{print $2}' | awk '{print $1}')

        if [ -z "$TIME_SEC" ]; then
            echo "ERR"
            echo "    Output di mpirun (prime righe):"
            echo "$OUTPUT" | head -n 8 | sed 's/^/    | /'
            TIME_SEC="ERR"
        else
            echo "$TIME_SEC s"
        fi

        ROW_DATA[$idx]="${ROW_DATA[$idx]},$TIME_SEC"
    done
done

echo "==================================================="
echo "Test completato!"