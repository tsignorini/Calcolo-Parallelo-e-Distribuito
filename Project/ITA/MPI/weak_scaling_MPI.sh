#!/bin/bash

# Forza il punto come separatore decimale
export LC_ALL=C

# ---------------- Parametri del programma MPI ----------------
EXEC="./AreaMC_MPI"
BASE_M=10000000        # Punti PER SINGOLO PROCESSO (Weak Scaling)
INPUT_FILE="input.txt" # File dei vertici di input
OUTPUT_FILE="risultati_weak_scaling_mpi.csv"
NUM_RUNS=10

# Sequenza di processi MPI
PROCS=(1 2 4 8 16 32 40 64 80)

# Soglia stimata dei core fisici disponibili sul nodo (es. 40 su DGX-1)
PHYSICAL_CORES=40

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

# ---------------- Opzioni mpirun di base ----------------
# Nota: --bind-to e --map-by vengono gestiti dinamicamente in base a $P nel ciclo
MPI_OPTS=(--oversubscribe --mca btl ^openib)

# Necessario se lo script viene eseguito come root (es. in container Docker / sistemi particolari)
if [ "$(id -u)" -eq 0 ]; then
    MPI_OPTS+=(--allow-run-as-root)
fi

# ---------------- Inizializzazione CSV ----------------
HEADER="Processi,Punti_Totali"
for (( i=1; i<=NUM_RUNS; i++ )); do
    HEADER="$HEADER,Run$i"
done
echo "$HEADER" > "$OUTPUT_FILE"

echo "==================================================="
echo " Avvio test di Weak Scaling MPI ($NUM_RUNS esecuzioni)"
echo " Eseguibile       : $EXEC"
echo " Punti per nodo   : $BASE_M"
echo " Output           : $OUTPUT_FILE"
echo " Soglia core fis. : $PHYSICAL_CORES"
echo "==================================================="

# Inizializza l'array per memorizzare le righe dei risultati
declare -a ROW_DATA
for idx in "${!PROCS[@]}"; do
    P="${PROCS[$idx]}"
    CURRENT_M=$(( BASE_M * P ))
    ROW_DATA[$idx]="$P,$CURRENT_M"
done

# ---------------- Ciclo principale ----------------
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "--- Inizio Run $run di $NUM_RUNS ---"
    
    for idx in "${!PROCS[@]}"; do
        P="${PROCS[$idx]}"
        CURRENT_M=$(( BASE_M * P ))
        
        echo -n "  Test con $P processi ($CURRENT_M punti)... "
        
        # Gestione dinamica del binding in base al numero di processi
        if [ "$P" -le "$PHYSICAL_CORES" ]; then
            # Entro i core fisici: binding rigido e mappatura per core
            OUTPUT=$(mpirun "${MPI_OPTS[@]}" --bind-to core --map-by core -n "$P" "$EXEC" "$CURRENT_M" "$INPUT_FILE" 2>&1)
        else
            # Oltre i core fisici (over-subscription): disabilitiamo il binding rigido
            OUTPUT=$(mpirun "${MPI_OPTS[@]}" --bind-to none -n "$P" "$EXEC" "$CURRENT_M" "$INPUT_FILE" 2>&1)
        fi
        
        # Estrae il tempo di esecuzione in modo robusto
        TIME_SEC=$(echo "$OUTPUT" | grep "Tempo di esecuzione" | awk -F ': ' '{print $2}' | awk '{print $1}')
        
        if [ -z "$TIME_SEC" ]; then
            echo "ERR"
            echo "    Output di mpirun (prime righe):"
            echo "$OUTPUT" | head -n 8 | sed 's/^/    | /'
            TIME_SEC="ERR"
        else
            echo "$TIME_SEC s"
        fi
        
        # Accoda il tempo estratto alla stringa corrispondente
        ROW_DATA[$idx]="${ROW_DATA[$idx]},$TIME_SEC"
    done
done

echo "==================================================="
echo "Scrittura dei dati nel file CSV in corso..."

# Scrive tutte le righe accumulate nel file CSV
for idx in "${!PROCS[@]}"; do
    echo "${ROW_DATA[$idx]}" >> "$OUTPUT_FILE"
done

echo "Test completato! Dati salvati in $OUTPUT_FILE."