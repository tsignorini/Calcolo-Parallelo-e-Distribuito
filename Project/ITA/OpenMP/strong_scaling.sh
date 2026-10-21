#!/bin/bash

# Parametri
EXEC="./AreaMC 5000000000"
OUTPUT_FILE="risultati_strong_scaling_test.csv"
NUM_RUNS=10

# Sequenza di thread
THREADS=(1 2 4 8 16 32 40 64 80)

# Inizializza dinamicamente l'intestazione del CSV (Threads,Run1,Run2,...,Run10)
HEADER="Threads"
for (( i=1; i<=NUM_RUNS; i++ )); do
    HEADER="$HEADER,Run$i"
done
echo "$HEADER" > $OUTPUT_FILE

echo "==================================================="
echo " Avvio test di Strong Scaling ($NUM_RUNS esecuzioni per colonna)"
echo " Eseguibile : $EXEC"
echo " Output     : $OUTPUT_FILE"
echo "==================================================="

export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Inizializza un array per memorizzare l'intera riga di ogni configurazione di thread
declare -a ROW_DATA
for idx in "${!THREADS[@]}"; do
    ROW_DATA[$idx]="${THREADS[$idx]}"
done

# Ciclo ESTERNO: Esecuzioni (Run 1, Run 2, ..., Run 10)
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "--- Inizio Run $run di $NUM_RUNS ---"
    
    # Ciclo INTERNO: Numero di Thread (1, 2, 4, ...)
    for idx in "${!THREADS[@]}"; do
        T="${THREADS[$idx]}"
        export OMP_NUM_THREADS=$T
        
        echo -n "  Test con $T thread... "
        
        # Esegue il programma
        OUTPUT=$($EXEC)
        
        # Estrae il tempo[cite: 5]
        TIME_SEC=$(echo "$OUTPUT" | grep "Tempo di esecuzione" | awk '{print $5}')
        
        # Accoda il tempo estratto alla stringa corrispondente a questo thread
        ROW_DATA[$idx]="${ROW_DATA[$idx]},$TIME_SEC"
        
        echo "$TIME_SEC s"
    done
done

echo "==================================================="
echo "Scrittura dei dati nel file CSV in corso..."

# Scrive tutte le righe memorizzate nell'array all'interno del file CSV
for idx in "${!THREADS[@]}"; do
    echo "${ROW_DATA[$idx]}" >> $OUTPUT_FILE
done

echo "Test completato! Dati salvati in $OUTPUT_FILE."