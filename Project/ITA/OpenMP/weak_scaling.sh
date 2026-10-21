#!/bin/bash

EXEC="./AreaMC"
OUTPUT_FILE="risultati_weak_scaling_test.csv"

# Definisci il carico di lavoro per SINGOLO thread (es. 5 milioni)
BASE_M=5000000
NUM_RUNS=10

# Inizializza dinamicamente l'intestazione del CSV (Threads,Punti_Totali,Run1,Run2,...,Run10)
HEADER="Threads,Punti_Totali"
for (( i=1; i<=NUM_RUNS; i++ )); do
    HEADER="$HEADER,Run$i"
done
echo "$HEADER" > $OUTPUT_FILE

THREADS=(1 2 4 8 16 32 40 64 80)

echo "==================================================="
echo " Avvio test di Weak Scaling ($NUM_RUNS esecuzioni separate)"
echo " Punti per thread : $BASE_M"
echo " Output           : $OUTPUT_FILE"
echo "==================================================="

export OMP_PROC_BIND=true
export OMP_PLACES=cores

for T in "${THREADS[@]}"; do
    
    # Calcola il numero totale di punti per questo step (T * BASE_M)[cite: 4]
    CURRENT_M=$(($BASE_M * $T))
    
    echo -n "Test con $T thread ($CURRENT_M punti, $NUM_RUNS esecuzioni)... "

    export OMP_NUM_THREADS=$T

    # Inizializza la stringa della riga con il numero di thread e punti totali
    ROW_DATA="$T,$CURRENT_M"

    # Esegue il programma 10 volte e accoda i tempi sulla stessa riga
    for (( i=1; i<=NUM_RUNS; i++ )); do
        
        # Esegue il programma passando CURRENT_M come argomento[cite: 4]
        OUTPUT=$($EXEC $CURRENT_M)
        
        # Estrae il tempo di esecuzione[cite: 4]
        TIME_SEC=$(echo "$OUTPUT" | grep "Tempo di esecuzione" | awk -F ':' '{print $2}' | awk '{print $1}')
        
        # Aggiunge la virgola e il tempo estratto alla riga
        ROW_DATA="$ROW_DATA,$TIME_SEC"
    done
    
    # Scrive l'intera riga nel file CSV
    echo "$ROW_DATA" >> $OUTPUT_FILE
    
    echo "Fatto."
done

echo "==================================================="
echo "Test completato! Dati salvati in $OUTPUT_FILE."