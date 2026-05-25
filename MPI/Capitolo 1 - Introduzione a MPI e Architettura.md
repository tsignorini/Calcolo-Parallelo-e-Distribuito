# Introduzione a MPI e Architettura


## MPI 

> MPI è lo standard utilizzato nel calcolo distribuito sui supercalcolatori. Si basa sul paradigma SPMD (*Single Program Multiple Data*), in cui lo stesso programma viene eseguito simultaneamente da processi distinti su macchine differenti. A differenza di OpenMP, non si basa su direttive ma è una libreria software che offre un controllo a basso livello sulle componenti del sistema, garantendo un'ottimizzazione spinta della computazione. Il suo utilizzo è necessario quando la taglia del problema eccede le capacità di una singola macchina.
> 
> **Vantaggi:**
> * **Scalabilità:** Permette di superare i limiti di memoria della singola macchina, perciò tratta problemi di grossa taglia.
> * **Sicurezza:** Assenza di *data race*, dato che ogni processo ha il proprio spazio di memoria privato.
> 
> **Svantaggi:**
> * **Overhead di rete:** Lo scambio di messaggi e la sincronizzazione avvengono tramite subroutine via rete, introducendo latenze che possono rallentare l'esecuzione.
> * **Errori di comunicazione:** Essendo che i dati vanno inviati, ci si espone a possibili errori di comunicazione.
> * **Complessità:** Il codice diventa sensibilmente più lungo, articolato e difficile da gestire rispetto al paradigma a memoria condivisa.

### Pseudo-Codice


```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;

    // 1. Inizializza l'ambiente MPI, e si passano i parametri perché MPI ne aggiunge altri
    MPI_Init(&argc, &argv);

    // 2. Ottiene il numero totale di processi avviati
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 3. Ottiene l'identificativo (ID) del processo corrente
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Ogni processo stampa questo messaggio autonomamente
    printf("Ciao! Sono il processo %d di un totale di %d\n", rank, size);

    // 4. Termina in modo sicuro l'ambiente MPI
    MPI_Finalize();
    
    return 0;
}
