# Appunti Seconda Parte IN480

---

#

## MPI 

> MPI è lo standard utilizzato nel calcolo distribuito sui supercalcolatori. Si basa sul paradigma SPMD (*Single Program Multiple Data*), in cui lo stesso programma viene eseguito simultaneamente da processi distinti su macchine differenti. A differenza di OpenMP, non si basa su direttive ma è una libreria software che offre un controllo a basso livello sulle componenti del sistema, garantendo un'ottimizzazione spinta della computazione. Il suo utilizzo è necessario quando la taglia del problema eccede le capacità di una singola macchina.
> 
> **Vantaggi:**
> * **Scalabilità:** Permette di superare i limiti di memoria della singola macchina, perciò tratta problemi di grossa taglia.
> * **Sicurezza:** Assenza di *data race*, dato che ogni processo ha il proprio spazio di memoria privato.
> 
> **Svantaggi:**
> * **Overhead di rete:** Lo scambio di messaggi e la sincronizzazione avvengono tramite subroutine via rete, introducendo latenze che possono rallentare l'esecuzione.
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
```


## Comunicazione tra Processi

Per ottimizzare lo scambio di dati nel calcolo distribuito, MPI organizza i processi in **comunicatori** (gruppi di processi), dove ciascun processo riceve un identificativo univoco chiamato `rank`. Il comunicatore globale che include tutti i processi del sistema è `MPI_COMM_WORLD`.

Per migliorare le prestazioni e la portabilità, MPI permette di definire appositi **datatypes**. Questi consentono di descrivere layout di memoria complessi, facilitando il trasferimento rapido dei dati tra i nodi.

La comunicazione punto-a-punto sfrutta inoltre un `tag` (un intero scelto dall'utente) per qualificare il messaggio. Se i criteri di una `MPI_Recv` non coincidono con il tag del messaggio in arrivo, la funzione rimane in attesa, mentre il messaggio non corrispondente viene temporaneamente preservato nel buffer di rete. Spesso si utilizza la wildcard `MPI_ANY_TAG` per accettare messaggi con qualunque identificativo, recuperando successivamente il tag effettivo tramite il parametro di `status`.
