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
```


## Comunicazione tra Processi

Per ottimizzare lo scambio di dati nel calcolo distribuito, MPI organizza i processi in **comunicatori** (gruppi di processi), dove ciascun processo riceve un identificativo univoco chiamato `rank`. Il comunicatore globale che include tutti i processi del sistema è `MPI_COMM_WORLD`.

Per migliorare le prestazioni e la portabilità, MPI permette di definire appositi **datatypes**. Questi consentono di descrivere layout di memoria complessi, facilitando il trasferimento rapido dei dati tra i nodi.

La comunicazione punto-a-punto sfrutta inoltre un `tag` (un intero scelto dall'utente) per qualificare il messaggio. Se i criteri di una `MPI_Recv` non coincidono con il tag del messaggio in arrivo, la funzione rimane in attesa, mentre il messaggio non corrispondente viene temporaneamente preservato nel buffer di rete. Spesso si utilizza la wildcard `MPI_ANY_TAG` per accettare messaggi con qualunque identificativo, recuperando successivamente il tag effettivo tramite il parametro di `status`.

Le comunicazioni in MPI si dividono in punto-punto e collettive:

* **Punto-punto:** Coinvolgono coppie di processi. Su supercalcolatori con un alto numero di nodi, utilizzarle per comunicazioni globali comporta lo svantaggio di dover eseguire moltissime chiamate di funzione, generando un grande overhead e problemi di scalabilità.
* **Collettive:** Operano a un livello di astrazione più alto su interi gruppi di processi. Sostituiscono i cicli di comunicazioni punto-punto, ottimizzando intrinsecamente lo scambio di dati e la sincronizzazione tramite l'hardware di rete.

## Comunicazioni Punto-Punto
Le comunicazioni punto-punto sono necessarie per lo scambio di dati tra singoli nodi. Tuttavia quando lo stesso dato deve essere distribuito a più processi — o più in generale quando tutti i processi partecipano a uno scambio coordinato — usare ripetutamente MPI_Send/MPI_Recv risulta poco efficiente: comporta molte chiamate a funzione (overhead elevato), aumenta il rischio di errori nella gestione dei rank, e non sfrutta gli algoritmi ottimizzati (es. broadcast ad albero) che le funzioni collettive implementano internamente.
### MPI_Send

#### Definizione

```c
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

#### Parametri

| Parametro  | Tipo            | Descrizione                                        |
|------------|-----------------|----------------------------------------------------|
| `buf`      | `const void *`  | Puntatore al buffer contenente i dati da inviare   |
| `count`    | `int`           | Numero di elementi da inviare                      |
| `datatype` | `MPI_Datatype`  | Tipo MPI degli elementi (es. `MPI_INT`, `MPI_DOUBLE`) |
| `dest`     | `int`           | `rank_id` del processo ricevente                   |
| `tag`      | `int`           | Etichetta del messaggio (intero non negativo)      |
| `comm`     | `MPI_Comm`      | Comunicatore MPI (es. `MPI_COMM_WORLD`)            |


#### Comportamento bloccante

`MPI_Send` è una funzione **bloccante**: ritorna solo quando il buffer `buf` è **sicuro da riutilizzare**, ovvero quando MPI ha copiato i dati nel proprio sistema interno (buffer di sistema o canale di trasmissione).

> ⚠️ **Attenzione:** il ritorno della funzione **non garantisce** che il messaggio sia stato consegnato al processo ricevente. La consegna può avvenire in modo asincrono, dopo il ritorno.

#### MPI_PROC_NULL
Se `dest` è impostato a `MPI_PROC_NULL`, la chiamata **non ha alcun effetto**: viene trattata come un'operazione nulla e ritorna immediatamente.

### MPI_Recv

#### Definizione

```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```

#### Parametri

| Parametro  | Tipo            | Descrizione                                              |
|------------|-----------------|----------------------------------------------------------|
| `buf`      | `void *`        | Puntatore al buffer in cui verrà scritto il dato ricevuto |
| `count`    | `int`           | Numero **massimo** di elementi che il buffer può contenere |
| `datatype` | `MPI_Datatype`  | Tipo MPI degli elementi (es. `MPI_INT`, `MPI_DOUBLE`)    |
| `source`   | `int`           | `rank_id` del mittente (o `MPI_ANY_SOURCE`)              |
| `tag`      | `int`           | Etichetta del messaggio (o `MPI_ANY_TAG`)                |
| `comm`     | `MPI_Comm`      | Comunicatore MPI (es. `MPI_COMM_WORLD`)                  |
| `status`   | `MPI_Status *`  | Struttura con informazioni sul messaggio ricevuto        |


#### Comportamento bloccante

`MPI_Recv` è una funzione **bloccante**: ritorna solo quando il dato è stato effettivamente scritto nel buffer `buf` ed è pronto per essere usato. Per eseguire il resto del codice la funzione aspetta che il dato venga ricevuto. Questo comportamento risulta utile perché, tipicamente, il dato atteso è necessario per proseguire con la computazione.

> `MPI_Recv` **garantisce** che il dato sia disponibile nel buffer al momento del ritorno.


#### MPI_Status

La struttura `status` contiene informazioni sul messaggio ricevuto:

| Campo              | Descrizione                          |
|--------------------|--------------------------------------|
| `MPI_SOURCE`       | Rank del mittente effettivo          |
| `MPI_TAG`          | Tag del messaggio ricevuto           |
| `MPI_ERROR`        | Codice di errore                     |

Utile soprattutto quando si usano wildcards (`MPI_ANY_SOURCE`, `MPI_ANY_TAG`), per sapere chi ha effettivamente inviato il messaggio e con quale tag.  
Se le informazioni non servono, si può passare `MPI_STATUS_IGNORE`.

#### Count e dimensione del messaggio

`count` specifica la **capienza massima** del buffer, non il numero esatto di elementi attesi.

- Se il messaggio ricevuto contiene **meno elementi** di `count`: nessun errore, il buffer viene riempito parzialmente.
- Se il messaggio ricevuto contiene **più elementi** di `count`: errore di overflow (`MPI_ERR_TRUNCATE`).

Per sapere quanti elementi sono stati **effettivamente ricevuti** si usa `MPI_Get_count()`:

```c
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)
```

Legge dalla struttura `status` il numero di elementi ricevuti e lo scrive in `count`. Va chiamata **dopo** `MPI_Recv`, passando lo stesso `datatype` usato nella ricezione.

```c
int received;
MPI_Get_count(&status, MPI_INT, &received);
```

#### Wildcards

| Costante          | Usata per | Effetto                                        |
|-------------------|-----------|------------------------------------------------|
| `MPI_ANY_SOURCE`  | `source`  | Accetta messaggi da qualsiasi mittente         |
| `MPI_ANY_TAG`     | `tag`     | Accetta messaggi con qualsiasi etichetta       |

### Deadlocks
 
Nell'utilizzo di `MPI_Send` e `MPI_Recv` bisogna prestare particolare attenzione a possibili deadlock. Poiché entrambe le funzioni sono **bloccanti**, se due processi si aspettano a vicenda senza che nessuno dei due proceda, il programma si blocca indefinitamente.
 
#### Scenario classico
 
Il caso più comune si verifica quando due processi vogliono scambiarsi dati e chiamano entrambi `MPI_Send` prima di chiamare `MPI_Recv`:
 
```
Processo 0                  Processo 1
──────────────────          ──────────────────
MPI_Send → P1   ────┐  ┌─── MPI_Send → P0
                    │  │
                (entrambi bloccati: aspettano
                 che l'altro chiami Recv)
                    │  │
MPI_Recv ← P1   ────┘  └─── MPI_Recv ← P0
         (mai raggiunta)      (mai raggiunta)
```
 
`MPI_Send` è bloccante: il processo 0 resta fermo finché il messaggio non viene preso in carico dal sistema MPI. Se anche il processo 1 sta facendo lo stesso contemporaneamente, nessuno dei due chiama mai `MPI_Recv` e il programma si blocca.
 
> ⚠️ Il comportamento dipende dall'implementazione MPI: per messaggi **piccoli** alcuni sistemi bufferizzano internamente il messaggio e la Send ritorna subito, mascherando il deadlock. Per messaggi **grandi** il buffer si esaurisce e il deadlock si manifesta. Non fare affidamento su questo comportamento.
 
#### Soluzione: invertire l'ordine su un processo
 
Basta che uno dei due chiami `MPI_Recv` prima di `MPI_Send`:
 
```
Processo 0                  Processo 1
──────────────────          ──────────────────
MPI_Send → P1   ──────────► MPI_Recv ← P0    ✓
MPI_Recv ← P1   ◄────────── MPI_Send → P0    ✓
```
 
In questo modo il processo 1 è già in ascolto quando il processo 0 invia, e lo scambio avviene senza blocchi.


