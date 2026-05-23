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

### MPI_Sendrecv — Scambio simultaneo e sicuro

Un'alternativa più robusta per evitare i deadlock quando due processi devono scambiarsi dati contemporaneamente (come nel caso dello scambio di *ghost cells*) è l'utilizzo di `MPI_Sendrecv`. Questa funzione esegue un'operazione di invio e una di ricezione in un'unica chiamata, demandando a MPI la gestione interna delle precedenze e dei buffer per scongiurare qualsiasi blocco.

#### Definizione

```c
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status)
```

I parametri sono semplicemente la concatenazione di quelli di una `MPI_Send` e di una `MPI_Recv`.

#### Parametri

| Parametro | Tipo | Descrizione |
|---|---|---|
| `sendbuf` | `const void *` | Puntatore al buffer dei dati da inviare |
| `sendcount` | `int` | Numero di elementi da inviare |
| `sendtype` | `MPI_Datatype` | Tipo MPI degli elementi da inviare |
| `dest` | `int` | Rank del processo ricevente |
| `sendtag` | `int` | Etichetta del messaggio in uscita |
| `recvbuf` | `void *` | Puntatore al buffer in cui salvare i dati in ingresso |
| `recvcount` | `int` | Numero massimo di elementi da ricevere |
| `recvtype` | `MPI_Datatype` | Tipo MPI degli elementi da ricevere |
| `source` | `int` | Rank del processo mittente |
| `recvtag` | `int` | Etichetta del messaggio in ingresso |
| `comm` | `MPI_Comm` | Comunicatore MPI |
| `status` | `MPI_Status *` | Status della ricezione (o `MPI_STATUS_IGNORE`) |

#### Perché utilizzarla?

* **Zero Deadlock:** I processi possono richiamarla simultaneamente senza preoccuparsi dell'ordine di esecuzione, perché MPI gestisce la comunicazione bidirezionale in modo sicuro.
* **Codice più pulito:** Sostituisce i blocchi condizionali (es. `if (rank % 2 == 0) { invia; ricevi; } else { ricevi; invia; }`) rendendo il codice molto più leggibile, compatto e meno prono a errori umani.

> ⚠️ **Attenzione ai buffer:** I buffer `sendbuf` e `recvbuf` devono essere **disgiunti** in memoria. Non è possibile inviare e ricevere dati sovrascrivendo la stessa area di memoria (se serve questo comportamento, esiste una variante apposita chiamata `MPI_Sendrecv_replace`).

### MPI_Isend e MPI_Irecv — Comunicazioni non bloccanti

`MPI_Isend` e `MPI_Irecv` sono le versioni **non bloccanti** di `MPI_Send` e `MPI_Recv`: avviano l'operazione di comunicazione e ritornano **immediatamente**, senza aspettarne il completamento.

#### Definizioni

```c
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request)

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request)
```

Il parametro aggiuntivo rispetto alle versioni bloccanti è `MPI_Request *request`: un handle che identifica l'operazione in corso e viene usato in seguito per verificarne il completamento.

#### Completamento: MPI_Wait e MPI_Test

Poiché le funzioni ritornano prima che la comunicazione sia conclusa, è necessario sincronizzarsi esplicitamente prima di accedere al buffer:

```c
// Blocca finché l'operazione non è completata
MPI_Wait(MPI_Request *request, MPI_Status *status)

// Non bloccante: controlla se l'operazione è completata (flag = 1) o no (flag = 0)
MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
```

> ⚠️ **Regola fondamentale:** il buffer `buf` **non deve essere letto né modificato** tra la chiamata a `MPI_Isend`/`MPI_Irecv` e il completamento confermato da `MPI_Wait` o `MPI_Test`. Farlo è undefined behavior.

#### Pattern d'uso tipico

```c
MPI_Request req;

MPI_Isend(buf, count, MPI_INT, dest, tag, MPI_COMM_WORLD, &req);

// computazione indipendente dalla comunicazione...
do_work();

MPI_Wait(&req, MPI_STATUS_IGNORE); // attende il completamento
```

In questo modo la comunicazione avviene **in parallelo** con `do_work()`, nascondendo la latenza di rete.

#### Confronto bloccante vs non bloccante

```
MPI_Send (bloccante)
────────────────────────────────────────────────────
[  Send (attesa)  ][  computazione  ]

MPI_Isend (non bloccante)
────────────────────────────────────────────────────
[Isend][ computazione  ][Wait]
          ↑
    comunicazione in corso in background
```

| Aspetto               | MPI_Send / MPI_Recv       | MPI_Isend / MPI_Irecv              |
|-----------------------|---------------------------|------------------------------------|
| Ritorno               | Solo a completamento      | Immediato                          |
| Buffer dopo chiamata  | Subito riutilizzabile     | Non toccare prima di MPI_Wait      |
| Rischio deadlock      | Alto (ordine critico)     | Basso (non si bloccano a vicenda)  |
| Overlap comm/compute  | ✗                         | ✓                                  |
| Complessità del codice| Bassa                     | Più alta                           |

#### Perché MPI_Isend è molto utile

Nelle applicazioni HPC il tempo speso in comunicazione è spesso il collo di bottiglia. `MPI_Isend` permette di **sovrapporre comunicazione e computazione**: mentre i dati viaggiano sulla rete, il processo può già lavorare sulla parte di computazione che non dipende da quei dati. Questo pattern, detto *latency hiding*, può portare a speedup significativi rispetto all'alternativa bloccante, soprattutto in applicazioni con molti scambi tra processi.

Inoltre, poiché `MPI_Isend` e `MPI_Irecv` non si bloccano, due processi possono chiamarle entrambi nello stesso ordine **senza rischio di deadlock**, semplificando la gestione della sincronizzazione rispetto alle versioni bloccanti.

### MPI_Abort

La funzione `MPI_Abort()` viene utilizzata nell'ambito della programmazione parallela con MPI (Message Passing Interface) per forzare la terminazione anomala di tutti i processi appartenenti a un determinato comunicatore (generalmente `MPI_COMM_WORLD`). 
Il caso d'uso tipico si verifica quando **un errore critico e irrecuperabile intercorre su uno solo dei nodi** o processi della griglia computazionale. In un'applicazione parallela, i nodi sono strettamente interconnessi tramite operazioni di comunicazione bloccanti (come `MPI_Send`, `MPI_Recv` o barriere di sincronizzazione come `MPI_Barrier`). Se un singolo nodo dovesse fallire o interrompere la sua esecuzione silenziosamente senza notificare gli altri, l'intera applicazione entrerebbe in uno stato di **deadlock** (blocco indefinito), continuando a consumare risorse del cluster senza produrre alcun risultato. Invocando `MPI_Abort(MPI_COMM_WORLD, error_code)`, il nodo che ha riscontrato l'anomalia ordina all'ambiente di runtime di MPI di terminare immediatamente e in modo pulito tutti gli altri processi, restituendo un codice d'errore (`error_code`) utile per il debugging.
L'uso di una normale funzione di sistema come `exit()` terminerebbe unicamente il processo locale in cui si è verificato l'errore. Essendo i nodi distribuiti su un'infrastruttura di rete, propagare in modo affidabile e automatico questo stato di uscita a tutti gli altri nodi risulterebbe estremamente difficile. Di conseguenza, il nodo guasto si spegnerebbe in isolamento, mentre il resto del cluster continuerebbe l'esecuzione rimanendo fatalmente bloccato in attesa di comunicazioni (deadlock).

## Le Comunicazioni Collettive in MPI

Nel calcolo parallelo con **MPI (Message Passing Interface)**, le comunicazioni collettive (come `MPI_Bcast`, `MPI_Reduce` o `MPI_Gather`) coordinano lo scambio di dati tra tutti i processi di un comunicatore in un'unica operazione, offrendo vantaggi prestazionali netti rispetto alle comunicazioni punto-punto (`MPI_Send`/`MPI_Recv`). Saperle usare correttamente è quindi un requisito chiave per un programmatore nell'ambito del calcolo distribuito.

### Perché le Collettive sono più efficienti delle Punto-Punto:

1. **Complessità Algoritmica $O(\log N)$**: Se un processo master invia un dato a $N$ nodi tramite punto-punto, impiega un tempo lineare $O(N)$. Le funzioni collettive utilizzano invece algoritmi ad albero (es. alberi binomiali) in cui i nodi intermedi ridistribuiscono il messaggio, abbattendo la complexity a livello logaritmico $O(\log N)$.
2. **Ottimizzazione Topologica (Hardware Awareness)**: Le librerie MPI riconoscono l'architettura sottostante. Sfruttano la memoria condivisa per i processi sullo stesso nodo e il multicast hardware (es. InfiniBand) per il traffico di rete, ottimizzando il percorso dei dati in modo trasparente all'utente.
3. **Riduzione dell'Overhead e dei Deadlock**: Gestire manualmente decine di comunicazioni punto-punto aumenta il rischio di colli di bottiglia sul master e di blocchi reciproci (*deadlock*). Le collettive offrono una sincronizzazione implicita e ottimizzata, riducendo l'overhead di controllo e semplificando la struttura del codice.

### MPI_Barrier
La funzione `MPI_Barrier(comm)` è un'operazione di sincronizzazione collettiva che agisce su un gruppo di processi associati a uno specifico comunicatore, come ad esempio `MPI_COMM_WORLD`. Quando un processo raggiunge ed esegue questa istruzione, si blocca e attende finché tutti gli altri processi appartenenti al gruppo non hanno raggiunto a loro volta la medesima chiamata. Solo quando tutti i processi sono arrivati alla barriera, questi vengono sbloccati e sono liberi di proseguire la loro esecuzione.

#### Vantaggi
* **Gestione sicura delle fasi:** Rappresenta un modo estremamente semplice e diretto per separare due fasi distinte di una computazione parallela, garantendo che i messaggi generati in una fase non interferiscano con quelli della fase successiva.

#### Svantaggi
* **Overhead prestazionale:** Trattandosi di un'operazione globale che richiede la partecipazione e l'attesa di tutti i processi del comunicatore, può risultare molto dispendiosa in termini di tempo e rallentare significativamente l'esecuzione del programma.
* **Spesso evitabile:** In molti casi, l'invocazione di `MPI_Barrier()` dovrebbe essere evitata, poiché la sincronizzazione tra i processi può essere ottenuta in modo più efficiente strutturando correttamente l'indirizzamento esplicito delle comunicazioni (ad esempio sfruttando i `tag`, l'identificativo del mittente `source` o l'isolamento garantito da comunicatori separati).

### MPI_Bcast()
La funzione `MPI_Bcast()` (Broadcast) permette a un singolo processo, definito "root", di inviare una copia esatta degli stessi dati a tutti gli altri processi appartenenti a un determinato gruppo o comunicatore. Essendo un'operazione collettiva, la funzione deve essere obbligatoriamente invocata da tutti i processi del gruppo.

#### Argomenti della funzione
La sintassi di base è `MPI_Bcast(buffer, count, datatype, root, comm)`:
* **`buffer`**: puntatore all'area di memoria dei dati. Per il processo `root` rappresenta l'indirizzo da cui leggere i dati da inviare; per tutti gli altri processi rappresenta l'indirizzo in cui scrivere i dati ricevuti.
* **`count`**: numero di elementi che compongono il messaggio.
* **`datatype`**: tipo di dato degli elementi trasmessi (es. `MPI_INT`).
* **`root`**: l'identificativo (rank) del processo mittente che possiede i dati iniziali.
* **`comm`**: il comunicatore di riferimento (es. `MPI_COMM_WORLD`).

#### Schema di funzionamento
Ecco un esempio di come si propaga il buffer se il processo sorgente (`root`) è il *Proc 1* e deve inviare 3 elementi (identificati come 1, 2, 3) a un totale di 4 processi.

**Prima della chiamata a `MPI_Bcast()`**  
Proc 0: `[   |   |   ]`  
Proc 1: `[ 1 | 2 | 3 ]`  <-- ROOT  
Proc 2: `[   |   |   ]`  
Proc 3: `[   |   |   ]`  

**Dopo la chiamata a `MPI_Bcast()`**  
Proc 0: `[ 1 | 2 | 3 ]`    
Proc 1: `[ 1 | 2 | 3 ]`  
Proc 2: `[ 1 | 2 | 3 ]`  
Proc 3: `[ 1 | 2 | 3 ]`  



