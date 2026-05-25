# MPI Datatypes

## MPI Datatypes

In MPI, qualsiasi operazione di comunicazione (invio o ricezione) richiede la definizione dei dati tramite una specifica tripletta: l'indirizzo di memoria di partenza (`address`), il numero di elementi da trasmettere (`count`) e il tipo di dato (`datatype`). 

Oltre ai tipi predefiniti che mappano i tipi standard del linguaggio C (come `MPI_INT` per gli `int` o `MPI_DOUBLE` per i `double`), MPI offre la potentissima funzionalità dei **Tipi di Dato Derivati (Derived Datatypes)**. Questa caratteristica permette di superare uno dei limiti fisici della memoria: la necessità di avere dati contigui per le comunicazioni di rete. MPI permette infatti di definire tipi di dato personalizzati formati da blocchi spaziati in modo regolare, irregolare o composti da tipi eterogenei (struct).

Per comprendere la necessità vitale di questa astrazione, analizziamo un problema classico del calcolo parallelo: la suddivisione di un dominio bidimensionale.

#### Il Problema: Lo scambio di Colonne (Ghost Cells)
Immaginiamo una matrice bidimensionale suddivisa a blocchi verticali tra vari processi. Ad ogni iterazione, un processo deve inviare la sua colonna più esterna al processo vicino. 
Nel linguaggio C, le matrici sono memorizzate per righe (*row-wise*). Questo significa che, mentre gli elementi di una riga sono fisicamente contigui in memoria, **gli elementi di una colonna non lo sono**. Come possiamo inviare questi dati sparsi in un unico blocco?

Ecco l'evoluzione delle soluzioni, dalla peggiore alla migliore:

**1. La soluzione CATTIVA**  
L'approccio più ingenuo consiste nell'inviare ogni singolo elemento della colonna tramite chiamate individuali alla funzione `MPI_Send` (o `MPI_Isend`). 
*   *Perché è cattiva:* Eseguire decine o centinaia di chiamate di rete per inviare un singolo scalare ciascuna genera un overhead (tempo di latenza/start-up) enorme, uccidendo letteralmente le prestazioni del programma parallelo.

**2. La soluzione BRUTTA**  
Il programmatore cerca di rimediare all'overhead di rete allocando un array temporaneo (buffer). I dati sparsi della colonna vengono letti uno a uno tramite un ciclo `for` e copiati (impacchettati) in questo array temporaneo in modo che diventino contigui. A questo punto si fa un'unica `MPI_Send` del buffer. Il ricevente fa la stessa cosa al contrario: riceve il buffer e lo spacchetta nella colonna di destinazione.
*   *Perché è brutta:* Sebbene riduca le chiamate di rete, questa soluzione spreca memoria (per allocare i buffer temporanei) e cicli di CPU (per copiare e ricopiare manualmente i dati da una zona all'altra della memoria).

**3. La soluzione OTTIMALE**  
La soluzione elegante offerta dallo standard consiste nel **definire un nuovo tipo di dato MPI** che descriva esattamente la forma della colonna in memoria, per poi passarlo direttamente a una singola chiamata `MPI_Send`.
Utilizzando funzioni per la costruzione di Derived Datatypes, come ad esempio `MPI_Type_vector()` (che permette di definire vettori di elementi separati da un passo/stride costante), il programmatore "insegna" a MPI come leggere la colonna.
*   *Perché è la soluzione migliore:* Non ci sono buffer temporanei né spreco di memoria. La libreria MPI, conoscendo ora la struttura esatta dei dati in memoria, si occuperà internamente e in modo ottimizzato di estrarre e trasmettere i byte necessari.

Ecco la sezione dedicata ai Tipi di Dato Derivati, scritta mantenendo lo stile asciutto e di facile consultazione, pronta per essere aggiunta ai tuoi appunti in Markdown:

### I 4 Tipi di Dato Derivati (Derived Datatypes)

La libreria MPI fornisce funzioni specifiche per superare il limite della contiguità in memoria, permettendo di costruire quattro principali categorie di tipi di dato personalizzati a partire da tipi base (o da altri tipi derivati creati in precedenza).

#### 1. Contiguous (Dati Contigui)
Crea un blocco contiguo composto da copie multiple di un tipo di dato esistente.
*   **Firma:** `int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype)`.  
*   **Parametri:**  
    *   `count`: numero di elementi nel blocco.  
    *   `oldtype`: il tipo di dato di base degli elementi.  
    *   `newtype`: puntatore al nuovo tipo di dato creato.  

#### 2. Vector (Vettore a spaziatura regolare)
Definisce un array di blocchi separati da un passo (stride) costante. È ideale per estrarre colonne da matrici memorizzate per righe.
*   **Firma:** `int MPI_Type_vector(int count, int blocklen, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype)`.  
*   **Parametri:**  
    *   `count`: numero totale di blocchi.  
    *   `blocklen`: numero di elementi contenuti in ciascun blocco.  
    *   `stride`: numero di elementi che separano l'inizio di un blocco dall'inizio del successivo.  
    *   `oldtype`: il tipo di dato degli elementi.  

#### 3. Indexed (Spaziatura irregolare)
Consente di definire un insieme di blocchi aventi lunghezze diverse e separati da spaziature (offset) irregolari.
*   **Firma:** `int MPI_Type_indexed(int count, const int array_of_blklen[], const int array_of_displ[], MPI_Datatype oldtype, MPI_Datatype *newtype)`.  
*   **Parametri:**  
    *   `count`: numero totale di blocchi.  
    *   `array_of_blklen`: array contenente la lunghezza specifica di ogni singolo blocco.  
    *   `array_of_displ`: array contenente gli spiazzamenti (misurati in numero di elementi) dell'inizio di ogni blocco rispetto all'inizio della struttura dati.  

#### 4. Struct (Struttura Eterogenea)
È la forma più generale e flessibile. Permette di definire blocchi di dimensioni diverse, separati da spaziature irregolari e, soprattutto, composti da **tipi di dato differenti** (replicando le `struct` del C).
*   **Firma:** `int MPI_Type_create_struct(int count, int *array_of_blklen, MPI_Aint *array_of_displ, MPI_Datatype *array_of_types, MPI_Datatype *newtype)`.  
*   **Parametri chiave (differenze con Indexed):**  
    *   `array_of_displ`: a differenza di Indexed, gli offset qui sono rigorosamente misurati in **byte** e richiedono l'uso del tipo specifico `MPI_Aint`.  
    *   `array_of_types`: array contenente i tipi di dato (`MPI_Datatype`) specifici per ciascun blocco.  


### Inizializzazione e Terminazione

La semplice dichiarazione o creazione strutturale di un nuovo `MPI_Datatype` non è sufficiente per poterlo utilizzare all'interno delle funzioni di comunicazione di rete. Il ciclo di vita del datatype richiede l'uso di due funzioni fondamentali per interfacciarsi col sistema MPI:

1.  **Commit (Inizializzazione): `MPI_Type_commit(...)`**  
    *   **Firma:** `int MPI_Type_commit(MPI_Datatype *datatype)`.  
    *   **Descrizione:** Questa funzione "registra" formalmente il nuovo tipo derivato nel sistema runtime di MPI, compilandone la mappa di memoria. **Deve essere obbligatoriamente invocata** prima di usare il `datatype` in qualsiasi operazione di comunicazione (es. `MPI_Send` o `MPI_Recv`).  

2.  **Free (Terminazione): `MPI_Type_free(...)`**  
    *   **Firma:** `int MPI_Type_free(MPI_Datatype *datatype)`.  
    *   **Descrizione:** Quando il tipo derivato non è più necessario, questa funzione deve essere invocata per deallocare in modo sicuro le risorse di memoria interne che MPI aveva riservato per descrivere l'oggetto, evitando *memory leak*.  


### MPI_Type_contiguous

La funzione `MPI_Type_contiguous()` è il metodo più semplice per creare un nuovo tipo di dato derivato. Essa permette di raggruppare un numero prefissato (`count`) di elementi adiacenti in memoria di un tipo già esistente (`oldtype`) per formare un unico blocco logico contiguo. 

Questo costrutto è perfetto per il linguaggio C, in cui le matrici sono allocate in memoria per righe (*row-major order*), rendendo gli elementi di una singola riga fisicamente adiacenti l'uno all'altro. Invece di inviare i singoli elementi o passare un `count` elevato alla funzione di rete, possiamo definire un datatype che rappresenti esattamente il concetto logico di "riga" e trasmetterla come una singola entità.

#### Esempio: Spedire una riga di una matrice
Supponiamo di avere una matrice quadrata `a` 4x4 composta da `float` e di voler spedire l'intera terza riga (ovvero quella all'indice 2, partendo dall'elemento `a`).

```c
int count = 4; /* Numero di elementi contigui (la lunghezza della riga) */
MPI_Datatype rowtype;

/* 1. Definizione del nuovo tipo formato da 4 MPI_FLOAT contigui */
MPI_Type_contiguous(count, MPI_FLOAT, &rowtype);

/* 2. Registrazione (Commit) del tipo nel sistema MPI prima dell'uso */
MPI_Type_commit(&rowtype);

/* 3. Invio della terza riga (indice 2) */
/* Passiamo l'indirizzo di partenza della riga &a e chiediamo a MPI 
   di inviare esattamente 1 blocco del nuovo tipo 'rowtype' */
MPI_Send(&a[2][0], 1, rowtype, dest, tag, MPI_COMM_WORLD);

/* ... a fine programma ... */
MPI_Type_free(&rowtype);
```

In questo modo, la funzione di invio sa che deve leggere esattamente 4 elementi di tipo `float` a partire dall'indirizzo di memoria `&a` e li trasmetterà in un unico blocco ottimizzato.

Ecco la sezione dedicata a `MPI_Type_vector()`, con la spiegazione dettagliata e l'esempio della matrice tratto direttamente dalle slide, pronta da copiare in formato Markdown:


### MPI_Type_vector

La funzione `MPI_Type_vector()` permette di creare un nuovo tipo di dato costituito da un array di blocchi separati tra loro da una spaziatura regolare (stride). 

Il caso d'uso classico e più importante per questa funzione in linguaggio C è l'**estrazione di una colonna da una matrice**. Poiché in C le matrici sono memorizzate per righe (*row-major order*), gli elementi di una riga sono contigui, ma quelli di una colonna sono separati in memoria da un numero di elementi pari alla larghezza della riga stessa. `MPI_Type_vector()` permette di istruire MPI su come "saltare" da una riga all'altra per leggere solo i dati della colonna desiderata.

#### Parametri Chiave
La firma della funzione è `MPI_Type_vector(count, blocklen, stride, oldtype, &newtype)`:  
*   **`count`**: numero totale di blocchi da leggere.  
*   **`blocklen`**: numero di elementi che compongono un singolo blocco.  
*   **`stride`**: il "passo", ovvero il numero di elementi che separano l'inizio di un blocco dall'inizio del successivo.  

#### Esempio: Spedire una colonna di una matrice
Consideriamo una matrice `a` di tipo `float` e supponiamo di voler estrarre e spedire la **seconda colonna** (quella all'indice 1).

La matrice in memoria ha i seguenti valori contigui:
`[ 1.0, 2.0, 3.0, 4.0 | 5.0, 6.0, 7.0, 8.0 | 9.0, 10.0, 11.0, 12.0 | 13.0, 14.0, 15.0, 16.0 ]`

Vogliamo estrarre la colonna composta dai valori: `2.0`, `6.0`, `10.0`, `14.0`.
Per farlo definiamo la geometria del vettore:
*   `count = 4` (vogliamo 4 elementi in totale, uno per ogni riga).
*   `blocklen = 1` (ogni elemento della colonna è un singolo float).
*   `stride = 4` (per passare da `2.0` a `6.0` dobbiamo fare un salto di 4 posizioni in memoria, pari alla larghezza della riga).

```c
/* Definizione dei parametri per estrarre una singola colonna da una matrice 4x4 */
int count = 4;
int blocklen = 1;
int stride = 4;
MPI_Datatype columntype;

/* 1. Creazione del tipo derivato vettore */
MPI_Type_vector(count, blocklen, stride, MPI_FLOAT, &columntype);

/* 2. Registrazione (Commit) del tipo nel sistema MPI */
MPI_Type_commit(&columntype);

/* 3. Invio della seconda colonna */
/* Passiamo come indirizzo di partenza a, ovvero il valore 2.0.
   Da lì, MPI estrarrà 1 blocco di tipo 'columntype', saltando di 4 in 4 */
MPI_Send(&a[0][1], 1, columntype, dest, tag, MPI_COMM_WORLD);

/* ... a fine programma ... */
MPI_Type_free(&columntype);
```

In questo modo, con una singola istruzione `MPI_Send` e senza dover copiare i dati in buffer temporanei, la libreria estrarrà e invierà esattamente i valori `2.0, 6.0, 10.0, 14.0`.

### MPI_Type_indexed

La funzione `MPI_Type_indexed()` permette di creare un nuovo tipo di dato costituito da un insieme di blocchi di elementi che presentano sia lunghezze irregolari sia spaziature (offset) irregolari tra di loro.

A differenza di `MPI_Type_vector`, in cui la dimensione del blocco e il passo (stride) sono costanti, qui il programmatore deve fornire esplicitamente due array: uno per specificare la lunghezza di ciascun blocco e un altro per indicarne la posizione (lo spiazzamento) rispetto all'inizio del buffer. Tutti gli spiazzamenti si misurano in *numero di elementi* (non in byte) del tipo di dato originario.

#### Parametri Chiave
La firma della funzione è `MPI_Type_indexed(int count, const int array_of_blklen[], const int array_of_displ[], MPI_Datatype oldtype, MPI_Datatype *newtype)`:
*   **`count`**: numero totale di blocchi che compongono il nuovo tipo.
*   **`array_of_blklen`**: array che contiene la lunghezza specifica (numero di elementi) per ogni singolo blocco.
*   **`array_of_displ`**: array che contiene l'offset (in numero di elementi) di ciascun blocco rispetto all'indirizzo di partenza.

#### Esempio: Estrarre blocchi irregolari da un array
Consideriamo un array `a` di 16 `float` con i valori sequenziali da `1.0` a `16.0`. Vogliamo estrarre tre frammenti specifici ed eterogenei:
1. Un primo blocco di **1 elemento** partendo dall'indice **2** (valore `3.0`).
2. Un secondo blocco di **3 elementi** partendo dall'indice **5** (valori `6.0, 7.0, 8.0`).
3. Un terzo blocco di **4 elementi** partendo dall'indice **12** (valori `13.0, 14.0, 15.0, 16.0`).

```c
/* Definizione dei parametri per i 3 blocchi irregolari */
int count = 3;
int blklens[] = {1, 3, 4};  /* Array delle lunghezze dei blocchi */
int displs[] = {2, 5, 12};  /* Array degli indici di partenza */
MPI_Datatype newtype;

/* 1. Creazione del tipo derivato indicizzato */
MPI_Type_indexed(count, blklens, displs, MPI_FLOAT, &newtype);

/* 2. Registrazione (Commit) del tipo nel sistema MPI */
MPI_Type_commit(&newtype);

/* 3. Invio dei dati */
/* Passando l'indirizzo di partenza &a, MPI applicherà gli spiazzamenti in displs[]
   ed estrarrà esattamente i 3 blocchi descritti in blklens[] */
MPI_Send(&a[0], 1, newtype, dest, tag, MPI_COMM_WORLD);

/* Ricezione lato destinatario (il processo riceverà 8 elementi contigui in totale) */
/* MPI_Recv(&b[0], 8, MPI_FLOAT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); */

/* ... a fine programma ... */
MPI_Type_free(&newtype);
```

In questo esempio, con una singola istruzione `MPI_Send` in cui indichiamo di inviare `1` solo blocco logico di tipo `newtype`, la libreria navigherà l'array originale ed estrarrà per la trasmissione esattamente gli 8 valori desiderati (`3.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0`). Il destinatario riceverà questi 8 valori "compattati" nel suo array di ricezione.


### Combinare i Tipi Derivati (Componibilità)

Una delle caratteristiche più potenti dei *Derived Datatypes* di MPI è la loro totale componibilità. Il parametro `oldtype` richiesto dalle funzioni di creazione (come `MPI_Type_contiguous`, `MPI_Type_vector` e `MPI_Type_indexed`) non deve necessariamente essere un tipo predefinito (es. `MPI_FLOAT` o `MPI_INT`), ma può essere **un altro tipo derivato creato in precedenza dal programmatore**. 

Questo approccio a "scatole cinesi" permette di mappare e trasmettere strutture dati gerarchiche e complesse con estrema eleganza.

#### Esempio: Un vettore di vettori
In questo esempio, creiamo prima un vettore logico (`vec`) composto da elementi base `MPI_FLOAT`, e poi lo utilizziamo a sua volta come mattoncino base (`oldtype`) per creare un tipo ancora più complesso (`vecvec`).

```c
int count, blocklen, stride; 
MPI_Datatype vec, vecvec;

/* 1. Creiamo il primo tipo derivato (un vettore base di float) */
count = 2; blocklen = 2; stride = 3;
MPI_Type_vector(count, blocklen, stride, MPI_FLOAT, &vec);
MPI_Type_commit(&vec);

/* 2. Usiamo 'vec' appena creato come 'oldtype' per un nuovo vettore */
count = 2; blocklen = 1; stride = 3;
MPI_Type_vector(count, blocklen, stride, vec, &vecvec);
MPI_Type_commit(&vecvec);

/* A fine utilizzo, ricordarsi di liberare entrambi i tipi */
/* MPI_Type_free(&vec); */
/* MPI_Type_free(&vecvec); */
```

### MPI_Type_create_struct

La funzione `MPI_Type_create_struct()` è la forma più generale e flessibile per la creazione di tipi di dato derivati. Permette di definire un nuovo tipo composto da un insieme di blocchi spaziati in modo irregolare e, soprattutto, costituiti da **tipi di dato esistenti differenti tra loro**.

Questo costrutto è fondamentale perché mappa direttamente le **`struct`** del linguaggio C in MPI. Consente infatti di trasmettere un'intera struttura dati eterogenea attraverso la rete in un'unica operazione di comunicazione ottimizzata, evitando di dover spacchettare i campi o inviare messaggi multipli.

#### Parametri Chiave
A differenza delle altre funzioni (come `MPI_Type_indexed`), in cui le spaziature si misurano in "numero di elementi", qui i blocchi hanno tipi base di dimensioni diverse. Di conseguenza, tutti gli offset devono essere rigorosamente misurati in **byte**. Per farlo si utilizza un tipo di dato intero speciale offerto da MPI: `MPI_Aint`.

La firma della funzione è `int MPI_Type_create_struct(int count, int *array_of_blklen, MPI_Aint *array_of_displ, MPI_Datatype *array_of_types, MPI_Datatype *newtype)`:
*   **`count`**: numero totale di blocchi che compongono la struttura.
*   **`array_of_blklen`**: array contenente il numero di elementi per ciascun singolo blocco.
*   **`array_of_displ`**: array di tipo `MPI_Aint` contenente lo spiazzamento (offset) **in byte** di ciascun blocco rispetto all'indirizzo di partenza.
*   **`array_of_types`**: array contenente i tipi di dato (`MPI_Datatype`) specifici per ciascun blocco.

#### Esempio: Spedire una struct C eterogenea
Consideriamo una struttura `particle_t` utilizzata per rappresentare una particella in una simulazione, composta da coordinate/velocità (4 `float`) e da metadati interi (2 `int`):

```c
typedef struct {
    float x, y, z, v;
    int n, t;
} particle_t;
```

Per insegnare a MPI come mappare questa struct in memoria, dobbiamo descriverla come formata da 2 blocchi distinti,:
1. Un primo blocco di **4 elementi** di tipo `MPI_FLOAT` con offset 0 byte.
2. Un secondo blocco di **2 elementi** di tipo `MPI_INT` il cui offset in byte inizierà esattamente dove finiscono i 4 float.

```c
/* Parametri per descrivere la geometria della struct */
int count = 2;
int blklens[2];
MPI_Aint displs[2], lb, extent;
MPI_Datatype oldtypes[2], newtype;

/* Configurazione del 1° blocco: 4 float (x, y, z, v) */
oldtypes[0] = MPI_FLOAT;
blklens[0] = 4;
displs[0] = 0; /* Il primo blocco parte dall'inizio (offset 0) */

/* Chiediamo a MPI di calcolare l'ingombro (extent) in byte del tipo float */
MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);

/* Configurazione del 2° blocco: 2 int (n, t) */
oldtypes[1] = MPI_INT;
blklens[1] = 2;
/* L'offset in byte del secondo blocco è pari all'ingombro di 4 float */
displs[1] = 4 * extent;

/* 1. Creazione del tipo derivato struct */
MPI_Type_create_struct(count, blklens, displs, oldtypes, &newtype);

/* 2. Registrazione (Commit) del tipo nel sistema MPI */
MPI_Type_commit(&newtype);

/* 3. Utilizzo per la spedizione */
/* Ora la struct può essere spedita interamente con una singola istruzione */
/* particle_t my_particle; */
/* MPI_Send(&my_particle, 1, newtype, dest, tag, MPI_COMM_WORLD); */

/* ... a fine programma ... */
MPI_Type_free(&newtype);
```

### Conclusione: Ottimizzazione e considerazioni finali su MPI

Per chiudere l'argomento sui **Derived Datatypes** e fare un bilancio finale sulla programmazione MPI, possiamo riassumere i concetti chiave in due grandi aspetti: l'ottimizzazione della memoria e le sfide generali del *message passing*.

#### Il ruolo cruciale dei Tipi Derivati
I *Derived Datatypes* (Contiguous, Vector, Indexed e Struct) rappresentano lo strumento "ottimale" messo a disposizione dallo standard MPI per scambiare dati non contigui in memoria. 
Sfruttandoli, il programmatore riesce a:
1. **Evitare l'overhead di rete:** Scongiurando la soluzione "cattiva", ovvero l'invio di decine di messaggi minuscoli per ogni singolo dato frammentato.
2. **Risparmiare CPU e Memoria:** Evitando la soluzione "brutta", ovvero lo spreco di cicli macchina necessari per copiare e impacchettare (pack/unpack) manualmente i dati in array temporanei prima dell'invio.
3. **Mappare strutture complesse:** Creando gerarchie di tipi componibili che ricalcano fedelmente le strutture dati eterogenee del linguaggio C.

## Considerazioni finali sul modello MPI
Allargando lo sguardo all'intero ecosistema MPI, è importante riconoscere la natura di questo standard. Il *message passing* è un modello di programmazione fondamentale, ma di **bassissimo livello e decisamente "pesante" (heavy weight)**.

Scrivere programmi MPI richiede uno sforzo notevole da parte dello sviluppatore, e presenta sfide precise:
* **Codice prolisso:** Gran parte del costo di sviluppo deriva dalla quantità di codice locale necessario per gestire la logica di rete. Molto spesso, il codice dedicato unicamente alla comunicazione supera la metà delle righe totali del programma.
* **Difficoltà di progettazione:** Progettare un'applicazione MPI che sia adattabile, flessibile e del tutto priva di errori (come i temuti *deadlock*) è estremamente complesso (in inglese: *tough to get right*).

Tuttavia, nonostante queste intrinseche difficoltà di sviluppo, MPI rimane il **modello di programmazione d'elezione in assoluto per la scalabilità**. La sua immensa diffusione e adozione globale sono garantite dalla sua **portabilità** tra sistemi diversi. Imparare a padroneggiare strumenti come le *comunicazioni collettive* e i *Tipi Derivati* è il prezzo da pagare per poter estrarre fino all'ultimo *flop* di potenza di calcolo dai moderni supercalcolatori distribuiti.