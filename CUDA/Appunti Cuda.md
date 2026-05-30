# Introduzione al Calcolo Parallelo su GPU e CUDA

Il calcolo ad alte prestazioni ha subito una vera e propria rivoluzione con l'introduzione delle GPU (Graphics Processing Unit) per scopi di calcolo generale. Mentre le CPU tradizionali sono progettate e ottimizzate per eseguire il più velocemente possibile singole sequenze di istruzioni (ottimizzazione della latenza) su pochi core, le GPU nascono per massimizzare il *throughput*, eseguendo simultaneamente migliaia di thread in parallelo per elaborare enormi moli di dati. 

Per programmare e sfruttare questo immenso parallelismo, sono nati modelli di programmazione specifici. Il più celebre e utilizzato è **CUDA** (Compute Unified Device Architecture), una piattaforma e un modello di programmazione introdotto da NVIDIA nel 2006.

## Il Modello Eterogeneo: Host e Device

La programmazione su GPU tramite CUDA si basa su un modello di calcolo "eterogeneo", nel quale due entità fisicamente separate lavorano in stretta sinergia: l'**Host** e il **Device**.

*   **Host**: Rappresenta la CPU (Central Processing Unit) e la sua memoria di sistema (la RAM del computer). L'Host funge da "direttore d'orchestra": esegue tutto il codice sequenziale tradizionale, gestisce le interazioni di I/O e decide quando è il momento di delegare le operazioni alla scheda video.
*   **Device**: Rappresenta la GPU e la sua memoria dedicata (VRAM). Lavora come un coprocessore ad altissime prestazioni. Quando riceve un comando dall'Host, il Device esegue la parte di codice parallelo (che in CUDA viene chiamata *kernel*) utilizzando le sue migliaia di core.

Il ciclo vitale di base di un'applicazione CUDA prevede che l'Host allochi memoria sul Device, vi copi i dati da elaborare, lanci in esecuzione il *kernel*, e infine attenda e ricopi i risultati elaborati dalla GPU per riportarli nella memoria della CPU.

## CUDA vs OpenCL: Pregi, Difetti e Diffusione

Nel panorama della programmazione su GPU, l'alternativa principale a CUDA è rappresentata da **OpenCL**, uno standard aperto sviluppato inizialmente da un consorzio di aziende (tra cui Apple, AMD, Intel e la stessa NVIDIA). 

Nonostante l'attrattiva di uno standard aperto, **CUDA gode oggi di un'adozione e una popolarità nettamente superiori nell'industria e nella ricerca**. 

### Perché CUDA domina su OpenCL?
Il motivo principale del predominio di CUDA è da ricercarsi nel miglior compromesso tra prestazioni e comodità per gli sviluppatori. CUDA C/C++ aggiunge solo un piccolo set di estensioni sintattiche al linguaggio C/C++ standard, mantenendo un'interfaccia di programmazione intuitiva e pulita. 
Al contrario, OpenCL è noto per avere una curva di apprendimento notevolmente più ripida: anche scrivere il programma più basilare richiede una comprensione complessa dell'architettura e la scrittura di molto codice accessorio ("boilerplate"). Inoltre, la promessa di portabilità di OpenCL è stata in parte frenata dal fatto che alcune sue implementazioni si sono rivelate meno efficienti (generando eseguibili anche fino al 30% più lenti rispetto all'equivalente codice CUDA) e talvolta affette da difetti o bug di sviluppo.

### Confronto riassuntivo

**CUDA**
*   **Pregi:** 
    *   Sintassi semplice, naturale estensione dello standard C/C++.
    *   Prestazioni eccellenti e garantite sull'hardware di riferimento.
    *   Ecosistema maturo con una suite ineguagliabile di librerie altamente ottimizzate (come cuBLAS per l'algebra lineare).
*   **Difetti:** 
    *   *Vendor lock-in*: essendo proprietario, un programma CUDA può girare unicamente su macchine dotate di GPU NVIDIA.

**OpenCL**
*   **Pregi:** 
    *   Portabilità e scalabilità estreme: essendo uno standard aperto, il medesimo codice può essere eseguito non solo su GPU di produttori diversi (NVIDIA, AMD, Intel), ma anche su CPU multi-core.
    *   Nessun vincolo tecnologico verso un singolo fornitore hardware (*no vendor lock-in*).
*   **Difetti:** 
    *   Codice sorgente estremamente prolisso (*verbose*) e oggettivamente complesso da apprendere e mantenere per un principiante.
    *   Adozione rallentata dal supporto incostante dei vari vendor; le prestazioni possono variare imprevedibilmente passando da un hardware all'altro.


## Il Flusso di Elaborazione (Processing Flow) su GPU

L'esecuzione di un programma su architettura CUDA prevede una stretta interazione tra la CPU (Host) e la GPU (Device), le quali possiedono spazi di memoria fisicamente separati. L'intero ciclo di vita di un'elaborazione su GPU si può riassumere in un flusso logico (Simple Processing Flow) composto da tre fasi principali:

### 1. Trasferimento dei dati in ingresso (Host to Device)
Poiché la GPU opera sui propri banchi di memoria, il primo passo spetta all'Host. La CPU alloca lo spazio necessario nella memoria globale della GPU (generalmente utilizzando la funzione `cudaMalloc`) e vi copia i dati da elaborare. Questo trasferimento fisico dei dati dalla memoria di sistema alla memoria del Device avviene viaggiando attraverso il bus PCI Express, utilizzando il comando `cudaMemcpy` e specificando la direzione `cudaMemcpyHostToDevice`.

### 2. Esecuzione del programma GPU (Kernel)
Una volta che i dati sono presenti e pronti sulla scheda video, la CPU carica e lancia l'esecuzione del programma parallelo sulla GPU, noto come *kernel*. Durante questa fase:
* L'hardware della GPU prende il controllo e suddivide il lavoro su migliaia di thread.
* Per massimizzare le prestazioni ed evitare continui e lenti accessi alla memoria globale, è bene ottimizzare l'accesso ai dati all'interno del chip, sfruttando memorie ultra-veloci come i registri privati e la memoria condivisa (Shared Memory).
* Il lancio del kernel è **asincrono**: il controllo ritorna immediatamente alla CPU, che può proseguire con altre operazioni oppure mettersi in attesa del completamento del kernel richiamando `cudaDeviceSynchronize()`.

### 3. Recupero dei risultati (Device to Host)
Al termine dell'elaborazione, i dati risultanti risiedono ancora nella memoria del Device. Affinché il programma principale possa utilizzarli, salvarli o stamparli, la CPU deve inviare un nuovo comando di copia. Si utilizza nuovamente `cudaMemcpy`, ma con direzione inversa (`cudaMemcpyDeviceToHost`), per copiare i risultati dalla memoria della GPU a quella della CPU, transitando sempre sul bus PCIe.

Come operazione finale di pulizia (Cleanup), la memoria precedentemente allocata sulla GPU deve essere liberata invocando la funzione `cudaFree()`.


# Anatomia di una GPU e Gerarchia di Memoria

A differenza di una CPU tradizionale, progettata per eseguire poche istruzioni complesse con bassissima latenza, una GPU è costruita con l'obiettivo di massimizzare il *throughput* parallelo. Per ottenere questo risultato, la GPU sacrifica complesse unità di controllo (come il *branch predictor* o l'esecuzione fuori ordine) in favore di un numero enorme di unità di calcolo.

## Componenti Principali

L'architettura hardware di una GPU (in particolare nel mondo NVIDIA/CUDA) si basa su una struttura altamente gerarchica:

*   **Streaming Multiprocessor (SM):** È il cuore computazionale della GPU (spesso chiamato *Compute Unit* in terminologia OpenCL). Una GPU è composta da un array (una griglia) di numerosi SM.
*   **Streaming Processor (SP) o CUDA Core:** All'interno di ogni singolo SM si trovano dozzine o centinaia di piccoli processori (ALU). Questi core eseguono materialmente i calcoli matematici.
*   **Warp Scheduler e Unità di Controllo:** I core all'interno di un SM non sono totalmente indipendenti, ma raggruppano i thread in unità da 32 chiamate *warp*, condividendo la logica di *fetch* e *decode* delle istruzioni. 
*   Ogni SM è inoltre dotato di veloci risorse fisiche condivise, tra cui un imponente **Register File** (banco dei registri) e memorie cache L1 e condivise.

## I Tipi di Memoria (Gerarchia)

Il modello di memoria di CUDA prevede spazi di memoria distinti, caratterizzati da diverse latenze, larghezze di banda e visibilità da parte dei thread. Procedendo dalla più veloce alla più lenta:

1.  **Registri (Registers):** Costituiscono la memoria più veloce in assoluto (on-chip). Sono rigorosamente privati per ogni singolo thread e vengono usati per memorizzare le variabili locali ad accesso più frequente.
2.  **Memoria Condivisa (Shared Memory):** È una memoria on-chip estremamente veloce (paragonabile alla cache L1) la cui allocazione e utilizzo devono essere esplicitamente gestiti dal programmatore nel codice. È visibile esclusivamente ai thread che appartengono allo stesso **blocco**, permettendo loro di cooperare e condividere risultati intermedi senza dover accedere alla memoria esterna.
3.  **Memoria Locale (Local Memory):** Questa memoria è privata per il singolo thread, ma fisicamente risiede nell'enorme e lenta memoria esterna del *Device* (DRAM). Viene utilizzata automaticamente dal compilatore quando un thread esaurisce i registri a disposizione (fenomeno di *register spilling*) o per allocare array locali troppo grandi.
4.  **Memoria Globale (Global Memory):** È lo spazio di memoria più capiente della GPU (misurabile in Gigabyte), accessibile in lettura e scrittura da *tutti* i thread di qualsiasi blocco, oltre che dall'Host (tramite bus PCIe). Tuttavia, è "off-chip", il che comporta latenze molto elevate (centinaia di cicli di clock). L'accesso a questa memoria dovrebbe sempre essere ottimizzato tramite accessi "coalescenti" per sfruttare la banda passante.
5.  **Memoria Costante (Constant Memory) e Texture Memory:** Sono porzioni specifiche della lenta memoria globale che godono però di **cache hardware dedicate on-chip**. Sono accessibili in sola lettura da parte dei thread della GPU (possono essere scritte solo dall'Host). La memoria costante è ottimizzata per distribuire in un colpo solo lo stesso valore a tutto un *warp* di thread, mentre la Texture Memory è ottimizzata per sfruttare la località spaziale 2D, rivelandosi utile in specifici pattern di accesso.

![Immagine descrittiva dell'architettura di una GPU.](../immagini/gpu_architectures.png)

# Il Modello di Programmazione CUDA: Griglie, Blocchi e Thread

Per sfruttare l'enorme parallelismo delle GPU, CUDA richiede al programmatore di adottare una specifica astrazione gerarchica. Invece di scrivere un programma sequenziale, il programmatore deve pensare a come scomporre il dominio del problema in porzioni indipendenti e via via più piccole. 

Questa astrazione si articola sui seguenti livelli:

### 1. Il Problema Globale (La Griglia / Grid)
Quando il programmatore lancia un *kernel* sulla GPU, crea un'entità chiamata **Grid** (Griglia). La griglia rappresenta l'intero problema computazionale da risolvere (ad esempio, l'elaborazione di un'intera immagine o la moltiplicazione di due grandi matrici). Una griglia può essere monodimensionale (1D), bidimensionale (2D) o tridimensionale (3D), a seconda della topologia dei dati da elaborare.

### 2. Sottoproblemi Indipendenti (I Blocchi / Thread Blocks)  
La griglia è a sua volta suddivisa in numerosi Blocchi. Ogni blocco è un raggruppamento di thread che elabora una specifica e circoscritta porzione del problema globale (ad esempio, un tassello 16x16 pixel dell'immagine). La divisione in blocchi è il segreto della Scalabilità Automatica di CUDA: l'hardware è libero di eseguire questi blocchi in qualsiasi ordine, in parallelo o sequenzialmente
. Se l'utente esegue il programma su una GPU economica con soli 2 Streaming Multiprocessor (SM), i blocchi verranno elaborati un po' alla volta, distribuiti sulle poche risorse disponibili; se usa una GPU di fascia alta con decine di SM, molti più blocchi verranno elaborati in contemporanea, riducendo drasticamente il tempo di esecuzione senza dover modificare una singola riga di codice
. Proprio per questo il programmatore mira a scrivere un programma cosiddetto longevo: suddividendo il problema in moltissimi blocchi indipendenti, si assicura che man mano che saranno disponibili GPU future più potenti, lo stesso codice potrà scalare automaticamente ed essere eseguito in un minor tempo.


### 3. L'Unità Minima Logica (I Thread)
Ogni blocco è composto da una matrice di **Thread**, che rappresentano l'unità minima logica di lavoro. Ciascun thread esegue la medesima sequenza di istruzioni (il *kernel*), ma applicata a un singolo elemento dei dati, calcolando il proprio indirizzo in memoria sfruttando le coordinate fornite dal sistema (`threadIdx` e `blockIdx`). Le dimensioni di un blocco sono decise dal programmatore, ma l'architettura impone un limite massimo invalicabile di 1024 thread per blocco. All'interno dello stesso blocco, i thread possono cooperare e scambiarsi dati velocemente tramite la memoria condivisa (Shared Memory).

### 4. L'Esecuzione Fisica (I Warp)
Mentre Griglia, Blocchi e Thread sono le astrazioni *logiche* usate dal programmatore per scrivere il codice, a livello *hardware* l'esecuzione avviene tramite un'ulteriore suddivisione: i **Warp**.
Quando un blocco viene assegnato a un multiprocessore (SM), l'hardware raggruppa automaticamente i suoi thread in gruppi indivisibili di **32 thread**, chiamati appunto Warp. Tutti i 32 thread di un warp operano in rigoroso regime SIMT (Single-Instruction, Multiple-Thread): eseguono contemporaneamente la stessa identica istruzione ma su dati diversi. Questo permette di semplificare enormemente i circuiti di controllo della GPU, sacrificando però efficienza qualora i thread dello stesso warp debbano prendere percorsi condizionali diversi (fenomeno noto come *thread divergence*).

Il vantaggio nell'utilizzo di tali astrazioni risiede nel fatto che, in questo modo, è possibile gestire la topologia di domini dati ricorrenti (come vettori 1D, immagini 2D o volumi 3D) in maniera semplice e naturale. Tuttavia, questa architettura richiede al programmatore una maggiore attenzione nel calcolo matematico per l'indicizzazione globale di thread e blocchi, al fine di evitare accessi in aree di memoria non consentite.


# Efficienza dei Warp e Divergenza dei Thread

L'architettura alla base delle GPU moderne è definita **SIMT** (Single-Instruction, Multiple-Thread). In questo modello, gli streaming multiprocessor (SM) creano, gestiscono ed eseguono i thread raggruppandoli in unità fisiche indivisibili da **32 thread**, chiamate **warp**.

Per come è progettato l'hardware, un warp può emettere e processare una singola istruzione alla volta. Pertanto, **l'efficienza massima e il pieno utilizzo delle risorse hardware si ottengono esclusivamente quando tutti e 32 i thread di un warp eseguono simultaneamente la stessa identica istruzione**, operando in rigoroso *lockstep*. In questa condizione ideale, nessuna pipeline di calcolo viene sprecata.

Tuttavia, ogni singolo thread possiede un proprio program counter e un proprio stato dei registri, il che gli permette di valutare in autonomia condizioni logiche e prendere percorsi di codice differenti. Quando all'interno del kernel si incontra un'istruzione condizionale dipendente dai dati (come un costrutto `if/then/else`), alcuni thread del medesimo warp potrebbero dover seguire il ramo logico del `then`, mentre altri potrebbero dover eseguire l' `else`. Questa situazione prende il nome di **divergenza dei thread** (*thread divergence*).

Poiché l'hardware del warp non è in grado di eseguire due istruzioni diverse nello stesso istante, la GPU è costretta a risolvere la divergenza **serializzando l'esecuzione**. Il multiprocessore eseguirà sequenzialmente ogni singolo ramo: mentre viene elaborato il percorso di una parte dei thread, i thread che non appartengono a quel ramo vengono momentaneamente **disabilitati e costretti all'attesa**. Questo comportamento fa sì che, per tutta la durata dell'istruzione divergente, una porzione delle unità di calcolo rimanga inattiva e non utilizzata, annullando di fatto il beneficio del parallelismo. Solo una volta terminata l'esecuzione di tutti i percorsi ramificati, i thread del warp si ricongiungono e riprendono a eseguire l'istruzione successiva in piena simultaneità.
![Immagine descrittiva della divergenza tra thread.](../immagini/warp_divergence.png)


# L'Obiettivo del Programmatore: Ottimizzare l'Uso dei Registri

Nella stesura di un programma CUDA, l'hardware ci mette a disposizione un parallelismo imponente, consentendoci di lanciare blocchi che possono contenere fino a un massimo di 1024 thread. Il nostro obiettivo primario come programmatori è proprio quello di **sfruttare appieno questa potenza, riuscendo a far girare tutti e 1024 i thread di un blocco in modo efficiente e simultaneo**. 

Tuttavia, bisogna scontrarsi con una regola fondamentale dell'hardware CUDA: l'allocazione delle risorse per un blocco segue la logica del **"tutto o niente"**. Quando un blocco viene assegnato a uno Streaming Multiprocessor (SM), l'hardware deve riservare i veloci registri privati e la memoria condivisa per *tutti* i thread del blocco in un colpo solo. Il sistema non disattiverà mai alcun thread per cercare di "fare spazio" a un blocco troppo grande.

### Il Rischio: Fallimento del Kernel e Register Spilling
L'ostacolo principale a questo obiettivo si presenta quando scriviamo il codice (il *kernel*) in modo troppo complesso, definendo un numero eccessivo di variabili locali per il singolo thread. Se moltiplichiamo queste variabili per i 1024 thread del nostro blocco, potremmo superare rapidamente la disponibilità fisica dei registri sul multiprocessore.

Se il fabbisogno di memoria del blocco supera le risorse fisiche dell'SM, andiamo incontro a due scenari che distruggono le prestazioni:
1. **Fallimento del Kernel:** Se le risorse non sono sufficienti per ospitare nemmeno un singolo blocco per intero, l'esecuzione fallisce e il programma non viene lanciato.
2. **Register Spilling:** Per evitare il crash, il compilatore tenta di minimizzare l'uso dei registri trasferendo (o "versando") in automatico le variabili in eccesso nella *Local Memory*. Sebbene il blocco riesca a girare, la Local Memory risiede fisicamente nella lentissima memoria globale esterna, e questo causa un crollo drastico della velocità di calcolo di tutti i nostri 1024 thread.

### L'Accortezza del Programmatore
Per questo motivo, programmare in CUDA richiede un'estrema attenzione alla gestione della memoria locale. Il programmatore non può limitarsi a scrivere codice funzionante, ma deve adottare ogni accortezza possibile per **mantenere il singolo thread il più "leggero" possibile**. 

Riducendo al minimo indispensabile le variabili locali definite nel codice e riutilizzando i registri, il programmatore si assicura che il fabbisogno totale del blocco non ecceda i limiti dell'SM. Solo attraverso questa meticolosa ottimizzazione possiamo raggiungere il nostro vero traguardo: garantire che il multiprocessore riesca a ospitare i nostri blocchi da 1024 thread interamente nelle sue memorie ultra-veloci, sfruttando fino all'ultima goccia la potenza di calcolo della GPU.


# Funzioni Global e il Concetto di Kernel

Nella programmazione CUDA, il codice che viene materialmente eseguito sulla GPU prende il nome di **kernel**. A differenza di una normale funzione C/C++ che viene elaborata una sola volta e in modo sequenziale, un kernel è progettato per essere eseguito simultaneamente in parallelo da decine di migliaia di thread indipendenti.

Per definire un kernel e istruire il compilatore (`nvcc`) a trattarlo come tale, CUDA estende il linguaggio C/C++ introducendo un qualificatore di spazio di esecuzione specifico: **`__global__`**.

### Il qualificatore `__global__`
Aggiungere la parola chiave `__global__` prima della definizione di una funzione stabilisce un vero e proprio ponte tra la CPU e la GPU. Le caratteristiche fondamentali di una funzione `__global__` sono tre:

1. **Esecuzione sul Device:** L'intero blocco di codice della funzione viene compilato nel formato binario della GPU ed eseguito fisicamente sui processori del *Device*.
2. **Chiamata dall'Host:** La grande particolarità di questa funzione è che **l'invocazione avviene da parte della CPU (*Host*)**, tipicamente all'interno del programma principale (es. dal `main`). Questo rappresenta il momento esatto in cui l'Host cede il controllo dell'operazione e sposta l'esecuzione sul Device. 
3. **Restituisce sempre `void`:** Poiché la funzione viene eseguita in parallelo da innumerevoli thread, non avrebbe senso farle restituire un singolo valore con un `return`. Per questo motivo, le funzioni `__global__` devono avere obbligatoriamente un tipo di ritorno `void`. I risultati dell'elaborazione vengono invece salvati scrivendoli in array o strutture allocate in precedenza sulla memoria della GPU, utilizzando i puntatori passati come argomenti.

### Il lancio del Kernel (Kernel Launch)
Essendo una funzione speciale, un kernel `__global__` non può essere chiamato con la normale sintassi del C/C++. Affinché la GPU sappia quanti thread attivare per svolgere il lavoro, il programmatore deve specificare la cosiddetta **configurazione di esecuzione**.

Questa configurazione si esprime inserendo il numero di griglie e di blocchi tra **tre parentesi angolari `<<< ... >>>`**, posizionate subito dopo il nome della funzione e prima degli argomenti standard:

```cpp
// Definizione del kernel tramite qualificatore __global__
__global__ void mio_kernel(int *dati_input, int *risultati) {
    // Codice che ogni singolo thread eseguirà sulla GPU...
}

int main() {
    // ... allocazione memoria e copia dati (Host to Device) ...

    // Lancio del kernel: la chiamata parte dall'Host, l'esecuzione va sul Device
    mio_kernel<<<dimGrid, dimBlock>>>(dati_input);

    // ... recupero dei risultati (Device to Host) ...
}
```

Un aspetto cruciale da ricordare quando si lancia una `__global__` è la sua **asincronia**. Non appena la CPU esegue l'istruzione di lancio (`<<<...>>>`), il controllo le viene restituito immediatamente. L'Host non aspetta che la GPU finisca i calcoli, ma passa istantaneamente all'istruzione successiva; se ha bisogno di attendere i risultati prima di procedere, dovrà invocare esplicitamente una barriera di sincronizzazione tramite il comando `cudaDeviceSynchronize()`.

