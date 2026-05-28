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
* Per massimizzare le prestazioni ed evitare continui e lenti accessi alla memoria globale, la GPU cerca di trattenere i dati all'interno del chip, sfruttando memorie ultra-veloci come i registri privati e la memoria condivisa (Shared Memory).
* Il lancio del kernel è **asincrono**: il controllo ritorna immediatamente alla CPU, che può proseguire con altre operazioni oppure mettersi in attesa del completamento del kernel richiamando `cudaDeviceSynchronize()`.

### 3. Recupero dei risultati (Device to Host)
Al termine dell'elaborazione, i dati risultanti risiedono ancora nella memoria del Device. Affinché il programma principale possa utilizzarli, salvarli o stamparli, la CPU deve inviare un nuovo comando di copia. Si utilizza nuovamente `cudaMemcpy`, ma con direzione inversa (`cudaMemcpyDeviceToHost`), per copiare i risultati dalla memoria della GPU a quella della CPU, transitando sempre sul bus PCIe.

Come operazione finale di pulizia (Cleanup), la memoria precedentemente allocata sulla GPU deve essere liberata invocando la funzione `cudaFree()`.

