# Perché Il Calcolo Parallelo

## Introduzione

Il calcolo parallelo rappresenta una delle evoluzioni più significative nell'informatica moderna, con l'obiettivo primario di **ridurre i tempi di calcolo** e **distribuire efficacemente il carico di lavoro** tra le varie unità computazionali. 

Sebbene l'hardware sia ormai fortemente parallelizzato, lo sviluppo di software parallelo rimane una sfida complessa. Il passaggio da un'architettura seriale a una parallela non è mai banale e richiede spesso un **ripensamento completo** dell'algoritmo e della struttura del problema.

---

### Il Problema del Consumo Energetico

La diffusione massiva dell'hardware parallelo è stata guidata principalmente dalla necessità di **ridurre i consumi energetici**. La potenza dissipata da un processore è descritta dalla formula:

$$P = C \cdot V^2 \cdot f$$

dove:
- **$C$** è la capacità del circuito (quanto il circuito può immagazzinare energia)
- **$V$** è il voltaggio operativo del circuito
- **$f$** è la frequenza di clock, ovvero il numero di cicli per unità di tempo in cui il processore può cambiare il proprio stato

### La Soluzione: Più Core a Frequenze Ridotte

Come evidenziato dalla formula, la potenza è **proporzionale al quadrato del voltaggio**. Questo significa che riduzioni anche piccole del voltaggio portano a significativi risparmi energetici.

La strategia adottata dall'industria è stata:
1. **Ridurre la frequenza** di clock dei singoli processori
2. **Abbassare il voltaggio** operativo (possibile grazie alla minore frequenza)
3. **Aumentare il numero di core** per mantenere o superare le prestazioni complessive

![Immagine descrittiva della riduzione della frequenza di un processore](immagini/frequenza_processore.png)

*Il grafico mostra come l'industria sia passata da processori single-core ad alta frequenza a processori multi-core a frequenze ridotte, mantenendo o incrementando le prestazioni complessive con consumi inferiori.*

---

## Obiettivi e Sfide del Calcolo Parallelo

### Obiettivi Principali

Il calcolo parallelo mira a:
- **Ridurre il tempo di calcolo** per problemi complessi e di grandi dimensioni
- **Migliorare l'efficienza energetica** rispetto a soluzioni seriali ad alte prestazioni
- **Permettere la risoluzione** di problemi altrimenti intrattabili

### Requisiti per un'Efficace Parallelizzazione

Per ottenere un reale vantaggio rispetto all'esecuzione seriale, è necessario:

1. **Scomporre il problema** principale in sottoproblemi più semplici e possibilmente indipendenti
2. **Assegnare i sottoproblemi** alle componenti hardware più appropriate (CPU, GPU, acceleratori)
3. **Minimizzare la comunicazione** e la sincronizzazione tra le unità di calcolo
4. **Bilanciare il carico** di lavoro per evitare che alcune unità restino inattive

---

## Il Problema della Concorrenza

### Cos'è la Concorrenza

Uno degli aspetti più critici nella programmazione parallela è la **gestione della concorrenza**: quando più elemnti accedono contemporaneamente alla stessa area di memoria, possono verificarsi **race condition** che portano a risultati inconsistenti o errati.

### Esempio Pratico: Prelievo e Bonifico Bancario

Consideriamo un'operazione bancaria comune: il prelievo da un conto corrente. L'operazione richiede tre passaggi:

1. **Lettura del saldo** corrente
2. **Calcolo del nuovo saldo**: `saldo_nuovo = saldo - somma_prelevata`
3. **Aggiornamento del saldo**: `saldo = saldo_nuovo`

#### Scenario Problematico

Supponiamo che, mentre è in corso un prelievo, arrivi un bonifico che deve aggiornare il saldo aggiungendo un importo. Se il bonifico viene elaborato **tra il passo 2 e il passo 3** dell'operazione di prelievo, si verifica il seguente scenario:

| Tempo | Prelievo | Bonifico | Saldo in Memoria |
|-------|----------------|----------------|------------------|
| T0 | Legge saldo: 1000€ | | 1000€ |
| T1 | Calcola: 1000€ - 100€ = 900€ | | 1000€ |
| T2 | | Legge saldo: 1000€ | 1000€ |
| T3 | | Calcola: 1000€ + 10000€ = 11000€ | 1000€ |
| T4 | | Scrive saldo: 11000€ | **11000€** |
| T5 | Scrive saldo: 900€ | | **900€** |

**Risultato**: Il bonifico di 1000€ viene completamente perso! Il saldo finale è 900€ invece dei corretti 10900€.

![Immagine descrittiva delle problematiche che possono insorgere nella programmazione parallela](immagini/prelievo_bonifico.jpeg)

*Visualizzazione della race condition: le operazioni concorrenti sul saldo portano alla perdita di dati.*

### Soluzioni alla Concorrenza

Per evitare questi problemi, la programmazione parallela richiede:
- **Mutex e lock**: meccanismi di esclusione mutua per garantire accesso esclusivo alle risorse condivise
- **Semafori**: per coordinare l'accesso a risorse limitate
- **Transazioni atomiche**: operazioni che vengono eseguite completamente o per nulla
- **Strutture dati lock-free**: progettate per minimizzare la sincronizzazione

---

## Conclusioni

Il calcolo parallelo rappresenta sia una **necessità tecnologica** (per gestire consumi energetici e limiti fisici) sia una **sfida ingegneristica** complessa. 

La sua efficacia dipende non solo dall'hardware disponibile, ma soprattutto dalla capacità dei programmatori di:
- Identificare opportunità di parallelizzazione
- Gestire correttamente la sincronizzazione
- Evitare race condition e deadlock
- Bilanciare il carico tra le unità di calcolo

Nonostante le difficoltà, il calcolo parallelo è ormai **indispensabile** per affrontare problemi moderni in ambiti come l'intelligenza artificiale, la simulazione scientifica, l'analisi di grandi quantità di dati e il rendering grafico.
