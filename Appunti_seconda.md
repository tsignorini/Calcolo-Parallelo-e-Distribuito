# Revisione sul Calcolo Distribuito con MPI

Ecco le due opzioni di riscrittura e miglioramento dell'estratto relativo a MPI e al calcolo distribuito.

---

#

## Opzione 2: Schematica e Chiara (Ottima per appunti o presentazioni)

> "MPI è lo standard utilizzato nel calcolo distribuito sui supercalcolatori. A differenza di OpenMP, non si basa su direttive ma è una libreria software che offre un controllo a basso livello sulle componenti del sistema, garantendo un'ottimizzazione spinta della computazione. Il suo utilizzo è necessario quando la taglia del problema eccede le capacità di una singola macchina.
> 
> **Vantaggi:**
> * **Scalabilità:** Permette di superare i limiti di memoria della singola macchina.
> * **Sicurezza:** Assenza di *data race*, dato che ogni processo ha il proprio spazio di memoria privato.
> 
> **Svantaggi:**
> * **Overhead di rete:** Lo scambio di messaggi e la sincronizzazione avvengono tramite subroutine via rete, introducendo latenze che possono rallentare l'esecuzione.
> * **Complessità:** Il codice diventa sensibilmente più lungo, articolato e difficile da gestire rispetto al paradigma a memoria condivisa."
