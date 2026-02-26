# Perché Il Calcolo Parallelo
Il calcolo parallelo ha come obiettivo quello di ridurre i tempi di calcolo e di distribuire meglio il carico di lavoro tra le varie componenti addette al calcolo. Sebbene la parte Hardware è fortemente parallelizzata, lo sviluppo di software parallelo è ancora limitato, in quanto il passaggio da seriale a parallelo non è sempre banale, ma anzi spesso richiede il ripensamento dell'intero problema da capo. La diffusione dell'hardware parallelo è stata guidata da una riduzione dei consumi, infatti la formula della potenza di una cpu è data da:
$$
P = C \cdot V^2 \cdot f
$$
dove $C$ è la capacità cioè quanto un circuito riesce a immagazzinare energia, $V$ è il voltaggio a cui il circuito lavora mentre $f$ è la frequenza di aggiornamento del circuito, vale a dire il numero di volte nell'unità di tempo in cui si è in grado di cambiare lo stato globale della macchina. 
Come si vede dalla formula, per ridurre fortemente i consumi bisogna abbassare il voltaggio a cui si lavora, per farlo si diminuisce la frequenza del processore, riducendo quindi anche il voltaggio, ma si usano più processori per ottenere un output nello stesso tempo dato da un processore ad una maggiore frequenza.
![Immagine descrittiva della riduzione della frequenza di un processore](immagini/frequenza_processore.png)
Quindi 
