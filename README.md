# Calibration bias

questa repository contiene codice e dati per il lavoro sulla calibration usata per calcolare i bias nelle predizioni dei modelli (ci sono i requirements.txt se volete riprodurre tutto in un virtual environment). La proposta di lavoro (e quindi il codice) è strutturata in due parti tra loro collegate

## come vengono caratterizzati i dataset disaggregati?
ormai ci sono molti dataset disaggregati ma cosa hanno in comune e in cosa invece divergono? In questo lavoro proviamo ad analizzare i dati su tre diversi assi: la distribuzione delle maggioranze, i gruppi sociali più rappresentati, la divergenza media degli annotatori dal voto di maggioranza.

nella cartella 'data/measuring_hatespeech' ci sono i seguenti file relativi a un primo esempio di dataset trattato completamente:
* 'corpus.csv': è un file con le seguenti colonne 'comment_id,text,annotator_id,social_group,violence,insult'. social_group è la colonna che divide gli annotatori nei quattro gruppi che conosciamo: 1=uomini bianchi; 2=donne bianche; 3=uomini non-bianchi; 4=donne non-bianche

Partendo da un file con queste colonne è possibile creare i seguenti file tramite lo script 'src/0.process_dataset.py':
* 'majority_types.csv' associa a ogni testo il tipo di maggioranza tra 4 opzioni: unanimità, assoluta (>=66.6%); maggioranza (>50%) e no maggioranza
* 'top_social_groups.csv' mostra il numero di volte in cui un social_group è il più rappresentato in un voto di maggioranza. Es, data la label '1' votata da 4 persone su 6. se 3 di queste 4 persone sono donne bianche allora il gruppo più rappresentato sono loro.
* annotator_false_ratios.csv' contiene la percentuale di volte in cui l'annotatore annota in accordo con la maggioranza

con questi quattro file è possibile compilare il file generale di assessment 'dataset_assessment.csv', che riporta le statistiche generali di ogni corpus (per ora ce n'è solo una)

### bug e cose da ottimizzare
* nel calcolare la divergenza degli annotatori dal voto di maggioranza non tengo conto del fatto che due label possano essere ugualmente maggioritarie. Per esempio, in una scala da 0 a 4 avere sia l'1 sia il 4 con 2 voti
* manca l'automatismo per fare l'assessment su tanti dataset diversi
* il codice potrebbe essere razionalizzato con classi etc.

### cose da fare
* aggiustare gli attuali bug
* preparare altri dataset
* ragionare su eventuali test statistici per trovare regolarità e differenze
* scegliere due dataset per la fase 2


## come poter usare la calibrazione per calcolare i bias nei modelli?
questa seconda parte è il follow-up del nostro precedente articolo. L'idea è che, preso un modello e le annotazioni di un utente, calcolo il conformity score basato sulla loss di brier tra le predizioni del modello e l'annotazione dell'utente. lo script che cercate si trova in 'src/2.calibration.py' e ha già un working example che prende in input tutte le annotazioni di un utente e restituisce come output tutte le conformity loss e la loro media, cioè il conformity score. 

### cose da fare
* scegliere su quali annotatori applicare questo metodo (i 10 con la divergenza maggiore, i 10 con quella minore e i 10 più vicini alla media?) 
* scegliere quali modelli testare. PROPOSTA: due famiglie di modelli con tre dimensionalità in termini di parametri. Es., llama instruct 1b, llama instruct 7b e llama instruct 24b
* a cosa ci serve la conformity? sicuramente per analizzare i bias del modello (conformity alta == modello insicuro sull'annotazione) ma anche per misurare la polarizzazione dell'annotatore rispetto ad altri? o per identificare feature latenti che magari hanno a che fare con la difficoltà del testo?


### timeline
è ottimale che tutti i risultati siano pronti per fine novembre, in modo da avere qualche settimana per scrivere il paper e non doverci lavorare nelle vacanze di natale. alcune cose pratiche (raccogliere i dataset e runnare i modelli) non hanno bisogno di discussione ma io non posso farle. su altre possiamo confrontarci anche in asincrono. 3 le più importanti per non bloccare il lavoro in pipeline:
* quali dataset scegliere
* quali modelli usare
* su quali sample di annotatori lavorare

