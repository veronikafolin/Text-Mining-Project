# Text Mining Project

Progetto svolto per il corso di Text Mining - Università di Bologna

Autore: Veronika Folin

## Descrizione del progetto

### Obiettivo
L'obiettivo di questo progetto è quello di addestrare e valutare dei modelli Transformer nella classificazione delle pronunce effettuate dalla Corte Costituzionale Italiana. 

### Descrizione del dominio

Una pronuncia della Corte Costituzionale è l'atto che conclude il procedimento costituzionale e consta di:
1. *epigrafe*
2. *testo*, ulteriormente suddiviso in:
  - *Ritenuto in fatto* - l'input della corte che argomenta la controversia
  - *Considerato in diritto* - la parte in cui la Corte espone le ragioni poste a fondamento della sua decisione
3. *dispositivo* - la parte conclusiva che contiene la determinazione della Corte

Le prununce sono classificabili in due tipologie:
- *ordinanza* - provveddimento temportaneo ed urgente per prevenire un danno immediato o proteggere un diritto. Hanno più probabilità di essere rigettate dalla Corte costituzionale.
- *sentenza* - decisione finale e definitiva.

Il *giudizio* della Corte può essere classificato in base alla ragione per cui un certo fatto le è stato presentato.

Una pronuncia può essere corredata da una o più massime, ossia un'elaborazione in forma sintetica della motivazione e del dispositivo delle pronunce della Corte.

## Task 

Il task si delinea nei seguenti macro-punti:

Nel file _requirements.txt_ sono elencate le dipendenze del progetto.

### 1. Web scraping

L'obiettivo di questo punto è ottenere la dicitura _giudizio_ e i _parametri costituzionali_ per ogni pronuncia del dataset di partenza e, conseguentemente, popolare il dataset.


Per effettuare il web scraping della [pagina web](https://www.cortecostituzionale.it/actionPronuncia.do) è stato sfruttato il tool open source [Selenium](https://www.selenium.dev/), sviluppato per la web automation e il web testing.

Nella cartella **_webScraping_** si trovano:
- _**chromedriver_win32**_: all'interno di questa cartella possiamo trovare il web driver, specifico per Google Chrome, necessario per il funzionamento di Selenium
- _**script.py**_: script per eseguire lo scraping della pagina web e salvare il dataset aggiornato in locale. È possibile, inoltre, rieseguire lo scraping di alcune porzioni del dataset per cui sono stati riscontrati numerosi errori di caricamento, modificando opportune righe di codice. 
- _**debug.py**_: script per verificare il risultato del processo di scraping e per risolvere manualmente specifici errori che si sono verificati a seguito della prima fase di correzione.
- _**executionTime.txt**_: file dove vengono riportati i tempi di esecuzione dello scraping.

Per eseguire gli script è necessario:
- creare una nuova cartella _datasets_ nella root del progetto
- scaricare il dataset di partenza in locale 
- spostare la directory appena scaricata nella cartella _datasets_

Per lanciare gli script, posizionarsi nella cartella _webScraping_ ed eseguire, ad esempio:
```powershell
$ python script.py
```

### 2. Esplorazione del dataset

L'obiettivo è esplorare il dataset e, in particolare, produrre delle statistiche rispetto le features prese in considerazione per i successivi task di classificazione.  

Per questo punto è stato prodotto un notebook _Colab_, che si può trovare al seguente [link](https://colab.research.google.com/drive/1AckUKN7L2ylpYRXp2kmDnA6ApP_tMJ-J?usp=sharing).

### 3. Task di classificazione

L'obiettivo è:
1. Addestrare modelli e classificare le pronunce in *ordinanze* o *sentenze* (2 classi), poi valutarne le performance.
2. Addestrare modelli e classificare le pronunce in base alla *tipologia  di giudizio* (13 classi), poi valutarne le performance.

I modelli utilizzati per effettuare i task sono stati:
- [dlicari/lsg16k-Italian-Legal-BERT](https://huggingface.co/dlicari/lsg16k-Italian-Legal-BERT)
- [facebook/mbart-large-50](https://huggingface.co/facebook/mbart-large-50)

Nella cartella **_classification_** si trovano:
- _**classify.py**_: script per eseguire fine-tuning di uno specifico modello (di libreria o disponibile in locale) su un dataset (di libreria o disponibile in locale) rispetto al task di sequence classification. Il codice è un adattamento di quello reso disponibile dalla libreria _HuggingFace_ a questo [link](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
- _**run.sh**_: per lanciare automaticamente lo script _classify.py_, specificando gli opportuni parametri di configurazione all'interno. 
- _**output**_: directory in cui vengono salvate le predizioni del modello, categorizzate per task.

Per lanciare lo script _run.sh_, posizionarsi nella cartella _classification_ ed eseguire:
```powershell
$ sh run.sh
```

Di seguito vengono riassunti gli esperimenti effettuati e i rispettivi risultati ottenuti.

|   |   |   |   |   |
|---|---|---|---|---|
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |