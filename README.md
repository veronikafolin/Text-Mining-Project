﻿# Text Mining Project

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
Il sito della Corte Costituzionale riporta le seguenti tipologie di giudizio:
- GIUDIZIO DI ACCUSA
- GIUDIZIO DI LEGITTIMITÀ COSTITUZIONALE IN VIA INCIDENTALE
- GIUDIZIO DI LEGITTIMITÀ COSTITUZIONALE IN VIA PRINCIPALE
- GIUDIZIO DI LEGITTIMITÀ COSTITUZIONALE IN VIA PRINCIPALE + GIUDIZIO DI LEGITTIMITÀ COSTITUZIONALE IN VIA INCIDENTALE
- GIUDIZIO PER CONFLITTO DI ATTRIBUZIONE TRA ENTI
- GIUDIZIO PER CONFLITTO DI ATTRIBUZIONE TRA POTERI DELLO STATO
- GIUDIZIO PER LA CORREZIONE DI OMISSIONI E/O ERRORI MATERIALI
- GIUDIZIO PER LA SOSPENSIONE DELL'ATTO IMPUGNATO
- GIUDIZIO SU CONFLITTO DI ATTRIBUZIONE TRA ENTI + GIUDIZIO DI LEGITTIMITA' COSTITUZIONALE IN VIA INCIDENTALE
- GIUDIZIO SU CONFLITTO DI ATTRIBUZIONE TRA ENTI + GIUDIZIO DI LEGITTIMITA' COSTITUZIONALE IN VIA PRINCIPALE
- GIUDIZIO SULL'AMMISSIBILITÀ DEI REFERENDUM
- GIUDIZIO SULL'AMMISSIBILITÀ DI RICORSO PER CONFLITTO DI ATTRIBUZIONE TRA POTERI DELLO STATO
- QUESTIONE INCIDENTALE DI LEGITTIMITA' COSTITUZIONALE

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
- _**debug.py**_: script per verificare il risultato del processo di scraping e per risolvere manualmente specifici errori che persistono a seguito della prima fase di correzione.
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
1. Addestrare modelli e classificare le pronunce in *ordinanze* o *sentenze* (2 classi), poi valutarne le performance [Ruling]
2. Addestrare modelli e classificare le pronunce in base alla *tipologia  di giudizio* (13 classi), poi valutarne le performance. [Jud

I modelli utilizzati per effettuare i task sono stati:
- [facebook/mbart-large-50](https://huggingface.co/facebook/mbart-large-50), ossia un modello encoder-decoder pre-addestrato su 50 lingue, in grado di processare sequenze fino a 1024 token;
- [ccdv/lsg-xlm-roberta-base-4096](https://huggingface.co/ccdv/lsg-xlm-roberta-base-4096), ossia un multi-lingual masked language model pre-addestrato su 100 lingue e potenziato con _Local + Sparse + Global attention (LSG)_ al fine di processare fino a 4096 token.

Per valutare le performance dei modelli sui task presi in esame sono state calcolate le seguenti metriche:
- **μ-F1**
- **m-F1**, che considera il problema della classi sbilanciate, in particolare nel secondo task di classificazione
- **Carburacy**, che prende in considerazione sia l'efficacia che l'ecosostenibilità del modello

Nella cartella **_classification_** si trova:
- _**run_classification.py**_: script per eseguire fine-tuning di uno specifico modello (di libreria o disponibile in locale) su un dataset (di libreria o disponibile in locale) rispetto al task di sequence classification. Il codice è un adattamento di quello reso disponibile dalla libreria _HuggingFace_ a questo [link](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

Per lanciare lo script è possibile eseguire, ad esempio, il seguente codice:

```powershell
python3 tasks/run_classification.py \
  --logging online \
  --lang es \
  --do_train \
  --do_eval \
  --do_predict \
  --output_dir ../output \
  --task_name ruling_classification \
  --model_name_or_path facebook/mbart-large-50 \
  --dataset_name disi-unibo-nlp/COMMA \
  --log_level error \
  --gradient_accumulation_steps 1 \
  --max_seq_length 1024 \
  --learning_rate 5e-5 \
  --num_train_epochs 4 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --fp16 \
  --gradient_checkpointing \
  --load_best_model_at_end \
  --overwrite_cache \
  --save_total_limit 1 \
  --weight_decay 0.01 \
  --label_smoothing_factor 0.1 \
  --remove_unused_columns \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
```

Di seguito vengono riassunti gli esperimenti effettuati per la lingua italiana e i rispettivi risultati ottenuti.

| Modello  |   |   |   |   |
|---|---|---|---|---|
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |
