# Report di implementazione — Localizzatore di perimetro (missione `PERIMETER`)

Questo documento descrive il modulo
`src/sft_cv/decision_system/decision/perimeter.py`, aggiunto per gestire la
localizzazione dei GCP che delimitano l'area di missione. È diviso in due parti:

1. **Approccio e giustificazione** — perché è fatto così.
2. **Uso del file e feature configurabili** — come si attiva e come si regola.

---

## 1. Approccio e giustificazione

### 1.1 Il problema

Nella missione `PERIMETER` ci sono **3 o 4 GCP identici** (stessa classe del
classificatore, `no_number_GCP` = classe 10) che **delimitano l'area** dentro la
quale si trovano i veri target da trovare. Poiché il modello **non fa instance
detection**, tutti i GCP arrivano al decision system come detection della stessa
classe: l'unico elemento che li distingue è la **posizione geografica**.

Le difficoltà reali, emerse dall'analisi:

- **Falsi positivi**: il detector produce detection sparse che non corrispondono
  a nessun GCP.
- **Recall variabile**: durante un survey un GCP può essere visto in **1 sola
  immagine oppure in 10**, a seconda dell'overlap, della posizione nello swath e
  delle occlusioni. Non sappiamo quante volte un GCP verrà visto.
- **Numero incerto**: i GCP sono 3 *oppure* 4, e potremmo non saperlo prima del
  decollo.
- **Costo asimmetrico dell'errore**: il perimetro serve solo a **limitare l'area
  di survey** (per avere tempo di trovare i veri target). Un perimetro troppo
  grande/falso fa perdere tempo; uno troppo piccolo esclude zone dove stanno i
  target reali. Entrambi gli errori costano.

### 1.2 Perché NON il vecchio `instance_splitter`

L'`instance_splitter` esistente usa **componenti connesse** su una soglia di
distanza euclidea fissa (15 m). Questo equivale a un clustering *single-linkage*
ed è fragile proprio nei casi che ci interessano:

- **chaining**: un singolo falso positivo a metà strada tra due GCP collega i due
  cluster (catena A → FP → B, ogni salto < 15 m) e li fonde in uno → perdiamo un
  vertice;
- **frammentazione**: lo scatter della geolocalizzazione può spezzare un singolo
  GCP in più componenti → vertici di troppo;
- **nessuna riconciliazione del conteggio**: emette tanti centroidi quante sono
  le componenti, senza mai ricondurli al numero noto (3–4).

Per questo motivo, **per la sola missione `PERIMETER`** è stato aggiunto un
percorso separato. Il codice esistente (`CentroidAnalyzer`, `Decision`,
`instance_splitter`) **non è stato toccato** e continua a servire tutte le altre
missioni (`DEFAULT`, `NUMBERS`, `BULLSEYE`), dove ogni target ha `clusters: 1`.

### 1.3 Il tipo di approccio scelto

L'idea di partenza viene dalla simulazione di brainstorming
(`target_localization_sim.py`): **DBSCAN + reiezione del rumore + centroide
robusto**. È la direzione giusta, ma con due correzioni importanti rispetto a
quella bozza:

1. **`min_samples` abbassato a 2** invece di 4. Con `min_samples=4` un GCP visto
   1–3 volte verrebbe scartato come rumore: esattamente il problema della recall
   variabile. Il filtro dei falsi positivi viene spostato dalla *soglia di
   conteggio* alla *fase di selezione* (ranking).
2. **Eliminato lo step geometrico "se trovo 3 inferisco il 4° vertice"** (regola
   del parallelogramma). Quello step assume che la configurazione vera sia un
   quadrilatero con un vertice mancante; ma le regole dicono che possono essere
   **genuinamente 3** (un triangolo). Inventare un 4° vertice in quel caso
   distorce l'area di survey. Non si fabbricano mai vertici.

Lo schema finale è una **pipeline a candidati + ranking + selezione**, che sfrutta
il fatto di sapere che i vertici sono 3 o 4: invece di "tagliare con una soglia
magica", si **classificano i candidati e si scelgono i migliori**.

#### Passo 1 — Generazione dei candidati (clustering)

`DBSCAN(eps=eps_m, min_samples)` sui punti in coordinate **ENU metriche**
(output di geoloc). I punti con etichetta `-1` (rumore: falsi positivi isolati)
vengono scartati. In alternativa si può usare `HDBSCAN`, che evita di dover
tarare `eps` (lavora su tutte le scale e tiene i cluster più stabili) — utile se
la taratura di `eps` risulta difficile.

#### Passo 2 — Centroide robusto per cluster

Due modalità:

- `weighted_mean` (default): **media pesata sulla precisione**, cioè risolve
  `(Σ Wᵢ) x = Σ Wᵢ pᵢ` con `Wᵢ` = inversa della covarianza di geoloc del punto.
  Le detection più certe pesano di più. È l'unico punto in cui si usano davvero
  le **covarianze** che geoloc già calcola.
- `median`: mediana spaziale per asse, robusta agli outlier interni; fallback
  automatico se le covarianze mancano o il sistema è singolare.

#### Passo 3 — Ranking

I cluster vengono ordinati per **support** (numero di detection che li
sostengono), in ordine decrescente. (È previsto un gancio per moltiplicare lo
score per la confidenza media / il detection-rate normalizzato sulla copertura,
quando questi segnali verranno propagati — vedi §1.5.)

#### Passo 4 — Selezione (decisione 3-vs-4)

- **Conteggio noto** (`clusters: n`, intero): si prendono i primi `n` cluster. Se
  ne sono stati trovati meno di `n`, si restituisce quello che c'è **con un
  warning**, senza inventare vertici.
- **Conteggio incerto** (`clusters: [min, max]`, es. `[3, 4]`): si prendono
  sempre i primi `min`; ogni ulteriore candidato fino a `max` viene accettato
  **solo se** supera due test contemporaneamente:
  1. **soglia di support**: `support ≥ keep_extra_frac × mediana(support dei
     primi min)` — un cluster troppo debole rispetto agli altri non diventa un
     vertice;
  2. **espansione dell'inviluppo convesso**: il candidato deve cadere **fuori**
     dal convex hull dei vertici già scelti. Un falso positivo *interno* all'area
     (anche se denso) non amplia l'inviluppo e viene rifiutato; un vero 4° vertice
     sta sempre sul bordo. È un test **indipendente dalla forma** (vale per
     qualsiasi triangolo/quadrilatero), giustificato dal fatto che "i GCP sono i
     punti estremi dell'area delimitata".

#### Passo 5 — Output

Vengono restituite solo le **coordinate dei vertici** (ENU), che passano per il
`put()` esistente del decision system → conversione ENU→GPS → lista di waypoint
nel `SFT_localization_report.json` sotto il nome del target. Le fasi successive
(planner di survey) restano una **black-box** che riceve solo i vertici.

### 1.4 Garanzie di robustezza

- **Deterministico**: DBSCAN/HDBSCAN sono deterministici; nessun campionamento
  casuale. Stesso input → stesso output (verificato da test).
- **Non fabbrica mai vertici**: se trova meno del minimo atteso, restituisce meno
  vertici con warning. Meglio un'area un po' più piccola che un vertice inventato.
- **Non perde i GCP poco visti**: `min_samples=2` permette di tenere un GCP visto
  solo 2 volte.
- **Reiezione dei FP**: gli isolati diventano rumore (DBSCAN); quelli interni
  all'area vengono bocciati dal test sull'inviluppo convesso.

### 1.5 Limiti noti (onestà intellettuale)

- **Falso positivo sistematico**: se il detector spara in modo ripetuto su un
  oggetto confondibile a terra (visto in molte immagini con overlap), questo forma
  un cluster denso e consistente **indistinguibile** da un vero GCP con i soli
  dati spaziali. La difesa naturale è la **confidenza del classificatore**, che
  oggi viene **scartata** in `input_format` (`decision_system.py`, legge solo
  `bbox` e `class`, ignora `conf`). Propagare `conf` e inserirla nello score
  (Passo 3) è il miglioramento a maggior valore, ma è fuori dallo scope additivo
  di questa implementazione.
- **`eps_m` è un punto di partenza**: va legato al σ reale della geolocalizzazione
  alla quota di survey (100 m). Da validare prima del volo, oppure usare
  `method: hdbscan` per evitare la taratura.

---

## 2. Uso del file e feature configurabili

### 2.1 Quando si attiva

Il localizzatore di perimetro viene eseguito **solo** quando la missione attiva è
`PERIMETER` (cioè `self._current_task == "PERIMETER"`, impostato via comando
inbound, vedi `pipeline_stage.check_inbound_command`). In tutte le altre missioni
si usa il percorso classico, invariato.

Il branch è in `DecisionSystem.mission()`: alla fine della missione, dopo aver
drenato gli input, se la missione è `PERIMETER` chiama
`self._produce_perimeter_output()` e salta il blocco `CentroidAnalyzer` standard.

### 2.2 Quali classi processa

`_produce_perimeter_output` itera sui target con `clusterize: true`. In
`mission_config.yaml`, sotto il task `PERIMETER`, è attiva la sola classe 10:

```yaml
- name: "PERIMETER"
  inference_engine:
    classes: [10]
  decision_system:
    class_handler:
      classes: [10]
    targets:
      10: {name: no_number_GCP, clusters: [3, 4], clusterize: true}
```

Il campo **`clusters`** del target decide il comportamento:

- `clusters: 4` → **conteggio noto**: restituisce esattamente 4 vertici (o meno,
  con warning, se ne trova meno). Da usare se al briefing si sa che sono 4.
- `clusters: 3` → conteggio noto a 3 (triangolo).
- `clusters: [3, 4]` → **conteggio incerto**: decide 3 o 4 a runtime (vedi
  Passo 4). È l'impostazione di default consigliata quando non si sa in anticipo.

### 2.3 Parametri configurabili

Nel `sft_config.yaml`, sotto `decision_system.perimeter` (valori di default;
sovrascrivibili nel task `PERIMETER` di `mission_config.yaml` tramite il merge di
`update()`):

```yaml
decision_system:
  perimeter:
    method: dbscan          # dbscan | hdbscan
    eps_m: 4.0              # raggio di vicinato DBSCAN, in metri
    min_samples: 2          # detection minime per formare un cluster
    centroid: weighted_mean # weighted_mean | median
    keep_extra_frac: 0.5    # soglia di support per accettare un vertice "extra"
```

| Parametro | Significato | Effetto pratico |
|---|---|---|
| `method` | algoritmo di clustering. `dbscan` usa `eps_m`; `hdbscan` non richiede `eps_m` (usa `min_samples` come `min_cluster_size`). | `hdbscan` se non vuoi tarare `eps_m` o se la densità dei cluster varia molto. |
| `eps_m` | distanza max (in metri) tra due detection per considerarle vicine (solo DBSCAN). | **Troppo piccolo** → un GCP si spezza in più cluster. **Troppo grande** → due GCP vicini si fondono. Va legato al σ di geoloc. |
| `min_samples` | numero minimo di detection per fare cluster; sotto questo numero il punto è rumore. | `2` = serve solo conferma da ≥2 frame. Alzarlo rende più conservativi (più reiezione FP, ma rischio di perdere GCP poco visti). |
| `centroid` | come si calcola il centro del cluster. | `weighted_mean` usa le covarianze di geoloc (consigliato); `median` è più robusto ma ignora l'incertezza. |
| `keep_extra_frac` | nel caso a conteggio incerto, frazione del support mediano dei vertici base che il candidato "extra" deve superare. | Più alto = più difficile accettare un 4° vertice (favorisce 3); più basso = più facile (favorisce 4). |

### 2.4 API del modulo (uso diretto / test)

La funzione pubblica è:

```python
from sft_cv.decision_system.decision.perimeter import localize_corners

corners = localize_corners(points, covs, spec, cfg)
```

- `points`: `np.ndarray (N, 2)` — coordinate delle detection in ENU metrico.
- `covs`: `np.ndarray (N, 2, 2)` — covarianze di geoloc, oppure `None` per forzare
  la mediana.
- `spec`: conteggio atteso — `int` (noto) oppure `[min, max]` (incerto).
- `cfg`: `dict` con gli override dei parametri sopra (le chiavi mancanti usano i
  default di `perimeter.DEFAULTS`).
- **ritorna**: `np.ndarray (k, 2)` — i vertici in ENU, ordinati per support. Può
  contenere meno di `min` righe (con warning) se il detector ha mancato dei GCP;
  mai più di `max`; mai vertici inventati.

Esempio minimo:

```python
import numpy as np
from sft_cv.decision_system.decision.perimeter import localize_corners

pts = np.array([...])                      # (N, 2) ENU
covs = np.tile(np.eye(2), (len(pts), 1, 1))
cfg = {"method": "dbscan", "eps_m": 4.0, "min_samples": 2}

corners = localize_corners(pts, covs, [3, 4], cfg)
```

### 2.5 Test

I test sono in `tests/test_perimeter.py` e coprono: conteggio noto 3/4, vertice
mancante senza fabbricazione, decisione 3-vs-4 nel caso incerto, reiezione di un
cluster falso interno all'area, accettazione di un 4° vertice più debole ma
geometricamente valido, sopravvivenza di un GCP visto poche volte, solo-rumore →
vuoto, input vuoto, modalità `median`, determinismo.

Esecuzione:

```
PYTHONPATH=src python -m pytest tests/test_perimeter.py -q
```
