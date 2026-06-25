# `datasets` — quantification dataset loaders (scikit-learn style)

A small **package** laid out like mlquantify: thematic private modules with shared helpers, re-exported
from `__init__.py`. Each loader downloads the **official/stable file** with the standard library
(`urllib`), caches it under `_data/<dataset>/`, and parses it -- the same way scikit-learn's `fetch_*`
functions work.

```python
from datasets import fetch_mushroom              # flat re-export (like sklearn.datasets)
from datasets._tabular import fetch_mushroom     # or the mlquantify-style submodule path
```

### Package layout (mirrors mlquantify)
```
datasets/
  __init__.py     # re-exports every fetch_* + Bunch / get_data_home / make_protocol  (+ __all__)
  _base.py        # Bunch, get_data_home, fetch_remote (urllib+cache), finish_tabular / finish_xy
  _protocol.py    # mlquantify bridge: make_protocol / run_protocol (APP/NPP/UPP/PPP)
  _tabular.py     # UCI static-CSV datasets + stream mirrors (15 loaders)
  _text.py        # newsgroups20, imdb, multidomain_sentiment, sentiment140, rcv1_v2
  _image.py       # mnist_usps, cifar10
  _graph.py       # planetoid_cora_citeseer_pubmed
  _synthetic.py   # sea_concepts
  _lequa.py       # lequa2024 (task='T1'..'T4')
```

No quapy / ucimlrepo / kagglehub / torchvision / torch-geometric / river (mlquantify only for `protocol=`).

## Common signature
```python
fetch_<name>(*, data_home=None, download_if_missing=True, return_X_y=False, as_frame=False,
             n_retries=3, delay=1.0,
             protocol=None, n_samples=1000, sample_size=500, random_state=None)
```
- Default -> a `Bunch` with `.data`, `.target`, `.feature_names`, `.target_names`, `.DESCR` (and `.frame` if `as_frame=True`).
- `return_X_y=True` -> `(X, y)`.
- **Quantification (via [mlquantify](https://github.com/luizfernandolj/mlquantify)):** set
  `protocol="app" | "npp" | "upp" | "ppp"` (or pass a configured `mlquantify.model_selection` protocol
  instance). The `Bunch` then has `.samples` (a list of index bags into `.data`), `.prevalences` (each
  bag's class distribution, from `mlquantify.utils.get_prev_from_labels`) and `.protocol` (the mlquantify
  protocol object). Mapping: `sample_size` -> protocol `batch_size`, `n_samples` -> number of prevalence
  points. `app`=Artificial, `npp`=Natural, `upp`=Uniform (Kraemer), `ppp`=Personalized (give explicit
  prevalences by passing a `PPP(...)` instance).
- A few add dataset-specific kwargs: `subset=` (20ng/imdb/cifar10/mnist_usps), `domain=` (mnist_usps,
  multidomain_sentiment), `name=` (planetoid), `task=` (lequa2024: T1/T2/T3/T4), `which=` (digits), `target_col=` (UCI).

```python
from datasets import fetch_mushroom
b = fetch_mushroom(protocol="upp", n_samples=1000, sample_size=500, random_state=0)
X_bag0 = b.data[b.samples[0]]      # one quantification bag (indices into b.data)
p_bag0 = b.prevalences[0]          # its true class prevalence
proto  = b.protocol                # the mlquantify protocol object (reuse in evaluation)
```

## Output format (scikit-learn style)

Same convention as scikit-learn -- features and label are kept SEPARATE:
- `.data` (feature matrix, no label), `.target` (label), plus `.feature_names`, `.target_names`, `.DESCR`.
- `return_X_y=True` -> `(X, y)`.
- `as_frame=True` (tabular datasets) -> `.data` is a DataFrame, `.target` a Series, and `.frame` is the
  combined table = features + a column named **`target`** (sklearn's name for the label).

```python
from datasets import fetch_mushroom
b = fetch_mushroom(as_frame=True)
b.frame.head()        # feature columns ... + a final 'target' column
b.data, b.target      # features (DataFrame) and label (Series), kept separate
X, y = fetch_mushroom(return_X_y=True)   # numpy arrays
```

Feature names are the dataset's own column names (UCI headers, SEA's f1/f2/f3, ...). The text/image/
graph datasets (IMDB, 20NG, MNIST/USPS, CIFAR, Planetoid, LeQua vectors) follow sklearn's array style
(`.data` + `.target`), like `load_digits` / `fetch_20newsgroups`. The label column is named `target`
to match sklearn; use `b.frame.rename(columns={'target': 'class'})` if you prefer `class`.

## Dependencies (only what you use)
```bash
pip install pandas numpy     # all tabular/CSV parsing
pip install mlquantify       # quantification protocols (only needed when you pass protocol=...)
pip install scipy            # planetoid graph parsing
pip install scikit-learn     # OPTIONAL: nicer RCV1 (falls back to raw figshare files)
```

## The 25 datasets
- **Tabular (UCI static CSV):** mushroom, banknote_authentication, haberman_survival, miniboone,
  digits_optical_penbased, dry_bean, covertype, yeast, sensorless_drive, statlog_shuttle,
  wine_quality (ordinal), online_news_popularity; plus pima_diabetes (mirror).
- **Streams / temporal:** electricity_elec2, airlines (scikit-multiflow mirrors), sentiment140 (timestamped), sea_concepts (generated, concept drift).
- **Text:** newsgroups20, imdb, multidomain_sentiment (covariate/domain shift), rcv1_v2.
- **Images:** mnist_usps (covariate shift), cifar10 (label shift).
- **Graph:** planetoid_cora_citeseer_pubmed.
- **Ordinal / competition:** lequa2024 (one file; `task='T1'|'T2'|'T3'|'T4'`, official Zenodo files).

## Excluded (need manual steps or credentials)
fact_oq (Julia extraction), insects (manual link from USP Google-Sites), semeval2017_task4
(Twitter-API hydration), diabetic_retinopathy_aptos (Kaggle credentials), amazon_oq_bk
(~41.9 GB; use github.com/mirkobunse/regularized-oq scripts). Add manually if needed.

## Notes
- `.data` / `.target` plug straight into mlquantify quantifiers (`model.fit(X, y)` / `model.predict`);
  the `protocol` objects (`.protocol`, or build your own with `mlquantify.model_selection`) drive evaluation.
- TLS is verified by default; for academic hosts with a non-standard cert chain (e.g. the USPS LIBSVM
  mirror) the download auto-retries without verification and prints a warning.
- RCV1 is multilabel: pick a topic column before using a `protocol=`.
- LeQua test zips are large (set `include_test=True` to also fetch them); train_dev is downloaded by default.
