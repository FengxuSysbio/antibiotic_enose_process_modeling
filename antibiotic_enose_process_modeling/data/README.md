# Data layout

Raw industrial fermentation data are intentionally excluded from the public repository.

## Result 2: stage classification

One CSV/XLSX file per batch under `data/raw/result2/`:

```text
Time_h, Stage, Is_abnormal, Channel_1, ..., Channel_16
```

- `Stage`: integer class labels `0,1,2` (or another consistently encoded three-stage scheme).
- `Is_abnormal`: optional binary indicator.
- `Batch` is optional; if absent the filename stem is used.

## Result 3: soft sensing

One CSV/XLSX file per paired batch under `data/raw/result3/`:

```text
Time_h, Channel_1, ..., Channel_16, PMV, RS, TS, NH4, Titer
```

`Batch` is optional and will otherwise be inferred from the filename.

## Result 7: processed transfer-learning data

Place the following files in `data/processed/result7/`:

```text
Gentamicin_REAL_source.csv
Erythromycin_REAL_aligned.csv
Cephalosporin_REAL_aligned.csv
Lincomycin_REAL_aligned.csv
```

`Gentamicin_REAL_source.csv` columns:

```text
Batch, Channel_1 ... Channel_16, PMV, RS, TS, NH4, Titer
```

Target files contain:

```text
Time_h, Channel_1 ... Channel_16, <available target variables>
```

Expected target variables:

- Erythromycin: `DCW, Titer, RS, NH4`
- Cephalosporin: `DCW, Titer, RS`
- Lincomycin: `PMV, Titer, RS, NH4`

The optional `scripts/prepare_transfer_data.py` can construct these processed files from the original Excel files used in the current analysis.
