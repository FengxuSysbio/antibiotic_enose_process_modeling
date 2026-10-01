# Continuous electronic-nose modeling for antibiotic fermentation

Publication-oriented code repository for the manuscript workflow covering:

1. **Fermentation-stage classification and early deviation detection**
   - PCA / t-SNE visualization
   - Random Forest, RBF-SVM, MLP and LSTM comparators
   - TSFC-Net with a time–sensor feature-attention module (TSFAM)
   - Ablation utilities
   - Mahalanobis-distance trajectory deviation detection
2. **Quantitative soft sensing**
   - Ridge, Lasso, ElasticNet, Random Forest, GBDT, SVR and KNN regression
   - LSTM, BiLSTM and BiGRU sequence regressors
   - Fermentation-kinetic reconstruction utilities
   - Independent-batch validation helpers
3. **Cross-antibiotic few-shot adaptation**
   - Gentamicin-derived supervised PLS latent representation
   - Progress-aware target calibration-point selection
   - Dynamic target baseline + source-representation residual correction
   - Repeated outer-holdout learning curves at 20/40/60/80/100% of the target calibration pool

## Repository structure

```text
.
├── configs/                    # experiment templates
├── data/                       # raw data are intentionally not distributed
├── docs/                       # manuscript mapping and evidence boundaries
├── notebooks/                  # clean entry-point notebooks
├── scripts/                    # command-line workflows
├── src/antibiotic_enose/       # reusable Python package
├── tests/                      # light unit/smoke tests
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

PyTorch is included in the core requirements because TSFC-Net uses it. TensorFlow is optional and is only required for the LSTM/BiLSTM/BiGRU modules:

```bash
pip install -r requirements-tensorflow.txt
```

Python 3.10–3.12 is recommended when TensorFlow-based models are required.

## Quick start

### 1. Stage classification

Prepare one file per fermentation batch under `data/raw/result2/` with columns
`Time_h`, `Stage`, `Channel_1` ... `Channel_16`; optionally add `Is_abnormal`.

```bash
python scripts/run_classification.py \
  --data-dir data/raw/result2 \
  --output-dir outputs/result2 \
  --window 60 \
  --epochs 200 \
  --batch-size 32 \
  --learning-rate 0.001
```

### 2. Soft-sensor regression

Place paired batch files under `data/raw/result3/`. Required columns are
`Time_h`, `Channel_1` ... `Channel_16`, `PMV`, `RS`, `TS`, `NH4`, `Titer`.

```bash
python scripts/run_regression.py \
  --data-dir data/raw/result3 \
  --external-batch BATCH_TO_HOLD_OUT \
  --output-dir outputs/result3

# RNN soft sensors (requires TensorFlow)
python scripts/run_rnn_regression.py \
  --data-dir data/raw/result3 \
  --external-batch BATCH_TO_HOLD_OUT \
  --output-dir outputs/result3_rnn
```

### 3. Few-shot cross-antibiotic adaptation

Prepare processed CSV files described in `data/README.md`, then run:

```bash
python scripts/run_transfer.py \
  --data-dir data/processed/result7 \
  --output-dir outputs/result7
```

If the original Excel files are available, `scripts/prepare_transfer_data.py` reproduces the
source/target ingestion and time alignment used by the current Result 7 workflow.

## Reproducibility boundaries

This repository deliberately preserves the evidential boundaries of the study:

- Stage classification used a **sample-level 7:3 split**, not strict leave-one-batch-out validation.
- The abnormal-process analysis contains **one real contaminated batch** and should be treated as proof-of-concept.
- The 1,000 high-density kinetic points are **model-generated temporal augmentation**, not additional experiments.
- Cross-antibiotic analysis contains predominantly one continuous electronic-nose trajectory per target antibiotic; system and run effects are therefore partly confounded.
- Result 7 demonstrates **few-shot process-state adaptation**, not independent multibatch cross-product generalization.
- No function in this repository alters measured values or predictions to force a desired R² range.

## Important note on TSFC-Net reproducibility

The network architecture implemented here follows the manuscript Methods description and the archived analysis workflow. The exact historical TSFC-Net optimizer/hyperparameter record was not fully preserved in the source materials. Therefore the command-line script requires explicit training parameters and the final values used for the publication should be archived together with the accepted manuscript. See `docs/EVIDENCE_BOUNDARIES.md`.

## Additional entry points

- `scripts/run_classification_lstm.py`: LSTM stage-classification comparator.
- `scripts/run_classification_ablation.py`: temporal/sensor and GAP/GMP ablation workflow.
- `scripts/run_kinetic_model.py`: model-constrained 1,000-point temporal reconstruction.

## Citation

A `CITATION.cff` template is included. Replace the placeholder manuscript metadata before making the repository public.
