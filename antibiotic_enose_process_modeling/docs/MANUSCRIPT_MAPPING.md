# Mapping between manuscript analyses and repository code

| Manuscript analysis | Code |
|---|---|
| Stage-space PCA/t-SNE | `classification/features.py`, `scripts/run_classification.py` |
| RF / RBF-SVM / MLP stage classifiers | `classification/baselines.py` |
| LSTM stage comparator | `classification/lstm.py` |
| TSFC-Net / TSFAM | `classification/tsfcnet.py` |
| Ablation analysis | `classification/tsfcnet.py` (`TSFCAblationNet`) |
| Mahalanobis process-deviation analysis | `classification/anomaly.py` |
| Conventional regression benchmark | `regression/baselines.py` |
| LSTM / BiLSTM / BiGRU soft sensors | `regression/rnn.py` |
| Kinetic reconstruction | `regression/kinetics.py` |
| External-batch validation | `scripts/run_regression.py` |
| Gentamicin source latent representation | `transfer/representation.py` |
| Sparse target selection | `transfer/fewshot.py` |
| Dynamic target baseline | `transfer/fewshot.py` |
| Source-latent residual correction | `transfer/fewshot.py` |
| 20–100% repeated outer-holdout learning curves | `transfer/evaluation.py` |
| Raw Excel ingestion/alignment for Result 7 | `scripts/prepare_transfer_data.py` |
