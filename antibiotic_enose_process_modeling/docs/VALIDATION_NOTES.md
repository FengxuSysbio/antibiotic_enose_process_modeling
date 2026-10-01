# Validation notes

The repository was checked with Python syntax compilation and smoke tests for TSFC-Net tensor dimensions and sparse calibration selection.

The Result-7 raw-data ingestion was additionally compared against the archived processed arrays. The reconstructed erythromycin, cephalosporin, lincomycin, and gentamicin source matrices matched the archived analysis arrays to numerical precision.

Using 100 repeated outer-holdout splits, the current transfer-learning implementation reproduced the manuscript learning-curve medians, including approximately:

- Erythromycin titer R²: 0.971 / 0.972 / 0.981 / 0.980 / 0.980 at 20/40/60/80/100% calibration-pool fractions.
- Cephalosporin titer R²: 0.809 / 0.966 / 0.979 / 0.984 / 0.986.
- Lincomycin titer R²: 0.948 / 0.969 / 0.974 / 0.981 / 0.985.

These checks validate the current Result-7 implementation. They do not remove the biological limitation that the target systems are predominantly represented by one continuous electronic-nose trajectory each.
