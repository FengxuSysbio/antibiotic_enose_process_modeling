# Evidence and reproducibility boundaries

## Classification

The reported stage-classification experiment used a sample-level 7:3 train/test split. It was not a strict leave-one-batch-out design. The code exposes the split explicitly so that the publication statement remains reproducible.

The TSFC-Net architecture is reconstructed from the manuscript Methods and archived analysis notebook. The exact optimizer / learning-rate / batch-size combination used in the original reported 99.7% run must be confirmed from the final experiment log before repository release. The CLI therefore exposes these parameters explicitly instead of hiding them in the model class.

## Kinetic augmentation

Dense kinetic trajectories are generated from fitted process models and should never be counted as independent biological observations. External performance metrics must use experimentally measured offline values only.

## Cross-antibiotic adaptation

The current target-domain experiments are few-shot adaptation analyses within the available continuous target trajectories. They are not independent multibatch generalization tests. Each target antibiotic is represented predominantly by one electronic-nose run.

The primary current Result 7 algorithm is:

1. six real gentamicin paired batches → supervised PLS latent representation;
2. approximately 25% of target labels retained as outer test data;
3. 20/40/60/80/100% of the remaining calibration pool selected using relative progress + source-latent PC1 coverage;
4. a target-specific time trajectory is interpolated from labeled target points;
5. gentamicin-derived latent features correct only the residual error using Ridge regression (`alpha=10`, shrinkage `0.10`);
6. results are summarized over 100 repeated outer splits.

This design intentionally prevents direct transfer of absolute gentamicin concentration mappings into a different antibiotic system.
