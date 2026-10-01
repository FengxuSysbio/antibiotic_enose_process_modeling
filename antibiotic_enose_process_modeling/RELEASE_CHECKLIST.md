# Pre-publication GitHub checklist

- [ ] Replace placeholder metadata in `CITATION.cff`.
- [ ] Confirm final TSFC-Net optimizer, learning rate, epochs, batch size, and random seed from the archived experiment log.
- [ ] Verify the regression RNN implementation against the final training scripts.
- [ ] Confirm the kinetic ODE solver / optimizer actually used in the published analysis.
- [ ] Do not upload restricted industrial raw data without authorization.
- [ ] Add DOI / repository links used in the manuscript Data and code availability statement.
- [ ] Freeze a tagged release matching the accepted manuscript, e.g. `v1.0.0`.
- [ ] Record package versions (`pip freeze > environment-lock.txt`) on the machine used for final reproduction.
