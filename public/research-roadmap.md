# QEDNET Research Features and Evidence Gaps

## Current scope

QEDNET is a research and education platform for evaluating classical and quantum-inspired machine-learning methods on biomedical benchmark datasets. Its current outputs are exploratory model comparisons. The platform does not establish early detection, clinical utility, patient benefit, or quantum advantage.

## Work made visible in the dashboard

- A benchmark summary with saved AUC mean and standard deviation, sample composition, completed-fold counts, simulator details, and known caveats.
- Downloadable report, fold-level comparison table, classical and quantum raw records, Parkinson's grouped-CV output, MLP diagnostic output, and QSVM timing records.
- An explicit distinction between this saved local benchmark and experiments run by the current website deployment.

## Research weaknesses and concrete next work

### Validation integrity

- Parkinson's voice rows from the same subject occur in both training and test folds in the 80-row random-fold benchmark. Treat those quantum and classical rows as leakage-affected exploratory evidence. Re-run the quantum comparison with subject-grouped folds before using it to compare Parkinson's models.
- The 80-row grid uses one fixed subsample and one three-fold split set. Repeat the complete paired comparison across independently fixed seeds, retaining a shared split manifest for all models.
- The timeout runner exceeded its configured limit for a quantum fold. Repair and verify parent-enforced termination before any long grid; preserve checkpointing and timeout status per fold.
- Add and run the parameter-matched MLP on the same samples and folds before claiming that the quantum comparison is parameter-fair.

### Generalization and early detection

- Full-dataset quantum results are not measured. The small samples do not establish population generalization.
- Early detection is not measured. It requires longitudinal data with a defensible prediction time point before the outcome and a split that prevents subject, site, and temporal leakage.
- External or prospective validation is not measured. Secure authorized cohorts and an independent research partner before making transportability claims.
- For every dataset, publish its provenance, license, target definition, inclusion rules, missingness, duplicate policy, and population limitations.

### Reliability and uncertainty

- The small-fold MLP with early stopping is unstable in the saved benchmark. Preserve the historical result and compare a predeclared alternative on the same training-only folds; do not replace the historical result silently.
- Quantum calibration and conformal prediction are not measured. Add held-out calibration and conformal evaluation only with the calibration and test partitions kept separate; report coverage, abstention, and performance on kept and all samples.
- Explainability outputs need model-specific fidelity checks and clear limits. SHAP and other explanations do not demonstrate causal effects.

### Quantum evidence

- Current quantum rows are PennyLane `default.qubit` analytic simulations. Real-hardware runs and finite-shot noise are not measured.
- Record device, shots, qubits, feature map, circuit depth, trainable parameters, per-fold fit/prediction time, and failures. Compare against parameter-matched classical models under identical data and splits.
- Do not claim quantum advantage unless a preregistered, resource-aware study supports it and an independent reproduction confirms it.

### Human-health scope

- The current public benchmark tables are tabular and do not provide validated whole-body, menstrual, wearable, imaging, genomic, EHR, or surgical-outcome capabilities.
- Add each health domain as a research module only after identifying suitable authorized datasets, defining a narrow research question, documenting consent and data rights, and specifying subgroup and external validation.
- Keep all outputs research-only. Clinical decision support requires a separate clinical, privacy, security, human-factors, and regulatory program.

## Completion gates

- Every displayed metric links to its source artifact, dataset version, split manifest, seed, and model configuration.
- Leakage checks pass for the declared group and time boundaries.
- All planned folds have a recorded terminal status; timeout enforcement is verified independently.
- The same held-out partitions are used for all compared models, and incomplete runs remain visible.
- Claims about generalization, early detection, calibration, conformal coverage, clinical value, and hardware are withheld until each has direct supporting evidence.
