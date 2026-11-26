# QEDNET: problem-statement product and research plan

For the full early-detection workflow, feature requirements, evaluation design, architecture, and evidence references, see [EARLY_DETECTION_PRODUCT_SPEC.md](EARLY_DETECTION_PRODUCT_SPEC.md).

For the whole-body product scope, women's-health and cycle research, clinician/surgeon boundaries, quantum/ML evaluation plan, and research-production gates, see [WHOLE_BODY_HEALTH_RESEARCH_PLAN.md](WHOLE_BODY_HEALTH_RESEARCH_PLAN.md). These are roadmap requirements, not current capabilities.

## Product definition

QEDNET is a whole-body health research platform direction for testing classical and hybrid quantum-classical machine-learning methods against strong classical baselines on narrowly defined biomedical research questions. It includes a dedicated women's-health research track, with a private cycle journal as a possible first product feature and cycle modeling only after suitable data and protocol approval. Its workflow is: ingest documented data, validate it, split it according to how observations were collected, fit preprocessing on training partitions only, train and compare models, inspect explanations and uncertainty, and export a reproducible report.

The project brief says “early disease detection.” That is a hypothesis and intended research direction, not an established capability. A model trained to classify existing diagnostic labels on cross-sectional benchmark rows does not show that it detects disease earlier, improves care, or reduces cost. Any such claim needs a prediction time, reference standard, clinically meaningful lead-time horizon, representative longitudinal cohort, and independent validation that respects patient, site, and time boundaries.

This repository is not a medical device and must not direct diagnosis, triage, or treatment. The near-term product is a research and education tool. The codebase may support future clinical research, but clinical decision support is outside the current release scope and cannot be enabled by a feature flag or a model score.

## Problem-to-product feature map

| Brief requirement | Product capability | Evidence needed for completion |
|---|---|---|
| Biomedical data ingestion | Dataset adapters and documented tabular uploads with schema, target, provenance, missingness, and validation report | Supported-format contract, data rights/source record, schema tests, safe error handling, reproducible fingerprint |
| Classical preprocessing and feature engineering | Imputation, scaling, selection, and dimensionality reduction inside each training partition | Automated fold-locality checks; transformations persisted with each model artifact |
| Hybrid quantum-classical learning | QSVM, variational classifiers, and data re-uploading as research candidates; same input rows and folds as baselines | Correct backend identity, simulator/hardware provenance, circuit and parameter metadata, resource/time reporting |
| Classical comparison | Simple interpretable baseline plus strong kernel/tree/neural baselines and capacity-matched controls | Shared splits, preprocessing, tuning budget, metrics, and uncertainty; complete per-fold records |
| Explainability | Feature-level explanation with stability and fidelity checks; distinguish model explanation from causal explanation | Explanation evaluated for repeatability, perturbation validity, and mapping through feature transformations |
| Performance and generalization | AUROC/AUPRC, sensitivity/specificity, calibration, uncertainty, subgroup and shift evaluation | Prespecified endpoint and threshold; external/temporal validation; confidence intervals and missingness accounting |
| Near-term device compatibility | Backend interface that records actual device, shots, transpilation/compilation, and execution errors | Integration tests against an accessible provider or device; simulator equivalence is not hardware evidence |
| Scalable, interpretable platform | Reproducible experiment jobs, artifacts, provenance, and bounded resource use | Load/recovery/security checks and cost/runtime measurements on declared workloads |

## Existing implementation: read as code, not as a promise

The repository contains a Python research core, a FastAPI backend, and a web dashboard. It includes adapters for the scikit-learn breast-cancer data and local Heart Disease, Parkinson’s voice, and Pima diabetes files. Current datasets are tabular, small public benchmarks with existing labels. Imaging, genomics, longitudinal EHR ingestion, clinical workflows, and real-time medical-device ingestion are not established by these adapters.

Model code includes logistic regression, random forest, RBF SVM, MLP variants, XGBoost, QSVM, VQC, data re-uploading, and a HybridQNN implementation. Treat HybridQNN according to the executed implementation and report; its presence in a registry is not evidence of a valid quantum experiment. The project’s NumPy statevector and PennyLane simulator paths are simulation. A backend field or a simulator equivalence test does not establish real quantum-hardware execution.

## High-priority correctness work

1. **Split at the unit of independence.** Preserve patient/subject, site, and time keys for splitting, exclude them from model inputs, and fail explicitly when requested grouped folds cannot be formed. Never silently fall back to random-row folds.
2. **Keep all learned transforms inside folds.** Imputation, scaling, feature selection, PCA, resampling, and tuning must be fitted on the current training partition. This applies recursively to inner tuning, calibration, and threshold selection.
3. **Repair calibration evidence.** Fit calibrators on out-of-fold training predictions, then evaluate once on untouched outer-test predictions. Report raw and calibrated Brier/ECE, calibration method, fold protocol, and failure/skip reason. Do not use in-sample fit probabilities as calibration evidence.
4. **Make the quantum execution path truthful.** Metadata must describe the backend actually used. Reject unsupported backend requests instead of echoing them as if executed. Record analytic versus finite-shot mode and separate simulated results from hardware results.
5. **Make comparisons paired and auditable.** Persist split identities, config and source versions, seeds, per-fold predictions, preprocessing metadata, runtime, and model resources. Use identical samples/splits and disclose incomplete, timed-out, or failed runs.
6. **Define early detection before modeling.** Pick the condition, population, index time, prediction horizon, outcome definition, allowable inputs, and intended intervention. Remove features measured after the index time and validate on later time periods and independent sites.

## Unsolved research and engineering problems

- **Quantum advantage is unproven.** Small feature dimensions and current simulator results do not show a practical advantage. A credible study needs a pre-registered task, strong classical kernels and neural baselines, matched inputs/capacity, noise and shot studies, hardware execution, repeated independent data splits, resource accounting, and independent reproduction.
- **Small benchmark cohorts do not establish generalization.** Row-wise cross-validation can leak repeated measurements or site-specific signals. Patient-level external cohorts and time-separated validation are needed, along with confidence intervals that reflect the independent sampling unit.
- **Early detection is not the same as diagnosis classification.** Existing disease labels often describe current status. Without longitudinal pre-diagnostic measurements and a lead-time endpoint, early detection cannot be measured.
- **Explainability can mislead.** Feature attribution is not causal evidence, may vary across correlated features and folds, and may be distorted by PCA or feature selection. Test stability and faithful behavior; show limitations next to explanations.
- **Calibration and uncertainty can fail under shift.** Calibration measured on one validation distribution may not transfer across populations, prevalence, device, or site. Monitor calibration and abstention behavior on external cohorts; do not interpret confidence as clinical certainty.
- **Biomedical data governance is part of correctness.** Real-world use requires source permissions, consent/legal basis, minimization, retention/deletion, access controls, audit, encryption, incident response, and clear separation of identifiers from research features.
- **NISQ resource constraints matter.** Circuit depth, qubit connectivity, finite shots, noise, queue time, compilation, and provider availability can dominate runtime and accuracy. Report measured resources; do not infer hardware scalability from ideal simulation.

## Delivery order and acceptance gates

### Gate A — Trustworthy research core

- Add explicit group/time split metadata and validate group disjointness.
- Ensure nested preprocessing and tuning are fold-local.
- Make calibration and operating-threshold selection cross-fitted inside training data.
- Add per-fold provenance and schema-checked report outputs.
- Make model/backend metadata describe actual execution.

**Accept when:** focused tests prove no group overlap and no fit sees held-out rows; reports can be regenerated from recorded configuration and data fingerprint; unsupported features fail clearly.

### Gate B — Useful biomedical research platform

- Version dataset cards and adapters; document label semantics, collection setting, missing data, and limitations.
- Add cohort-aware metrics, subgroup review, missingness sensitivity, and external validation workflows.
- Provide stable model cards, reports, prediction explanations, and reproducible comparison runs.
- Measure runtime and resource use on declared hardware and dataset sizes.

**Accept when:** another researcher can reproduce a run, inspect data/splits, trace each result to a versioned artifact, and distinguish internal from external evidence.

### Gate C — Early-detection evidence program

- Select one condition and a clinically meaningful prediction horizon with qualified domain partners.
- Acquire appropriately governed longitudinal data and define index time, comparator, reference standard, and censoring rules.
- Establish patient/site/time-separated development, calibration, and independent test cohorts before model selection.
- Compare against existing clinical/standard-care baselines and measure lead time, sensitivity at prespecified specificity, false alarms, calibration, and subgroup performance.

**Accept when:** the intended early-detection endpoint is prespecified and independently evaluated. Until then, describe the system as benchmark classification research.

### Gate D — Hardware and operational scale

- Implement tested provider adapters with secrets held outside source, bounded jobs, retry/cancel semantics, and durable experiment records.
- Run simulator-to-device parity checks where meaningful; record shots, noise/device, compilation, queue/execution time, and failures.
- Load-test the web/API job flow and verify access control, isolation, backups, recovery, observability, and deletion.

**Accept when:** measured hardware and operational evidence supports the exact claims made. If device credentials or access are unavailable, mark hardware results NOT MEASURED.

## Product claims

Safe today only when supported by the particular run: QEDNET is a research prototype for comparing classical and simulator-based quantum ML on supported biomedical tabular benchmarks. Avoid claims that it detects disease early, improves outcomes, has quantum advantage, is hardware-compatible in production, is clinically validated, or is production-ready until the corresponding gates have evidence.
