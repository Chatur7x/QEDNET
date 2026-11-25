# QEDNET Product Update Plan

## Product boundary

This repository and its hosted service are the **research platform**. They support research and education with permitted, de-identified data. They are not a medical device and must not be used to diagnose, triage, recommend treatment, or make patient-care decisions.

The **clinical decision-support product** is a separate product and release program. It needs separate source control, identity and access rules, data stores, audit records, risk controls, release process, user-facing intended-use statement, and validation. Do not copy a QEDNET model or login into a clinical workflow and treat that as validation.

This plan describes proposed work. A roadmap item is not evidence that the feature exists or is verified.

## Current baseline

- QEDNET has a Python research core, FastAPI service, and Next.js dashboard.
- Quantum models use simulators in the described setup; real hardware results are not established.
- The hosted Render service is a private research preview using filesystem-demo storage. Durable experiment storage and durable worker execution are not configured there.
- Supabase email/password authentication and a single-email allowlist are configured for the preview. The confirmation-resend routing fix is deployed, but successful email delivery and account confirmation remain unverified because Supabase rate-limited confirmation messages.
- The clinical product is not implemented or validated.
- A separate Android client is not verified as present. A future client should use the research API only and must not present outputs as clinical advice.

## Track A — Research platform

### A0. Safety and release boundaries

- Keep research-only disclaimers on the site, API responses, reports, exported artifacts, and model cards.
- Remove stale or unsupported product claims from the README and dashboards.
- Establish permitted-data rules: public or approved, de-identified data only; document source, license, cohort, consent/governance basis, retention, and deletion expectations.
- Add a versioned release checklist with dependency/image scanning, secret scanning, software bill of materials, rollback instructions, and a clean-checkout verification run.

**Done when:** every user-facing and machine-readable output identifies research status and provenance; release evidence is attached to a tagged build; unresolved checks are marked not measured or blocked.

### A1. Dataset provenance and leakage controls

- Add a dataset card for every adapter and upload: source and license, cohort and target definition, inclusion/exclusion rules, sample/feature schema, missingness, collection context, known limitations, and cryptographic content fingerprint where permitted.
- Require an explicit split strategy for subject-, site-, or time-dependent data. Preserve group/time identifiers for splitting, then exclude identifiers from model features.
- Save split membership as safe row hashes or stable record keys so runs can be audited without copying direct identifiers into reports.
- Add leakage checks for duplicate records, subject overlap, site overlap, temporal overlap, target leakage, and preprocessing fitted outside the training fold.
- Reject or quarantine datasets whose required provenance or split fields are absent.

**Done when:** deliberate leakage fixtures fail the run; grouped/site/time splits have tests; each report links to dataset fingerprint and exact split manifest.

### A2. Reproducible benchmark engine

- Run simple and strong classical baselines before expensive quantum models.
- Enforce the same data rows, split indices, fold-local transforms, search budget, metric definitions, and compute budget for all compared models.
- Record random seeds, software versions, hardware, config, fit/predict times, memory where measured, failures, timeouts, and exact model settings.
- Checkpoint each completed fold atomically so a crash does not erase finished work; support safe resume without duplicating records.
- Preserve per-fold outputs and the exact command/configuration that produced every aggregate.
- Keep negative, failed, and incomplete results visible. Never substitute simulated or placeholder results for a missing run.

**Done when:** a clean checkout can reproduce the same run protocol from a frozen manifest; automated consistency checks rebuild summary tables from raw fold records.

### A3. Evidence ledger and research claims

- Link each chart, table, certificate, and report claim to raw fold-level evidence, split manifest, config hash, and run ID.
- Distinguish exploratory, internal validation, external validation, and prospective evidence in the UI and exports.
- Define the quantum-advantage criteria before running a comparison. Require matched classical kernel, random-feature, parameter/capacity, and resource baselines on identical folds.
- Make certificates fail closed when evidence, sample counts, protocol details, or uncertainty estimates are missing.
- Show per-fold results, confidence intervals, uncertainty, sample counts, and failed runs alongside means.

**Done when:** every quantitative claim is traceable and certificates cannot assert an advantage from incomplete or mismatched evidence.

### A4. Calibration, uncertainty, and robustness research

- Evaluate calibration on a separate held-out calibration set or using a prespecified cross-fitting protocol; report Brier score, calibration error, and reliability plots with uncertainty.
- Evaluate conformal coverage and abstention on held-out data; report target and measured coverage, abstention, and accuracy on kept versus all cases.
- Add subgroup reporting with small-count suppression and uncertainty intervals; do not treat one aggregate fairness score as proof of fairness.
- Add missingness, measurement-noise, preprocessing perturbation, and relevant distribution-shift tests.
- Add out-of-distribution detection and explicit “outside validated data” states. Do not imply that a detector guarantees safety.
- Explore privacy-preserving collaboration such as federated evaluation only after threat modeling, governance, and data-owner approval.

**Done when:** protocols and thresholds are frozen before evaluation, and all measures are reported on data not used to fit the evaluated model or thresholds.

### A5. Research service operations and access control

- Configure durable PostgreSQL, object storage, and a durable queue before retaining user uploads or experiment results.
- Route dashboard jobs through the durable worker API; remove or disable detached per-request child processes in hosted mode.
- Enforce ownership checks for datasets, jobs, runs, reports, certificates, and artifacts at every read/write boundary.
- Add request and upload limits, rate limits, quotas, audit events, timeouts, concurrency controls, safe errors, and deletion/retention workflows.
- Treat pickle files as executable trusted content. Restrict access and move toward a versioned, validated artifact format with integrity checks.
- Verify database and object-store backups, restore procedure, worker restart behavior, monitoring, alerting, and rollback.
- Keep free-tier hosting limited to disposable, non-sensitive preview use if it cannot provide these guarantees.

**Done when:** isolation, authorization, recovery, queue behavior, and operational controls pass integration tests in the actual deployment environment.

### A6. External validation and distribution shift

- Add site-held-out and temporal evaluation protocols where dataset design supports them.
- Report performance, calibration, missingness, and subgroup behavior by site/time period.
- Monitor covariate and label shift in research datasets; present alerts as investigation prompts, not clinical assurances.
- Require independent reproduction before publishing broad performance claims.

**Done when:** each claimed deployment domain has representative external evidence and limitations are stated in the report.

### A7. Website and Android client

- Keep the public information website separate from the authenticated research workspace.
- Publish methods, dataset provenance, model cards, simulator/hardware status, limitations, and links to reproducibility bundles.
- Build an Android client only after the API contract, authentication lifecycle, privacy behavior, offline/error states, and supported research workflows are stable.
- Use the Android client for research/education workflows only. Do not display patient-specific diagnosis or treatment language.

**Done when:** website content is evidence-linked and the Android client passes API, security, accessibility, and device tests against the supported research service.

## Track B — Separate clinical decision-support product

### B0. Clinical discovery and intended use

- Select a narrowly defined clinical problem, intended user, care setting, population, decision point, and action supported.
- Identify clinical owners, data-governance authority, patient-safety lead, security/privacy lead, quality lead, and regulatory counsel.
- Define foreseeable harms, failure modes, escalation behavior, human oversight, contraindications, and when the system must abstain.
- Decide whether the product is informational, prioritization/triage, or recommendation support before selecting a model.

**Gate:** no patient-facing recommendation or clinical integration until clinical, privacy, security, quality, and regulatory owners approve the intended-use plan.

### B1. Data and validation program

- Establish lawful data access, consent/governance, minimization, de-identification, retention, access logging, and breach response.
- Define an analysis plan before model development: reference standard, cohort, exclusions, endpoints, subgroups, missing data, and statistical methods.
- Use representative multi-site data and independent external validation; assess calibration, subgroup performance, shift, and clinically relevant error costs.
- Run prospective workflow and human-factors studies. Measure whether use improves outcomes or workflow and whether it creates automation bias, delay, or excess alerts.
- Validate any transferred research method from scratch in the clinical product's intended population and workflow.

**Gate:** no clinical performance claim without independently reviewed, representative validation and uncertainty estimates.

### B2. Clinical software and operations

- Build a separate application, repository, deployment, identity/access policy, data store, audit trail, and model release process.
- Add risk management, traceability, change control, test evidence, monitoring, incident handling, rollback, and post-release surveillance.
- Threat-model clinical data, enforce least privilege, encryption, key rotation, secure updates, backup/restore, and penetration testing.
- Determine applicable regulatory obligations in each intended jurisdiction with qualified counsel and regulators.

**Gate:** do not launch for patient care until required quality, security, clinical, and regulatory gates are satisfied.

## Cross-track rules

- Research results are hypotheses for follow-up, not clinical evidence by themselves.
- Shared code or branding does not transfer validation, authorization, or safety claims.
- Do not use real patient data in the current QEDNET Render preview.
- Do not claim quantum advantage, clinical benefit, diagnostic accuracy, or production readiness unless the relevant prespecified evidence and independent review exist.
- When a requirement cannot be measured or verified, say **NOT MEASURED** and give the reason.

## Immediate execution order

1. Correct stale documentation and stabilize the current private research preview, including confirmation email delivery and account recovery.
2. Freeze research dataset provenance, split manifests, and reproducibility requirements.
3. Add evidence-ledger and leakage checks before running additional headline benchmarks.
4. Build and integration-test durable storage, queue execution, ownership checks, and recovery before enabling retained uploads or multi-user access.
5. Add calibration, conformal, subgroup, shift, and robustness evaluations with prespecified held-out protocols.
6. Publish research documentation and reproducible bundles; then stabilize the website and Android research client against the API.
7. Start the clinical track with intended-use, clinical governance, and regulatory discovery. Implement clinical software only after those gates are staffed and approved.

No completion date is stated: delivery depends on infrastructure, data access, clinical partners, and review inputs not currently available in this repository.
