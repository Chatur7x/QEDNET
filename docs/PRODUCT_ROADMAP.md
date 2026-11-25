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
- Use separate application IDs, signing, release channels, API authorization, local storage, and privacy disclosures for the research and future clinical apps. Share code only after a security and data-boundary review.
- For Google Play distribution, prepare the required health-app declaration, privacy policy, prominent data-use disclosure, and least-privilege permission justification before submission.
- Use the Android client for research/education workflows only. Do not display patient-specific diagnosis or treatment language.

**Done when:** website content is evidence-linked and the Android client passes API, security, accessibility, and device tests against the supported research service.

### A8. Real-time Android and body-sensor research pipeline

#### Sensor and source strategy

- Treat Android Health Connect as a user-permissioned health-data import and synchronization layer, not as a guaranteed low-latency sensor stream.
- For an active, user-visible Wear OS measurement or workout, evaluate Health Services `MeasureClient` or `ExerciseClient` against the supported device and use case.
- For passive monitoring, treat Health Services updates as batched and irregular. Do not promise a fixed update interval or use passive/background Android collection as a dependable emergency-alert channel.
- Add one hardware family at a time, starting with synthetic replay and documented device metadata. Do not infer device equivalence from matching field names.
- Collect only data types needed for an explicitly described research feature; ask for granular permissions at the point of need and keep the app useful when permission is declined.

#### End-to-end data path

1. **Consent and permissions:** explain purpose, source, fields, retention, sharing, and deletion; record permission state and allow revocation.
2. **Capture:** store value, unit, event time, receive time, source app/device, hardware/firmware metadata when available, sampling context, and session identity.
3. **Quality gate:** validate units and ranges, timestamps, duplicates, signal quality, wear time, calibration metadata, missingness, and device changes. Mark invalid or stale windows unavailable.
4. **On-device queue:** minimize and pseudonymize records, protect local state, avoid health details in logs/notifications, and queue safely during offline periods.
5. **Sync:** use authenticated, encrypted, idempotent uploads with retry/backoff, ordering/watermarks, deduplication, and deletion/revocation handling. Never label queued or delayed data as live.
6. **Server ingestion:** enforce identity/ownership, schema and payload limits, consent scope, device/session policy, audit events, and abuse/resource limits before accepting data.
7. **Processing:** use a durable queue and versioned pipeline to validate windows, extract features, run research models, record model/preprocessing versions, and preserve failure states.
8. **Research feedback:** display timestamp and freshness, signal quality, missing inputs, model/version, uncertainty, and an explicit research-only status. No diagnosis, treatment, emergency response, or automated clinical action.
9. **Operations:** monitor delivery lag, dropped/duplicate records, queue backlog, processing failures, device disconnects, battery/network behavior, and deletion propagation without exposing sensitive values in telemetry.

#### Real-time claim and completion tests

- Define a latency objective only after the sensor, use case, data route, and safety class are selected. The objective is **NOT SET** in this plan.
- Measure capture-to-display latency distributions, missing/dropped/duplicate data, stale-value behavior, reconnection, permission revocation, offline replay, clock skew, battery use, and server/queue outages on named devices.
- Test with synthetic playback first, then approved de-identified recordings, then a consented usability study if governance permits.
- Test the actual Android OS versions and hardware on which the feature will be supported; emulators alone do not establish wearable sensor behavior.
- On every failure, show unavailable/stale status and preserve user control. Never silently reuse old readings as current.

**Done when:** an end-to-end validation report proves the stated behavior and measured latency on the documented device/configuration matrix; any unsupported device or background mode is clearly unavailable.

**Safety boundary:** this is a research/education flow. It is not for emergency monitoring, diagnosis, or care decisions. Android Health Connect background access requires appropriate user permission, and its synchronization behavior is not a real-time guarantee. Wear OS passive updates can be batched at intervals that the app cannot predict; rapid measurement APIs are intended for active experiences, not continuous background capture.

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

### B3. Clinical mobile and real-time workflow

- Select approved, fit-for-purpose clinical devices or regulated data sources for the intended use. Do not use Health Connect or consumer passive wearable sync for a clinical function where failure could reasonably cause harm; Google Play's Health Connect policy excludes such uses. Confirm the exact policy and regulatory implications for the intended product with Google Play and qualified counsel.
- Build explicit source validation, patient/device matching, timestamp/freshness checks, signal-quality gates, and missing/stale-data behavior before inference.
- For EHR workflow integration, evaluate standards-based FHIR APIs and CDS Hooks with the target health system. Treat FHIR Subscription notifications as triggers to reconcile state: the specification allows missed or duplicate notifications, so implement durable event IDs, idempotency, retry, replay/reconciliation, authorization checks, and a visible failure state.
- Define the complete human workflow: who receives an alert, what context is shown, how a qualified user reviews it, how acknowledgement and escalation work, and what fallback exists when the device, network, service, or model is unavailable.
- Prevent silent failures and duplicate or alert-fatigue behavior; maintain an auditable event trail and safe downtime procedures.
- Evaluate latency and reliability end to end under real device, connectivity, and operating conditions. Set acceptance criteria before collecting validation results; do not invent service levels in advance of the use case.
- Conduct prospective and human-factors validation for the actual mobile workflow. Validate alert timing, comprehension, workload, false alarms, missed events, and downstream clinical impact.
- Assess app-store policies, consent, disclosures, health-data permissions, privacy, and medical-device regulatory obligations for each target market before distribution.

**Gate:** no real-time clinical alert or patient-care reliance until the device pathway, end-to-end reliability, clinical workflow, independent validation, risk controls, and market-specific regulatory review are approved.

### B4. Advanced and future-facing clinical capabilities

These are research and product hypotheses, not currently implemented or validated capabilities. Build only after intended use, data rights, and risk controls are established. Start each capability in offline or shadow evaluation; do not expose experimental outputs to patient care.

- **Multimodal trajectory modeling:** combine approved longitudinal records, waveforms, labs, medication events, and patient-reported measures with explicit time alignment, provenance, missingness handling, and modality-specific quality gates. Compare against simpler single-modality baselines and test transfer across sites and devices.
- **Personalized baselines:** investigate within-person change detection and calibrated forecasts. Separate stable population risk from personal deviation; test baseline drift, device changes, and sparse histories. Do not let the model silently update itself from new patient data.
- **Clinician-reviewed adaptive updates:** permit only versioned, tested model releases with predefined monitoring, change-control boundaries, revalidation, approval, and rollback. Any predetermined change-control plan must be evaluated with regulatory experts for the target jurisdiction; it is not permission for uncontrolled online learning.
- **Edge/cloud split:** investigate on-device quality checks and low-risk feature extraction to reduce raw-data transfer, while keeping model versions, uncertainty, audit records, security updates, and cloud/offline behavior controlled. Measure battery, latency, data loss, and performance parity on actual devices.
- **Federated, multi-site learning and evaluation:** start with federated evaluation of frozen models. If training is later explored, test secure aggregation, privacy leakage, site bias, contribution failures, and utility loss. Federated processing alone does not prove privacy or fairness.
- **Digital physiology models / digital twins:** treat patient-specific simulation and counterfactual trajectories as exploratory research only. Validate simulator assumptions and uncertainty against prospective measurements; never present simulated futures as established patient outcomes.
- **Quantum-assisted components:** compare quantum kernels, variational circuits, and hybrid pipelines against strong matched classical controls on identical patient/site/time splits. Require hardware provenance and measured end-to-end costs. No quantum method receives clinical status or a superiority claim without independent evidence for the specific intended use.
- **Human-feedback and learning loop:** collect clinician accept/ignore/override and outcome signals only under explicit governance. Use them for safety monitoring and prespecified studies; do not feed them directly into a live model without a controlled, validated release process.
- **Auditable evidence view:** show the data source and freshness, relevant validated population, model and preprocessing version, uncertainty, known exclusions, evidence tier, and reason for abstention in the clinician workflow. Do not imply that an explanation proves correctness or causality.

**Gate:** each capability needs its own intended-use fit, risk analysis, benchmark protocol, external validation, failure-mode testing, human-factors evaluation, security review, and jurisdiction-specific regulatory assessment before clinical use.

### B5. Clinical end-to-end acceptance path

1. **Define intended use:** identify condition, population, setting, user, decision point, output, and supported action; document prohibited uses and foreseeable misuse.
2. **Approve source and governance:** obtain clinical partner, data authority, privacy/security review, and permission for each data source and device. Use synthetic data for early integration.
3. **Connect source safely:** authenticate and authorize device/EHR access; verify patient/device association; validate schema, units, provenance, freshness, and consent before processing.
4. **Process with bounded failure:** invoke a locked model version, enforce input/output contracts and time limits, record the event idempotently, and return unavailable/abstain when inputs or dependencies fail.
5. **Present for human review:** show concise evidence, context, uncertainty, limitations, and source timestamps; require an appropriately qualified user to decide and record review/override. No autonomous diagnosis, treatment, or medication changes.
6. **Operate alerts safely:** define ownership, acknowledgement, escalation, duplicate suppression, downtime paths, and contact procedures with the clinical partner. An unacknowledged software notification must not be treated as care delivered.
7. **Audit and monitor:** retain access-controlled records of source data references, model version, output, uncertainty, user action, alert delivery, failures, and relevant performance metrics under an approved retention policy.
8. **Validate progressively:** test synthetic replay and hardware-in-loop; perform retrospective validation; run silent/shadow evaluation; then conduct approved prospective workflow studies. Move to clinical use only after all clinical, quality, security, and regulatory gates are signed off.
9. **Maintain and recover:** test backup/restore, incident response, device replacement, revoked access, model rollback, software patching, and safe operation during network or service outages.

**Acceptance criteria:** choose latency, availability, data-loss, alert-burden, subgroup, and clinical-outcome thresholds with clinical and safety owners before validation. They are **NOT SET** in this plan because intended use and data/device sources are undecided.

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

## Official platform and regulatory references

- Android Health Connect read permissions and background access: [Read raw data](https://developer.android.com/health-and-fitness/health-connect/read-data).
- Wear OS collection modes and timing behavior: [Health Services](https://developer.android.com/health-and-fitness/health-services) and [device compatibility](https://developer.android.com/health-and-fitness/health-services/compatibility).
- Google Play requirements for health apps and Health Connect data: [Health apps policy](https://support.google.com/googleplay/android-developer/answer/18265346) and [health apps declaration](https://support.google.com/googleplay/android-developer/answer/14738291).
- EHR event triggers: [HL7 FHIR R5 Subscriptions](https://hl7.org/fhir/R5/subscriptions.html) (notifications may be lost or duplicated; design for reconciliation) and [HL7 CDS Hooks](https://cds-hooks.hl7.org/) (workflow-triggered decision-support calls; standard is Trial Use).
- India medical-device software planning reference: [CDSCO Guidance on Medical Device Software under MDR-2017](https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-document-on-Medical-Device-Software-under-MDR-2017.pdf). Product classification and obligations depend on intended purpose and require qualified regulatory review.
- United States clinical decision-support planning reference: [FDA Clinical Decision Support Software Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software). Applicability depends on the product function and intended use; obtain jurisdiction-specific regulatory advice.
- AI model update/change-control and cybersecurity planning: [FDA Predetermined Change Control Plans](https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/predetermined-change-control-plans) and [FDA medical-device cybersecurity guidance](https://www.fda.gov/medical-devices/medical-device-cybersecurity).
