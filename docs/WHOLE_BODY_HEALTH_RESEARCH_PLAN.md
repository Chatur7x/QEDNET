# QEDNET Whole-Body Health Research Plan

## Purpose and current status

QEDNET's research direction is a whole-person biomedical research platform for studying early signals of health change with classical machine learning and carefully evaluated quantum-machine-learning candidates. Women's health, including menstrual-cycle and life-course research, is a first-class research area rather than a small add-on. The platform should also support male health research and mixed-population studies where the protocol calls for them.

This is a product and research specification. The dashboard now includes an interactive body-system atlas and an in-memory menstrual-journal prototype for fictional/demo entries. Journal entries are not sent to an API and disappear when the view is left or reloaded; this is not a persistent participant-data tracker. The underlying implementation remains a tabular research system with a breast-cancer benchmark adapter and optional local files for heart disease, Parkinson's, and diabetes. There is no Android app, wearable integration, imaging/genomics/EHR pipeline, whole-body model, or validated clinical workflow. No included benchmark establishes early detection.

QEDNET may be made operational for research, but it is not currently production-ready for sensitive health data: the local Compose stack works, while local Supabase authentication is not configured and a multi-user governed health-data service has not been established. Production readiness for research does not mean clinical readiness.

## Product principles

1. **Whole body means a portfolio of defined studies.** Do not create one opaque "health score" that mixes unrelated organs, populations, time horizons, and outcomes.
2. **People and clinicians remain in control.** The research platform may organize, validate, compare, and explain research outputs. It must not diagnose, triage, prescribe, recommend surgery, or replace a clinician.
3. **A tracker is not a diagnostic tool.** A period diary may help a person record and share their own history. Any forecast or pattern analysis is a separate model hypothesis that needs its own evidence and clear uncertainty.
4. **Classical baselines come first.** Quantum models are experimental challengers. Use the quantum path only when the data representation, fair comparison, and resource costs make a test meaningful. Do not claim quantum advantage from using a quantum simulator.
5. **Every result is tied to a protocol and provenance.** Preserve cohort definition, data dictionary, units, source, consent/permission basis, split IDs, preprocessing fit scope, seeds, code/environment version, failures, and raw predictions.
6. **Collect sensitive variables only for a justified study.** Record sex-related biology and gender as distinct concepts when scientifically relevant; do not infer either from a name, appearance, or device data. Support "not provided" and other protocol-appropriate values.

## Whole-body research portfolio

The system should add one domain at a time. Each entry below is a possible research program, not an implemented feature or a validated target.

| Domain | Candidate research questions | Candidate data, only with permission | Important evaluation boundary |
|---|---|---|---|
| **Menstrual and reproductive health** | How do bleeding dates, duration, symptoms, pain, medication/hormonal-contraception context, sleep, activity, and optional temperature relate to an individual's observed cycle patterns? How do patterns vary across adolescence, reproductive years, postpartum/lactation, perimenopause, and menopause? | Voluntary diary entries; clinician-adjudicated outcomes for a defined study; optional validated wearable signals. | Do not infer ovulation, fertility, pregnancy, PCOS, endometriosis, or another diagnosis from calendar dates alone. Test across ages and contexts; do not treat irregularity as user error or assume every user menstruates. |
| **Cardiovascular and vascular health** | Can prespecified longitudinal signals support research on a defined future cardiovascular outcome or recovery endpoint? | Clinical measurements, ECG, blood pressure, laboratory records, device/source metadata. | Validate by participant, site, and time; distinguish screening research from an individual risk estimate. A consumer wearable is not a diagnostic ECG or blood-pressure device by default. |
| **Metabolic and endocrine health** | How do repeated measurements and care-context data relate to a defined diabetes, thyroid, or metabolic outcome? | Labs, medication and diagnosis history, anthropometrics, activity/sleep where permitted. | Account for treatment and care-access effects, measurement units, missing-not-at-random data, and timing relative to diagnosis. |
| **Respiratory health** | Can longitudinal symptoms, spirometry, or environmental exposure measurements characterize a defined exacerbation or recovery endpoint? | Quality-controlled spirometry, symptom records, medication context, environment, validated device data. | Do not turn consumer audio or a single sensor into an asthma/COPD diagnosis. Validate device/site shift and missingness. |
| **Neurologic, cognitive, and sleep health** | Can repeated symptom diaries, sleep data, or approved clinical measures characterize a specific longitudinal outcome? | Validated questionnaires, clinical assessments, sleep/device records, event timestamps. | Separate correlation from causal explanation; guard against label delay, practice effects, and device artifacts. No seizure, stroke, dementia, or mental-health diagnosis from passive data. |
| **Renal, urinary, digestive, and liver health** | Can longitudinal labs, symptoms, and care events support a prespecified trajectory or outcome study? | Laboratory values with units/reference context, coded care events, symptom diaries, approved imaging where needed. | Handle repeated tests and care intensity; avoid converting sparse or incidental measurements into a disease alert. |
| **Musculoskeletal, orthopedic, and rehabilitation** | Can approved imaging, function measures, pain reports, or rehabilitation signals characterize recovery or a defined complication? | De-identified imaging, mobility/function assessments, patient-reported outcomes, therapy timelines. | Patient-level and site-level splits; assess image acquisition and device variation; distinguish outcome association from a treatment recommendation. |
| **Perioperative and surgical research** | For a specified procedure and cohort, can preoperative variables estimate a prespecified postoperative research outcome or recovery trajectory? | Preoperative history/labs and timestamped postoperative outcomes under a governed protocol. | The model must use only information available at the stated preoperative index time. No intraoperative guidance, operative-plan recommendation, or autonomous decision. Surgeons help define endpoints and review errors; they remain decision-makers. |
| **Cancer and hematology** | Can a defined screening, diagnostic-workup, recurrence, or treatment-toxicity question be evaluated using a justified cohort? | Approved pathology/imaging/lab data, treatment timeline, reference-standard adjudication. | Keep screening, diagnosis, prognosis, and recurrence as distinct tasks. Prevent post-diagnosis leakage and report false-positive burden and subgroup uncertainty. |
| **Immune, infectious, skin, eye, oral, and ENT health** | Can a clearly scoped modality-specific dataset support a reproducible research task? | Approved laboratory, image, signal, or clinical records with modality-specific labels. | These are separate data products and model cards; do not imply coverage because the platform accepts CSV. |
| **Mental and social well-being across systems** | How do validated patient-reported measures and context relate to specified health outcomes? | Voluntary validated instruments and carefully governed social/environmental data. | Minimize collection, protect against sensitive inference, avoid crisis or treatment claims, and use qualified human oversight. |

### First build: menstrual-health journal and research dataset contract

The cycle feature should begin as a private, user-controlled diary and export path. Its initial job is accurate record-keeping and clinician communication, not medical inference.

- Record optional period start/end dates, spotting, flow description, pain, symptoms, mood, medication/hormonal contraception context, and notes. Preserve user-entered values and corrections with timestamps; never silently rewrite history.
- Support missing entries, unknown dates, irregular cycles, no bleeding, pregnancy/postpartum/lactation context when the user chooses to record it, perimenopause/menopause, and users who do not menstruate. Do not force a regular 28-day template.
- Explain what was entered, what is missing, and what an estimate used. A future cycle-date estimate must show a personal uncertainty interval and abstain when history is sparse, changing, or outside its validated population.
- Do not market or use a calendar estimate for contraception, fertility planning, ovulation confirmation, pregnancy detection, or diagnosis. Do not label a pattern as abnormal or reassuring without an approved, context-appropriate clinical pathway.
- Let the user export and delete their diary. Reproductive-health data requires explicit opt-in before research reuse, granular consent, minimal collection, encryption, access logs, and a retention/deletion policy. No advertising, sale, or unrelated secondary use.
- For minors, pregnancy, sexual/reproductive records, and shared-device scenarios, complete a dedicated privacy and ethics review before enabling collection; define access and consent according to the study and applicable jurisdiction.

ACOG describes asking adolescents about the first day of the last menstrual period and menstrual pattern as part of clinical assessment; this supports taking the record seriously, not treating an app's prediction as a diagnosis. NIH policy calls for considering sex as a biological variable in relevant research. The research design should also account for gender, anatomy, age, hormonal context, and social factors without conflating them.

## AI/ML research design

### Cohort and target definition

Before fitting a model, register the population, setting, inclusion/exclusion rules, index time, predictor-availability window, target/reference standard, prediction horizon, censoring rules, intended research use, and primary endpoint. For early detection, establish that predictors precede the outcome and compare predicted lead time with the real usual-care pathway. If the source is cross-sectional, call it status classification, not early detection.

For cycle research, distinguish prediction of a future bleeding date from explanation of symptoms, fertility/ovulation estimation, and clinical disease detection. These are different estimands and require different labels and evidence. Use participant-level and future-time holdouts. For other domains, use site-, patient-, procedure-, and time-aware splits as the design requires.

### Modality and validation sequence

1. **Structured clinical/tabular data:** finish schema, provenance, units, target validation, missingness, group/time split manifests, and fold-local preprocessing.
2. **Patient-reported and cycle time series:** define event-time semantics, correction history, irregular sampling, missingness, and participant-level longitudinal evaluation.
3. **Wearables and mobile sensors:** add a source adapter only for a named device/data type; record model, firmware, units, sampling, quality flags, permissions, gaps, and clock drift. Validate device-held-out and participant-held-out behavior. No emergency monitoring.
4. **Imaging, omics, EHR, and signals:** each gets its own adapter, data-use basis, modality-specific preprocessing, reference standard, comparators, governance, and model card. A generic file upload is not modality support.
5. **Multimodal fusion:** only after each input path has independent quality and validation evidence. Compare complete-case, single-modality, and fused models; measure the impact of missing modalities and their collection process.

### Modeling and evidence

- Establish transparent statistical and strong classical baselines before adding quantum models.
- Keep participant sets, splits, preprocessing, tuning budget, target definition, and evaluation code identical across models. All learned transforms, feature selection, calibration, and threshold selection belong inside training/calibration partitions.
- Report discrimination and task-appropriate measures: AUROC and AUPRC with uncertainty; sensitivity, specificity, PPV/NPV and false alerts at prespecified operating points; calibration; coverage/abstention; lead-time and net benefit only when the cohort and decision context support them. For date forecasting, report forecast error and interval coverage rather than diagnostic classification metrics.
- Report participant-level denominators, event counts, missingness, subgroup uncertainty, external-site/time performance, failure rates, latency and resource use. Small groups are not evidence of equal performance; suppress unsafe small-cell reporting and state when subgroup performance is not estimable.
- Explanations are descriptions of model behavior, not causal mechanisms or medical reasoning. Record stability and show counterexamples. A clinician must be able to inspect data, target, model version, evidence, and uncertainty before any future CDS evaluation.
- Use a locked external cohort after development. Then, if justified and approved, run a silent prospective research evaluation before any clinical workflow study. Do not silently retrain on incoming data.

## Quantum/ML plan for whole-body research

Quantum ML is a hypothesis to test on a narrow, low-dimensional, well-defined representation—not a shortcut for ingesting an entire body, genome, image, or EHR. The existing QEDNET quantum implementations are simulator-oriented; real quantum hardware execution is not established.

For every study, compare classical linear/logistic, tree/boosted, and kernel baselines against the relevant quantum model on the same participants and folds. Record the full feature map/encoding, compression fit, qubits, layers/depth, trainable parameters, shots, simulator/device, runtime, failures, memory/queue costs, and input dimension. Include a capacity-matched classical control and classical approximation/dequantization where applicable. Hardware runs are separate from ideal simulator runs.

Advance a quantum candidate only if a preregistered result independently reproduces and improves a defined quality or resource objective under uncertainty and relevant noise. Otherwise publish the negative result. Do not use quantum model outputs in a user-facing health tracker or clinical pathway merely because they are available in the model registry.

## Clinician and surgeon participation

Doctors, nurses, midwives, reproductive-health specialists, physiotherapists, radiologists, pathologists, and surgeons should contribute as study designers and reviewers where relevant. Their defined roles are to select meaningful outcomes, document standard-care comparators, review data definitions, adjudicate reference labels, identify unacceptable errors, and assess workflow burden. Patient/community partners should review consent, usability, privacy, and language.

This is not a product that lets an AI "act like a doctor or surgeon." It must not recommend medication, contraception, fertility action, surgery, or treatment; interpret a scan for care; or monitor emergencies. A future clinical decision-support product is a separate project requiring a precise intended use, qualified owners, jurisdiction-specific regulatory review, independent validation, human-factors testing, safety controls, and formal release approval. FDA guidance is one jurisdictional reference, not a global determination.

## Research production-readiness gates

The research service can be called production-ready for a defined research use only after all of these are evidenced:

- Authentication, least-privilege authorization, tenant/data isolation, secret management, secure artifact handling, audit logging, deletion/export, and an approved data-governance policy.
- Durable database, queue, and object storage with backups, tested restore, monitoring, resource limits, incident response, and documented upgrades/rollback.
- Approved datasets and licenses; participant consent/ethics review where applicable; documented provenance, retention, and secondary-use permissions.
- Reproducible locked environments and CI; automated tests for leakage, group/time isolation, data validation, access control, artifact lifecycle, and job failure/retry behavior.
- Model/run registry with immutable configurations, code/data fingerprints, split manifests, raw predictions, exact metrics, uncertainty, and downloadable reports.
- A named research protocol and an external validation plan for each claimed use case. No feature becomes production because its screen renders or its model class is registered.

As of this plan, whole-body and women's-health model research is **NOT IMPLEMENTED**; a persistent menstrual journal and its secure participant-data lifecycle are **NOT IMPLEMENTED** (only an in-memory UI prototype exists); Android and wearable integrations are **NOT IMPLEMENTED**; early-detection validity is **NOT ESTABLISHED**; and clinical/surgical use is **NOT AUTHORIZED**. Current local Docker checks establish that the app and supporting services can run, but local authentication and an authenticated research session remain **NOT VERIFIED**.

## Delivery sequence and exit criteria

| Phase | Work | Exit evidence |
|---|---|---|
| **A. Research service trust** | Configure supported auth, data access boundaries, persistent storage, backup/restore, monitoring, deletion, and environment/release controls. | Security review, access-control tests, recovery rehearsal, and reproducible clean deployment. |
| **B. Choose first whole-body study** | Convene clinical and patient/community reviewers; select one target population, body system, endpoint, source, and permitted cohort. Women's-health cycle research is a candidate, not an assumed dataset or diagnosis feature. | Approved protocol, data/ethics determination, data dictionary, sample/event feasibility, and explicit out-of-scope list. |
| **C. Cycle diary, if selected** | Build private record-keeping, export/delete, consent, offline behavior, and careful handling of variable cycles; no model forecasts in this phase. | Usability/privacy review and tests for corrections, missing periods, variable patterns, non-menstruating users, and data deletion. |
| **D. Baselines and temporal validation** | Build a protocol-specific cohort, simple classical models, group/time splits, uncertainty, and error review. | Locked evaluation and independently reproducible external/site/time validation; publish negative findings. |
| **E. Quantum challenger** | Add the quantum pipeline to the same approved cohort and protocol only after baselines are credible. | Fair matched report with device/simulator metadata, capacity/resource comparison, uncertainty, and reproducible code/data provenance. |
| **F. Real-time research** | Add one approved device/EHR source and silent shadow capture, without user alerts or care recommendations. | Data-quality, latency, downtime, stale-input, privacy, and prospective shadow-study evidence. |
| **G. Separate clinical program** | Start only with a specific intended use, clinical sponsor, human-factors and regulatory plan, quality system, and new isolated service. | Independent clinical and operational approvals; never an automatic promotion of the research system. |

No calendar estimate is given: dataset access, ethics review, clinical owners, hardware access, and hosting controls are not secured by this repository.

## Evidence references

- WHO, [Ethics and governance of artificial intelligence for health](https://www.who.int/publications/i/item/9789240029200): human rights, autonomy, privacy, inclusiveness, accountability, and governance.
- WHO, [Artificial intelligence-related health research: ethics review and oversight](https://www.who.int/publications/i/item/9789240124073): oversight needs for AI-related health research.
- NIH Office of Research on Women's Health, [NIH policies and guidelines](https://orwh.od.nih.gov/toolkit/nih-policies-inclusion/guidelines): inclusion and consideration of sex as a biological variable.
- ACOG, [Menstruation in Girls and Adolescents: Using the Menstrual Cycle as a Vital Sign](https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2015/12/menstruation-in-girls-and-adolescents-using-the-menstrual-cycle-as-a-vital-sign): clinical context for recording menstrual patterns; not a product validation study.
- FDA, [Clinical Decision Support Software Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software): a US-specific planning reference for future CDS; it is not a legal determination for QEDNET or another jurisdiction.
- NIH-hosted study, [Menstrual cycle variations in wearable-detected signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC11416332/): an example of limited wearable research, not evidence that a consumer device or QEDNET can predict a person's health outcome.
