# QED-Net Production Readiness Review

## Scope

This review scopes production work to a secure research and educational platform. It does not treat QED-Net as a clinical decision-support product. Benchmark performance, simulator verification, or a successful cloud health check does not establish clinical safety or effectiveness.

See [RESEARCH_COMPLETION_STATUS.md](RESEARCH_COMPLETION_STATUS.md) for the latest local code fixes, smoke-run evidence, and current release position. The verification record below documents earlier checks and is historical where it refers to the prior deployment/auth configuration.

## Current observed state

| Area | Observed state | Readiness |
|---|---|---|
| Research core | Python package contains classical and simulator-based quantum models, evaluation, calibration, explainability, uncertainty, certificates, and file-backed experiment artifacts. | Present in source; release verification remains required. |
| Dataset validation | Validation reports, pass/fail status, and validation timestamps are now persisted in the local dataset registry and displayed after reload. | Verified locally; registry remains non-durable in filesystem-demo mode. |
| API boundary | The colocated FastAPI service binds to loopback by default from the repository launcher; API dependency and CLI errors return generic client messages. | Verified in source and regression tests; production network policy still needs an environment-level check. |
| Legacy Caddy example | The query-controlled local reverse-proxy target was replaced with a fixed `/socket.io/*` route to port 3003. | Source reviewed; Caddy parser/runtime unavailable locally, so Caddy execution is NOT MEASURED. |
| Dashboard/API | Next.js app and FastAPI app are deployed together on Render. Liveness, readiness, models, dataset listing, and experiment listing responded successfully in the deployed environment. | Live research demo. |
| Application revision checked live | Render served application commit `20b028d8ae18f661527734301d10466f6dcba5f5`. Public `/live` and `/signin` returned HTTP 200; unauthenticated `/api/health` and `/api/datasets` redirected with HTTP 307. | Deployment and anonymous route behavior verified; authenticated user flows are not verified. |
| Authentication | Supabase email/password sign-in and one-email allowlist are configured. Email confirmation is disabled. The current account has signed in, but fresh signup and password reset were not re-tested in this review. | Private preview only; disabling confirmation means email ownership is not verified. |
| Job execution | Dashboard `/api/jobs` launches a local child process. A separate FastAPI training route can use ARQ only when database mode is enabled. These paths are not integrated. | Blocker for reliable queued training. |
| Persistence | Render is in filesystem-demo mode, with no attached durable database, object store, or persistent disk. Render filesystem writes are not durable across service replacement. | Blocker for durable data and runs. |
| Dataset availability | Breast cancer loads through scikit-learn. Heart disease, Parkinson's, and Pima adapters require local files that are not in the repository or cloud deployment. | Limited dataset support in deployment. |
| Model artifacts | Prediction loads Python pickle artifacts from run directories. Artifacts must be treated as trusted executable content and protected from untrusted writes. | Blocker until access and artifact trust boundaries are enforced. |
| Clinical claims | The project README and model cards say research/educational use only and not clinically validated. | Clinical deployment prohibited by current evidence. |
| Compose | `docker compose config --quiet` passed. The containers and worker lifecycle have not been executed because the local Docker daemon is unavailable. | Not runtime-verified. |
| License and data rights | No project license file is present; local raw dataset files are not bundled. | Legal review and explicit data-rights records required before external/commercial distribution. |

## Decisions for hosting, access, jobs, uploads, and claims

- **Hosting (free-only):** keep Render Free as a private demonstration environment using public benchmark data only. It is not a suitable host for retained uploads or reliable long-running jobs because its filesystem is temporary and there is no durable database, object store, or worker attached. Do not connect the unrelated Chhaya database. For a persistent research service, use a managed PostgreSQL database, durable object storage, and a separate worker/queue with tested backups and recovery; this requires a paid plan or another provider whose current terms meet those requirements. No free deployment should be represented as durable until persistence and restore have been demonstrated.
- **Access:** retain the existing single-email allowlist. Do not enable multiple accounts until ownership checks cover every dataset, upload, job, run, report, certificate, and model artifact. Email confirmation is disabled, so the allowed account's email ownership is not verified.
- **Jobs:** the Next.js dashboard currently spawns local child processes, while FastAPI's ARQ path is separate and only used when `USE_DB_STORAGE` is enabled. The paths are not integrated and Render Free cannot preserve them through restarts. Treat hosted training as best-effort preview behavior; reliable execution remains NOT IMPLEMENTED/NOT VERIFIED until a durable queue is integrated, the worker is independently deployed, and restart, retry, cancellation, timeout, and result recovery tests pass.
- **Uploads:** CSV/TSV uploads must pass schema validation and must not contain direct-identifier-like columns before they are registered. Failed validation must leave no registered dataset or orphaned upload file. Successful upload files remain on the service filesystem in preview mode; there is no automatic retention or durable deletion policy, so use public benchmark data only until storage and retention controls are implemented.
- **Scientific claims:** describe a certificate as a summary of the evidence recorded for that run. It does not establish quantum advantage, generalization, early detection, or clinical performance. Report only the split strategy, seed, cohort/data fingerprint, models, metrics, and hardware/simulator that are present in that run's saved artifacts. Do not imply nested CV or multi-seed evaluation unless that particular run record proves it.

## Production blockers

### Security and privacy

- Decide whether to retain disabled email confirmation. If retained, document that the account email is not verified and test password reset delivery. If confirmation is re-enabled later, configure transactional email and test signup and reset end to end. Next.js Proxy and protected route handlers check the server-side user and allowlisted email; only explicitly public auth/setup/liveness paths remain public.
- Add tenant and ownership checks before accepting or returning datasets, jobs, runs, reports, certificates, and model artifacts. Existing database tables alone do not enforce user-facing authorization.
- Add request rate limits, request-size limits at the edge and application, upload quotas, audit events, abuse detection, and safe error responses.
- Define what data may be uploaded. Reject direct identifiers and document retention, deletion, encryption, backup, and incident-response policies.
- Keep private signing keys, database credentials, and object-store credentials in a managed secret store. Rotate the local Compose defaults before any non-local use.
- Treat `.pkl` model artifacts as executable trusted inputs. Restrict writers and readers; define signed/versioned artifact formats before accepting artifacts from users or external storage.

### Runtime and data integrity

- Configure managed persistent PostgreSQL and object storage, apply migrations in a controlled deployment step, and verify backup and restore procedures.
- Route dashboard training through the durable queue and expose durable job status/cancellation. Remove the dashboard's detached-child-process production path.
- Make dataset registration and experiment artifacts use durable storage, content checksums, ownership metadata, and atomic writes.
- Configure resource limits, concurrency limits, worker health checks, graceful shutdown, job timeouts/retries, and observability. Test a worker restart during a run.
- Verify uploads and prediction requests against strict schemas and dataset feature contracts. Current demo presets are illustrative values, not patient measurements.
- Add a deployment rollback procedure, dependency update policy, vulnerability scanning, software bill of materials, and reproducible release artifacts.

### Scientific and clinical evidence

- Freeze intended research use, cohort definitions, inclusion/exclusion criteria, preprocessing, split strategy, seeds, and baselines before each benchmark.
- Re-run the benchmark from committed raw inputs or documented retrieval steps and preserve all per-fold records. Do not treat a small exploratory sample as validation evidence.
- Validate calibration and conformal coverage on properly held-out cohorts; assess subgroup performance, missingness, external-site shift, and uncertainty under distribution shift.
- Do not expose predictions as diagnosis or treatment recommendations. A clinical product would require a separate intended-use statement, clinical risk controls, independent clinical validation, quality-system work, human-factors evaluation, privacy/security review, and jurisdiction-specific regulatory assessment.

## Delivery gates

### Gate: private research preview

- Keep the current exact email allowlist and exercise sign-in, sign-out, password reset, and blocked-email behavior against the hosted project. Signup confirmation is intentionally disabled in the current Supabase configuration.
- Keep only de-identified research datasets; add upload quotas and operational logs.
- Connect dashboard training to the queue, or disable remote training until the queue is configured.
- Configure durable storage and verify backup restoration, or clearly limit the app to disposable demo data.
- Run unit, integration, API contract, and browser smoke tests from a clean checkout; execute the Compose stack on a Docker-capable runner.

### Gate: production research service

- Complete tenant authorization, threat modeling, security review, dependency/image scanning, secrets management, database/object storage backup and restore, monitoring, alerting, and rollback exercises.
- Establish a benchmark release process with frozen data fingerprints, exact splits/seeds, raw fold outputs, and independent reproduction.
- Publish data licenses, project license, model cards, acceptable-use rules, and retention/deletion policy.

### Gate: any clinical decision-support use

- Treat as a separate product and validation program. Obtain clinical, privacy, security, quality, and regulatory expertise; independently validate the model and workflow on representative intended-use populations before any patient-care use.
- Do not use the research Render demo or its current model artifacts for patient-care decisions.

## Inputs required to finish deployment hardening

- Keep the Supabase URL and publishable key, `QEDNET_ALLOWED_EMAIL`, and `NEXT_PUBLIC_SITE_URL` in Render. Email confirmation is currently disabled; if it is re-enabled, configure transactional SMTP and test confirmation and password reset. Never set the service-role key in the browser app or Render web service.
- Confirmation of whether paid managed database, object storage, and worker resources are permitted. Free Render mode does not meet durable-storage requirements.
- Confirmation that the target is research use only. A clinical target needs a separate validated product plan and evidence.
- Approved dataset sources, licenses, consent/governance requirements, and data-retention rules for any non-public uploads.

## Verification record

Latest local checks for dataset-validation persistence and API error handling are recorded in `results/research_completion_v2/COMMANDS.md`. The historical checks below are from earlier local verification passes and are not proof that these changes are deployed:

| Command | Result | Scope and caveats |
|---|---|---|
| `$env:PYTHONPATH='backend'; python -m pytest backend/tests -q` | Passed | Backend test suite; warnings from upstream sklearn/SHAP dependencies appeared. |
| `bun test tests/auth.test.ts` | Passed | Covers HTTPS auth configuration requirements, exact single-email allowlisting, and the production setup lockout. |
| `bun run lint` | Passed with warnings | ESLint is scoped by the package script to `src` and `next.config.ts`; existing unused-code warnings remain. |
| `bun run build` | Passed | TypeScript and Next.js production build passed; Turbopack emitted dynamic filesystem tracing warnings for the CLI/filesystem bridge. |
| Built standalone server smoke check with auth settings unset | `/live` returned 200; `/`, `/signin`, `/signup`, and `/api/auth/signup` redirected to `/setup-required`; `/setup-required` returned 200 | Confirms fail-closed behavior on the built app. Hosted signup and email confirmation were not tested without a Supabase project. |
| `docker compose config --quiet` | Passed | Validates Compose configuration only; containers and worker lifecycle were not run because Docker was unavailable. |
| Latest live Render check | Raw output: `results/research_completion_v2/live_render_verification.raw.txt` | Verifies deployment commit, liveness/sign-in pages, and anonymous redirects only. Does not verify signed-in flows, account data, persistence, training, or recovery. |

No hosted signup/confirmation/reset flow, tenant isolation, durable storage/recovery, worker restart, production upload policy, security scan, or clinical workflow was measured. Supabase Free is a preview tier: projects can pause after inactivity and do not include automatic backups. Re-run and retain command outputs from a clean checkout after release changes. A successful build or health check does not certify every feature, production security, durable persistence, or clinical readiness.
