# QEDNET repository comparison and readiness review

## Scope

This workspace contains two separately versioned applications plus an assets bundle:

- The workspace root (`QEDNT`) is the primary Python research package and Vite/React research dashboard.
- `qednet-2.0-project (2)` is a separate Next.js application with a Python backend/CLI and optional database/queue deployment pieces.
- `qednet-package` is a data and agent-tools bundle, not a third application with a complete user-facing product lifecycle.

These are not interchangeable folders in one application. They have separate Git metadata and dependency/build systems. The root and the nested app currently use the `production-readiness` branch; changes are not committed.

## End-to-end feature comparison

| Area | Workspace root | `qednet-2.0-project (2)` | Assessment |
|---|---|---|---|
| Research UI | Vite/React frontend calls the research API. The old Streamlit page was replaced with a warning and pointer so it no longer fabricates predictions. | Next.js 16 dashboard with server-side API routes and job-oriented UI. | The nested app has the more integrated web experience; the root has a simpler active research UI. |
| Python model layer | Reusable `qednet` package with classical estimators, PennyLane classifiers, preprocessing, evaluation, explainability, diagnostics, and certificates. | Python core and CLI with experiment/data workflows and a distinct custom statevector simulator path. | Keep these implementations clearly separated until a single supported model API and artifact format are agreed. |
| API | Root research API at `/research`, plus older API surface. Model support was restricted to implementations that actually exist; path, upload-size, and feature validations were tightened. | FastAPI backend plus Next route handlers. Training route now uses `BackgroundTasks` correctly; IDs, uploads, CORS defaults, subprocess environment, and timeout errors were hardened. | Neither API has production identity/access control verified. Do not expose either to public or clinical traffic yet. |
| Datasets | Breast cancer can load from scikit-learn; heart disease, Parkinson's, and diabetes require local raw files. Parkinson's now extracts subject IDs, excludes the filename from features, and supports grouped outer CV. | Includes a broader bundled data tree and a separate dataset registry/loader; Parkinson's metadata includes group information. | Dataset provenance, licensing, data checksums, versioning, and PHI governance still need a deliberate policy. |
| Preprocessing/evaluation | Pipeline fit/transform occurs within outer training folds. Group-aware Parkinson's evaluation is wired through the reusable benchmark, API, and reproduction script. The honesty comparison no longer fits compression on all rows before CV. | Experiment runner and CLI have more workflow/persistence surface, but this review did not establish equivalent reproducibility or clinical validity for every path. | Add automated leakage tests for every public evaluation entry point and preserve fold IDs and preprocessing artifacts with results. |
| Jobs/persistence | Root API persists local run folders and uses in-process background tasks in its simple profile. | Alternate app has filesystem job descriptors and optional Postgres/Redis/MinIO/ARQ pieces. | In-process and filesystem jobs are not durable production orchestration. Validate the configured queue/storage path under deployment before relying on it. |
| Packaging/deployment | Root Docker/compose configuration was aligned with the root API and built dashboard. Compose configuration validation passed earlier in this work. | Nested Docker/compose has more infrastructure components but also includes demo-oriented configuration and was not deployed or load-tested. | A successful local build/config check is not a production deployment or operational readiness proof. |
| Explainability and reporting | Explainability/diagnostic modules and result certificates exist. | Dashboard/API routes expose explainability and certificate-related actions. | Explanations are not clinical evidence and must be checked for leakage, stability, and user comprehension before product use. |

## Fixes made in this review

- Replaced the legacy Streamlit mock-results screen with a research-only notice pointing to the real frontend.
- Fixed the root reproduction script to call the repository's actual dataset registry, model registry, preprocessing factory, and CV implementation.
- Corrected the root CV documentation so it no longer claims nested hyperparameter tuning that is not implemented.
- Added optional subject-grouped stratified CV. Parkinson's subject IDs are retained as metadata, excluded from model features, and passed into CV by the API, benchmark helper, and reproduction command. Grouped mode rejects non-group-aware probability calibration rather than silently leaking groups.
- Removed a full-dataset preprocessing fit from the bottleneck-honesty comparison. The quantum and classical estimates now go through the same fold-local preprocessing on the same input rows.
- Corrected conformal score/order-statistic behavior and added tests for finite-sample thresholds, small calibration sets, class-specific margins, and seeded APS.
- Tightened API input and upload handling, removed unsupported/mock model choices, fixed the nested FastAPI training-route annotation, and stopped the Next.js build from skipping TypeScript checks.
- Brought the alternate app's liveness/readiness probe pattern into the root serving API, made optional uncertainty/conformal request flags effective, rejected non-finite/inconsistent feature payloads, and changed the unimplemented experiment lookup to return HTTP 501.
- Updated the root README and Docker compose wiring for the actual root application.

## Verification performed

- Root Python full suite: passed (`python -m pytest -q`); includes integration, API, deployment, and grouped-fold checks. Existing dependency/convergence warnings remain.
- Root focused integration/deployment suite: passed (`python -m pytest -q tests/integration/test_e2e_pipeline.py tests/integration/test_research_api.py tests/test_deploy.py`).
- Alternate backend full test suite: completed with exit code `0`.
- Alternate Next.js production build: completed, including TypeScript validation. Turbopack still warns that dynamic filesystem access in `src/lib/qednet.ts` can cause broad project tracing.
- Ruff checks and formatting now pass across the root Python source and tests. The whole-package MyPy run remains blocked by the repository's existing typing/stub debt; CI now checks the API schema module only, so package-wide static typing is still incomplete.
- Parkinson's grouped loader smoke check: raw file loaded; the generated three-fold split had no subject overlap. This verifies grouping behavior, not model quality.

## Production-readiness decision

QEDNET is a research prototype, not a production or clinical decision-support system. The fixes remove several misleading or unsafe behaviors, but they do not establish that the system is safe, secure, clinically valid, or operationally reliable for patient care.

Before any public or clinical deployment, the project still needs:

- Authentication, authorization, tenant separation, rate limiting, secrets management, and security review for every API route.
- A threat model and safe artifact format. Loading Python pickle artifacts is unsafe when artifact provenance is not fully trusted.
- Explicit retention/deletion controls, encryption, audit trails, backups, and governed handling of uploaded health data.
- Durable, bounded job execution with cancellation, retries, resource quotas, monitoring, and tested failure recovery.
- Versioned dataset/model/preprocessing provenance, immutable run manifests, and verified reproducibility across clean environments.
- External clinical validation, prospective evaluation, subgroup/fairness analysis, calibration and abstention evaluation, human-factors testing, and the required regulatory/privacy review for the target jurisdiction and intended use.
- CI cleanup: resolve the repository-wide Ruff backlog, verify formatting and type checking, build containers in CI, and test both supported deployment profiles.
- Deployment-specific remediation for the Next.js dynamic filesystem tracing warnings.

Until those gates are completed, present the project as an experimental research/benchmarking platform. Do not describe its scores or explanations as diagnostic performance or use it to guide patient treatment.
