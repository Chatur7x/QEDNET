# QEDNET Research Platform Completion Status

## Scope and verdict

This review covers the QEDNET research and education platform only. It does not authorize or establish clinical decision support. The repository now has stronger local verification and more accurate preview messaging, but the deployed service remains a **private, disposable research preview**, not a persistent production research service.

## Work completed in this pass

- Updated README and readiness/boundary documents to say email confirmation is disabled and access is limited to the configured email. The current account's sign-in was reported by the user; fresh signup and password-reset delivery were not tested here.
- Added an in-app notice that the Render preview uses temporary filesystem storage and must only be used with public or de-identified research data.
- Narrowed the dashboard's metric-provenance wording. It no longer claims every displayed value is a real experiment result.
- Persisted dataset validation status, full validation report, and validation time in the dataset registry. The dashboard reloads and displays the saved status instead of relying on page-local state. In filesystem-demo deployments, this registry still uses the non-durable service filesystem.
- Sanitized FastAPI responses for subprocess and training-queue failures; added regression tests that prove internal stderr/database details are not sent to clients.
- Removed the legacy Caddy configuration's query-selected proxy port and routed the fixed Socket.IO path to its fixed local service port. Caddy runtime parsing was not available locally.
- Fixed the Parkinson's validation command so it excludes `__subject_id__` from numeric feature validation while still checking the group key.
- Fixed Parkinson's dataset profiling so `__subject_id__` is excluded from predictive-feature counts and the registry profile. The key remains available for group-aware splitting.
- Fixed the VQC parameter-shift explanation import and added regression tests for the import and grouped dataset profiling/validation.
- Made the full backend integration test independent of ignored local benchmark files by having it create a deterministic synthetic CSV and call the real ingestion and training pipeline.
- Added `.github/workflows/ci.yml` for backend tests, frontend lint/build and route smoke tests, and a critical-severity Bun dependency audit. The workflow file parsed locally but has **not** run on GitHub.

## Local workflow evidence

All local run commands and raw outputs are indexed in `results/research_completion/`. The local data files are not committed to Git. The smoke runs below are workflow checks, **not benchmark claims**: one seed and three outer folds do not establish comparative performance.

The follow-up changes and their raw checks are indexed in [`results/research_completion_v2/COMMANDS.md`](../results/research_completion_v2/COMMANDS.md). The final backend suite exited successfully (92 progress dots counted from its saved output), the frontend lint/build passed, and the protected-route smoke passed against the local standalone build. These checks do not exercise an authenticated hosted session or the deployed Render revision. Caddy's fixed route passed a source assertion only; its parser/runtime was unavailable.

| Dataset | Local data profile | Validation | Saved experiment | Models in smoke run | Evidence |
|---|---:|---|---|---|---|
| breast_cancer | 569 rows, 30 predictive features | Passed | `qednet_completion_e2e_verified` | Logistic regression, VQC | Result, config, pipeline, models, explanation and report artifacts; prediction and explanation CLI calls succeeded |
| heart_disease | 303 rows, 13 predictive features | Passed with missing-value warnings | `qednet_completion_heart_disease` | Logistic regression, VQC | Result, config, pipeline, models, explanation and report artifacts |
| parkinsons | 195 rows, 22 predictive features; subject key excluded | Passed with subject-group validation | `qednet_completion_parkinsons` | Logistic regression, VQC | Result, config, pipeline, models, explanation and report artifacts |
| pima_diabetes | 768 rows, 8 predictive features | Passed with missing-value warnings | `qednet_completion_pima_diabetes` | Logistic regression, VQC | Result, config, pipeline, models, explanation and report artifacts |

Each local smoke experiment used the same configured protocol: stratified three-fold cross-validation, seed 42, fold-local pipeline, PCA to four components, logistic regression plus VQC, and a four-qubit, one-layer NumPy statevector simulation. Parkinson's used `__subject_id__` as the grouping column. Calibration and bootstrap evaluation were disabled for these workflow checks; certificates were disabled. The stored result explicitly marks conformal coverage `NOT MEASURED` because no independent calibration/evaluation cohort is held out.

The breast-cancer run also exercised the saved-model cascade prediction on a row loaded from the local scikit-learn benchmark adapter, and loaded the VQC explanation artifact. This is a software smoke check, not a patient prediction or clinical evaluation. Quantum execution was simulator-only; real quantum hardware was not measured.

## Verification results

| Check | Command / evidence file | Outcome |
|---|---|---|
| Backend suite | `python -m pytest backend/tests -q` — `backend_pytest_final_code.raw.txt` | Exit code 0; suite reached 100%. Existing scikit-learn/SHAP warnings remain. |
| Frontend auth/API smoke | `QEDNET_BASE_URL=http://127.0.0.1:3100 bun run test` — `frontend_smoke_ci_candidate.raw.txt` | 14 passed, 0 failed; run against the local standalone production build with auth configuration absent, so protected routes were verified to fail closed. |
| Frontend lint | `bun run lint` — `frontend_lint_ci_candidate.raw.txt` | 0 errors; 33 existing warnings. |
| Production build | `bun run build` — `frontend_build_ci_candidate.raw.txt` | Passed. Turbopack reported dynamic filesystem tracing warnings for the Python/filesystem bridge. |
| JavaScript dependency audit | `bun audit --audit-level=critical` — `bun_audit.raw.txt` | No critical vulnerabilities found in the checked JavaScript dependency set. This is not a Python, OS-image, or full security audit. |
| Compose syntax | `docker compose config --quiet` — `compose_config_final.raw.txt` | Passed configuration validation only. |
| Docker runtime | `docker info` — `docker_info.raw.txt` | NOT MEASURED: Docker CLI exists, but the Docker Desktop Linux engine pipe was unavailable. No containers, worker lifecycle, or backup/restore were run. |
| CI YAML parse | `python -c "import yaml; ..."` — `ci_yaml_parse.raw.txt` | YAML parsed locally; GitHub Actions execution is still unverified. |
| Diff whitespace | `git diff --check` — `git_diff_check.raw.txt` | Passed. |

## Remaining work before a persistent production research service

1. **Durability and jobs:** Render remains in filesystem-demo mode. Add and verify durable database/object storage and a durable worker queue, or retain the clearly labelled disposable-preview boundary. The dashboard still starts detached local child-process jobs; this is not the durable ARQ path.
2. **Access isolation:** Keep the one-email allowlist. Add ownership checks for every dataset, upload, job, experiment, report and artifact before admitting more users. FastAPI currently relies on being loopback-only behind the protected Next.js route; independently protect it if that network boundary changes.
3. **Operations:** Run the new CI workflow on GitHub; execute the Compose stack on a Docker-capable runner; verify worker restart, cancellation, timeouts, retries, storage failure, backup and restore; add monitoring, alerting, release rollback and an accountable operator.
4. **Data governance:** Document data source/license, consent or use permissions, provenance, retention, deletion and upload policy for each real dataset. The local raw benchmark files are ignored by Git, so a clean checkout cannot reproduce the local-data runs without authorized retrieval steps.
5. **Scientific evidence:** Treat the one-seed smoke runs as execution evidence only. Prespecify benchmark protocols, retain per-fold outputs, use group/site/time splits where required, and arrange an independent reproduction before publishing comparative performance claims.
6. **Artifacts and uploads:** Restrict trusted artifact writers/readers and replace or tightly control executable pickle loading before accepting external artifacts. Add per-user upload quotas and application/edge rate limits.
7. **Licensing and security:** No root `LICENSE` file was present. The Bun audit covered JavaScript dependencies only; Python packages, base images and runtime configuration still need a release security review.

## Not established

- Current deployed source/revision after these local changes: **not deployed or verified**.
- Fresh signup, password-reset email delivery, and a second account: **not tested**.
- Persistence across Render restart/replacement, multi-user isolation, or backup restoration: **not tested**.
- Full clinical use, early detection, calibration/conformal coverage on an independent cohort, external/prospective validation, or patient outcomes: **not established**.
- Real quantum hardware execution: **not measured**.
- SHAP/counterfactual outputs are research explanations, not causal or clinical reasoning. Counterfactual plausibility constraints were not validated in this pass.

## Release position

QEDNET has a locally verified research workflow across the four available benchmark adapters and a passing local test/build path. It is **not production-ready as a persistent multi-user research service**. Keep it private and disposable, use only public or properly de-identified research data, and do not use its outputs for care decisions until the remaining operational, security, governance and independent-evidence gates are met.
