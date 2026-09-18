# QED-Net 2.0 — Multi-Agent Worklog

---
Task ID: 0
Agent: main (Super Z)
Task: Read all uploaded files (AVINYA2k26.pdf, SKILL (2).md = apple-design, master build prompt) and set up environment.

Work Log:
- Read all 3 uploaded files completely. QED-Net 2.0: hybrid quantum-classical ML research platform, exactly 12 features F1-F12.
- Environment: Next.js 16 + shadcn/ui + framer-motion + recharts at /home/z/my-project (sandbox root = frontend).
- Python venv (3.12): numpy, scipy, scikit-learn, pandas, xgboost, shap, pennylane 0.45.1 (default.qubit + lightning.qubit verified), fastapi, imbalanced-learn, pandera, hypothesis, mlflow, lightgbm, pyyaml, pytest.
- Datasets downloaded to data/raw/: heart disease Cleveland (303x13), parkinsons (195x22), pima diabetes (768x8). Breast cancer via sklearn (569x30).
- Benchmark: PennyLane per-sample training ~12ms/sample -> too slow for full nested CV; decision: dual-backend quantum engine (custom batched NumPy statevector simulator for variational training + PennyLane reference/validation engine with equivalence tests).

Stage Summary:
- All prerequisites ready. Key decision: architecture = Next.js frontend (sandbox root) + Python `backend/qednet` package invoked via CLI subprocess from Next.js API routes; FastAPI + Celery + Redis + Postgres + MinIO provided for docker-compose deployment.
- IMPORTANT HONESTY RULES from spec: no fake results, no fabricated quantum advantage, research-only status, test fixtures clearly marked.

---
Task ID: 1-9
Agent: main (Super Z)
Task: Build complete Python backend (qednet package): F1-F12 features, dual-backend quantum simulator, evaluation engine, CLI, tests.

Work Log:
- Built backend/qednet/ (config, sim/statevector, data F1/F2, preprocess F3 with guarded transformers, encoding F4, models F5/F6, evaluation F7, explain F8, certificate F9 (dequant+ablation), honesty F10, uncertainty F11, cascade F12, tracking store, train/runner, cli).
- Batched statevector simulator verified against PennyLane default.qubit to 1.1e-16 machine precision (tests/quantum).
- GuardedTransformer wrappers enforce leakage guard at object level (any .fit() during eval raises LeakageError).
- Fixed: einsum transpose bug, mean_z broadcasting, DeLong placement-based implementation, xgboost param counting, pickling of guarded wrappers, ablation target selection.
- Test suite: 54 tests all passing (unit 30, leakage 9, quantum 10, integration 5).
- CLI: python -m qednet.cli {list-datasets,validate,upload,run-experiment,list-experiments,get-experiment,explain,certificate,predict,cascade-eval,env} — strict JSON on stdout.

Stage Summary:
- Backend complete and tested. Configs in configs/experiments/{breast_cancer,heart_disease,parkinsons,pima_diabetes}.yaml.
- Next: smoke-test CLI, launch real experiments in background, build Next.js dashboard.

---
Task ID: 11-14 (partial)
Agent: main (Super Z)
Task: Frontend (Next.js dashboard, 8 views, design tokens, charts, motion), API bridge, FastAPI app, Docker, reproduce script, README, model cards.

Work Log:
- src/lib/qednet.ts (Python CLI bridge + detached job system), src/lib/api-client.ts (typed TanStack Query hooks).
- API routes: health, datasets (GET/POST upload), validate, experiments (+[id] with report.md), explain, certificate, predict, jobs (GET/POST), jobs/[id].
- Design system in globals.css: domain accent tokens (quantum=violet, classical=teal, evaluation=amber, explain=emerald, uncertainty=rose, research=slate), tabular numerics, display tracking, reduced-motion CSS.
- Components: ui-bits (domain dots, badges, stats, states, spring presets bounce=0/duration=0.38), charts (ROC/PR/calibration/decision curve/model bars/SHAP diverging bars/histogram/sensitivity scatter — recharts, legends outside plots).
- Shell: sidebar nav + 8 views with AnimatePresence spring transitions + mobile nav + research-only badges; sticky footer.
- Views: overview, datasets (upload/validate), training (job launcher + live progress), prediction (cascade + abstention), explainability (SHAP global/local, permutation, sensitivity, counterfactual, quantum diagnostics), benchmark (metric table + 5 chart tabs + stats + resources), advantage (certificate verdict, ablation, dequantization, honesty meter, stats, costs), reports (conformal, cascade, histogram, markdown report, reproducibility).
- FastAPI app (backend/api/main.py), Dockerfile, docker-compose (Celery/Redis/Postgres/MinIO — honestly marked untested in sandbox), scripts/reproduce.py, README.md, docs/model-cards.md, requirements.txt, backend/pyproject.toml.
- Fixed: explainability.tsx ternary syntax error (&& ( ... ) : null), cli() string-vs-array args, Badge import, TS strict errors. tsc + eslint clean.
- Background experiments kept dying under the Bash session; solution: launched via dev-server job system (POST /api/jobs) — survives and progresses.

Stage Summary:
- Frontend complete; API bridge verified (health/datasets/jobs live with real data).
- exp_breast_cancer running through job system (job_mu5w8fdv_rxkt) with live progress.
- Next: verify all 8 views in browser, run remaining datasets, final QA + report.

---
Task ID: 15-17 (continuation session)
Agent: main (Super Z)
Task: Continue from context handoff — finish frontend verification, run all 4 experiments to completion, fix issues found.

Work Log:
- Fixed ROC "chance" line in charts.tsx: replaced index-fraction resampling on series[0]'s irregular FPR grid with linear interpolation of every model's ROC onto a shared uniform FPR grid (N=120). Chance diagonal now exact; curves aligned by FPR value. Verified in browser DOM: chance path = 121 points (24,185)->(918,6); model curves proper bezier paths.
- Fixed model family coloring bug: views pass display labels into charts but familyOf() only knew raw ids -> quantum models colored teal. familyOf() now resolves both raw ids and display labels via MODEL_LABELS inversion. Verified: QSVM violet #7c3aed in rendered SVG.
- Fixed n_samples inflation in evaluation engine (summed test sizes across models x seeds = 9N instead of N). engine.py now divides by (n_models x n_seeds). Patched 3 completed result.json files via scripts/patch_n_samples.py (5121->569, 2727->303, 1755->195). Verified in dashboard.
- Process management: discovered platform kills process trees by ancestry when a tool call ends; setsid -f (always-fork, instant orphan to init) is the correct detachment. First pima launch also failed on cwd (data/raw resolved relative to project root, not backend/) -> relaunched from project root with PYTHONPATH=backend. pima running as pid 13081 with job descriptor job_1a0b0e67715_pz94.
- Dev server was OOM-killed (next-server 1.75GB RSS, 4.1GB RAM, 2 CPUs, 3 experiments). Restarted after 3 experiments freed memory.
- Visual verification of all 8 dashboard views via agent-browser with REAL results: overview (stats + honest verdicts), benchmark (9-model metric table, ROC/PR/calibration/DCA tabs), advantage (certificate: No Demonstrated Advantage, QSVM 0.9933 vs LR 0.9943, entanglement ablation 0.9974 vs 0.9851, dequantization), explainability (SHAP global on original features), prediction (cascade), training (live pima job), datasets, reports (conformal 90%->91.1% coverage, cascade 4.2% routed). No console errors.
- Frontend tests: 10/10 pass (bun test). Backend: unit 30/30, integration 5/5 after engine fix.
- Screenshots saved to download/screenshots/ (overview, benchmark, benchmark_roc, advantage, explainability, prediction, training, datasets, reports).

Stage Summary:
- 3 experiments COMPLETE with real results (breast_cancer, heart_disease, parkinsons) — all honest verdicts: No Demonstrated Advantage (QSVM close on breast cancer: 0.9933 vs 0.9943 classical LR).
- exp_pima_diabetes running (~51%), ETA ~20 min.
- Remaining: pima completion + n_samples patch, final 4-experiment verification, README/docs final check, Complete tool.

---
Task ID: 18 (delivery)
Agent: main (Super Z)
Task: Package project as zip for user download.

Work Log:
- exp_pima_diabetes completed all artifacts (result.json, certificate.json, report.md, explainability.json, models.pkl, pipeline.pkl, conformal.pkl) at 20:26. n_samples correct (768). All 4 datasets now have complete real results.
- Built download/qednet-2.0-project.zip (6.9MB, 326 files): full backend source, frontend source, configs, datasets, ALL 4 experiment results/certificates/reports/artifacts, docs, scripts, Docker files.
- Excluded: node_modules, .next, .git, skills (internal), upload, db, caches.

Stage Summary:
- Project delivered as single zip. Best results: breast_cancer QSVM 0.9933 vs LR 0.9943 (No Demonstrated Advantage — honest), heart_disease/parkinsons/pima all complete with certificates.
