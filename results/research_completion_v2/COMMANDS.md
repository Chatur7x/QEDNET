# QEDNET incremental verification log

Run from `D:\Projects\QEDNT` in PowerShell. Raw outputs are saved beside this file. The parent `results/` directory is ignored by Git; keep this evidence folder with the corresponding source revision when sharing results.

| Exact command | Raw output | Outcome |
|---|---|---|
| `$env:PYTHONPATH='backend'; python -m pytest backend/tests/unit/test_cli_validation.py -q` | `backend_validation_persistence.raw.txt` | Passed; registry persistence and Parkinson's validation regression tests. |
| `$env:PYTHONPATH='backend'; python -m pytest backend/tests/unit/test_api_hardening.py backend/tests/unit/test_cli_validation.py -q` | `security_and_validation_tests.raw.txt` | Passed; generic API error responses and registry persistence. |
| `$env:PYTHONPATH='backend'; python -m pytest backend/tests -q` | `backend_pytest_final.raw.txt` | Final full backend suite after the security and persistence changes. Existing sklearn and SHAP warnings are in the log. |
| `bun run lint` | `frontend_lint.raw.txt` | Exit 0; 0 errors and 33 warnings. |
| `bun run build` | `frontend_build_final.raw.txt` | Exit 0; final local production build passed. |
| `$env:PORT='3100'; $env:HOSTNAME='127.0.0.1'; bun run start` | `local_server_final.raw.txt` | Standalone app started locally and was stopped after the smoke check. |
| `$env:QEDNET_BASE_URL='http://127.0.0.1:3100'; bun run test` | `frontend_smoke_final.raw.txt` | 14 passed, 0 failed; auth variables absent, so tests cover fail-closed protected routes, not signed-in account behavior. |
| `docker compose config --quiet` | `compose_config.raw.txt` | Exit 0; Compose syntax/configuration only, no containers started. |
| `git diff --check` | `git_diff_check.raw.txt` | Exit 0. |
| `$ErrorActionPreference='Stop'; $text=Get-Content Caddyfile -Raw; if($text -match 'XTransformPort|\{query\.') { throw 'Untrusted query value controls a proxy target' }; if($text -notmatch 'path /socket\.io/\*' -or $text -notmatch 'reverse_proxy localhost:3003') { throw 'Fixed websocket route missing' }; 'CADDY_STATIC_SECURITY_CHECK=PASS' | Set-Content results/research_completion_v2/caddy_static_check.raw.txt; Get-Content results/research_completion_v2/caddy_static_check.raw.txt` | `caddy_static_check.raw.txt` | Passed source assertions only; Caddy syntax/runtime was not measured. |

The legacy Caddy reverse-proxy configuration was not executed: `caddy` is not installed in this environment. Its upstream target is fixed in source; runtime parsing remains `NOT MEASURED`.

No CSV outputs were created. These checks do not establish live deployment behavior, durable storage, multi-user isolation, backup/restore, or clinical validity.
