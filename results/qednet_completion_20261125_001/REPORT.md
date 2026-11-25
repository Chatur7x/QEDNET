# QEDNET completion work log

## Changes completed locally

- Updated Next.js to `16.4.6`, matching `eslint-config-next`, and Sharp to `0.35.5` per the published Next.js image-optimization advisory.
- Migrated Prisma from v6 to v7.10.1 with the matching SQLite `better-sqlite3` adapter, moved the SQLite URL to `prisma.config.ts`, and removed `--accept-data-loss` from the normal `db:push` script. The app start command now uses Node for the standalone build; the native SQLite adapter is not supported by Bun at runtime.
- Kept Python backend `DATABASE_URL` separate from the optional Prisma SQLite variable `QEDNET_PRISMA_DATABASE_URL`.
- Added `results` to TypeScript excludes so audit scripts in the ignored evidence folder do not break application typechecking.
- Updated the root frontend route smoke tests to reflect the current auth-protected QEDNET app and explicitly exclude the preserved, separate project.
- Existing user folders `frontend/`, `runs/`, and `web/` were left unstaged and untouched.

## Verification evidence

Exact commands, raw outputs, and exit codes are alongside this report.

| Check | Result |
|---|---|
| `bun audit` | Passed; no vulnerabilities reported in the audited install |
| `bun install --frozen-lockfile` | Passed |
| `bun x tsc --noEmit` | Passed |
| `bun run lint` | Passed with 33 warnings and no errors |
| `bun run build` | Passed; Next reports 8 dynamic-filesystem tracing warnings for the Python CLI bridge |
| `bun run test` against the local standalone server | 13 passed, 0 failed |
| `python -m pytest` | 86 passed; 83 dependency deprecation warnings |
| Prisma generation and schema validation | Passed |
| Prisma `db:push` on a new temporary SQLite file | Passed through the Node CLI |
| Prisma create/read/delete smoke against that temporary file | Passed |
| Existing `prisma/prisma/dev.db` | SHA-256 before and after was identical; no migration or query was run against it |
| `git diff --check` | Passed |
| Render current deployment | Existing service is live on the free plan. `/live` returned HTTP 200; `/signin` and `/signup` returned HTTP 200; `/` and `/api/health` redirected unauthenticated requests to sign-in. |

The `bun audit` earlier in this task identified a Prisma 6 transitive advisory. After the supported Prisma 7 migration, the final full audit reports no vulnerabilities.

## Not completed / NOT MEASURED

- The newly verified code has not yet been deployed by this report's verification run. Render's current live deployment is the preceding commit. A deploy can be verified only after publishing the tested changes.
- Local Docker/container build is NOT MEASURED because the Docker daemon is unavailable. Render's Docker build will be checked after the authorized push/deploy step.
- Supabase email confirmation, password reset, and a signed-in research API session are NOT MEASURED. Public sign-in/sign-up pages render, but no confirmation email was sent and no test account was authenticated.
- Render free filesystem storage is ephemeral. Durable uploads/experiment artifacts, backups, and restore are NOT MEASURED or configured.
- Early-detection and clinical utility are NOT MEASURED: bundled data are cross-sectional; there is no authorized longitudinal cohort, defined horizon, prospective study, clinical partner, or regulatory determination.
- Quantum hardware and noisy-shot runs are NOT MEASURED.
- Full Prisma migration of any existing SQLite user database is NOT MEASURED. The tracked local SQLite file was deliberately left untouched; only a new throwaway database was used.
- Production runtime isolation and user-specific artifact storage remain incomplete. The existing Render service uses one allowlisted account and must not be opened to multiple users without tenant-isolated storage.

## External service state observed

Render CLI is authenticated. The QEDNET service is configured to deploy from `main` on the free plan. Local credentials are not present in this shell. The current checkout is based on the same commit as `origin/main` before this work, but contains uncommitted changes.

No secret values were written to the evidence files. Previously exposed Supabase credentials should be rotated and set only through the provider dashboards/secret manager.

## Release boundary

This is a verified local research-platform update, not a fully production-ready clinical product. Do not make clinical, early-detection, quantum-advantage, durability, or authenticated-email-delivery claims from these checks alone.
