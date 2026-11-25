/**
 * QED-Net Python bridge.
 *
 * All backend intelligence lives in the Python package (backend/qednet).
 * Next.js API routes invoke it through `python3 -m qednet.cli ...` which
 * emits strict JSON on stdout. Long-running training jobs are spawned
 * detached and tracked through job descriptor files.
 */
import { execFile, spawn } from "child_process";
import { randomUUID } from "crypto";
import fs from "fs";
import path from "path";
import os from "os";

const ROOT = process.cwd();

function getWritableDir(defaultRelativePath: string, tmpSubdir: string): string {
  const target = path.join(ROOT, defaultRelativePath);
  try {
    fs.mkdirSync(target, { recursive: true });
    const testFile = path.join(target, `.write_test_${Date.now()}`);
    fs.writeFileSync(testFile, "1");
    fs.unlinkSync(testFile);
    return target;
  } catch {
    const tmp = path.join(os.tmpdir(), "qednet", tmpSubdir);
    try {
      fs.mkdirSync(tmp, { recursive: true });
    } catch {}
    return tmp;
  }
}

export function getJobsDir(): string {
  return getWritableDir(path.join("experiments", "jobs"), "jobs");
}

export function getUploadsDir(): string {
  return getWritableDir(path.join("data", "uploads"), "uploads");
}


/**
 * Resolve the Python interpreter for the CLI bridge.
 * Prefers, in order: QEDNET_PYTHON env var, a project-local venv, the user's
 * venv (~/.venv), and finally bare `python3` from PATH. This makes the
 * dashboard independent of the PATH the dev server happened to start with.
 */
function resolvePython(): string {
  if (process.env.QEDNET_PYTHON) return process.env.QEDNET_PYTHON;
  const candidates = [
    path.join(ROOT, ".venv", "bin", "python3"),
    path.join(ROOT, ".venv", "bin", "python"),
    path.join(process.env.HOME ?? "/home/z", ".venv", "bin", "python3"),
  ];
  for (const c of candidates) {
    try {
      if (fs.existsSync(c) && fs.accessSync(c, fs.constants.X_OK) === undefined) {
        return c;
      }
    } catch {
      /* not usable — next candidate */
    }
  }
  return "python3";
}

const PYTHON = resolvePython();
const PYTHONPATH = path.join(ROOT, "backend");

export const JOBS_DIR = path.join(ROOT, "experiments", "jobs");
export const RUNS_DIR = path.join(ROOT, "experiments", "runs");
export const UPLOADS_DIR = path.join(ROOT, "data", "uploads");
export const CONFIGS_DIR = path.join(ROOT, "configs", "experiments");

export function isSafeId(value: string): boolean {
  return /^[A-Za-z0-9_-]{1,128}$/.test(value);
}

export interface CliResult<T = unknown> {
  ok: boolean;
  data: T | null;
  error?: string;
  stderrTail?: string;
}

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

/** Run a qednet CLI command and parse its JSON stdout payload. */
export function cli<T = unknown>(args: string[], timeoutMs = 120_000): Promise<CliResult<T>> {
  return new Promise((resolve) => {
    execFile(
      PYTHON,
      ["-m", "qednet.cli", ...args],
      {
        cwd: ROOT,
        env: { ...process.env, PYTHONPATH },
        timeout: timeoutMs,
        maxBuffer: 64 * 1024 * 1024,
      },
      (err, stdout, stderr) => {
        const lastJson = lastJsonLine(stdout);
        if (lastJson) {
          const data = lastJson as T & { ok?: boolean; error?: string };
          const dataSaysNo = typeof data.ok === "boolean" && !data.ok;
          resolve({
            ok: !dataSaysNo,
            data,
            error: dataSaysNo ? data.error : undefined,
            stderrTail: tail(stderr),
          });
          return;
        }
        resolve({
          ok: false,
          data: null,
          error: err ? String(err.message) : "no JSON output from backend",
          stderrTail: tail(stderr),
        });
      }
    );
  });
}

function lastJsonLine(stdout: string): unknown | null {
  const lines = stdout.trim().split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.startsWith("{") && line.endsWith("}")) {
      try {
        return JSON.parse(line);
      } catch {
        /* keep scanning */
      }
    }
  }
  return null;
}

function tail(s: string | undefined, n = 800): string | undefined {
  if (!s) return undefined;
  return s.length > n ? s.slice(-n) : s;
}

export interface JobDescriptor {
  jobId: string;
  experimentId: string;
  configName: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  pid: number | null;
  startedAt: number;
  endedAt: number | null;
  logFile: string;
  error?: string;
  isSimulated?: boolean;
}

// In-memory registry of jobs and uploaded datasets for serverless resilience
const memoryJobs = new Map<string, JobDescriptor>();
const memoryDatasets = new Map<string, Record<string, unknown>>();

/** Spawn a real detached training job; unavailable runtimes fail explicitly. */
export function startTrainingJob(configName: string, experimentId: string): JobDescriptor {
  const jobsDir = getJobsDir();
  const jobId = `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
  const logFile = path.join(jobsDir, `${jobId}.log`);

  let child: any = null;
  let spawnFailed = false;

  try {
    const out = fs.openSync(logFile, "a");
    child = spawn(
      PYTHON,
      ["-m", "qednet.cli", "run-experiment", "--config",
        path.join(CONFIGS_DIR, `${configName}.yaml`), "--exp-id", experimentId],
      {
        cwd: ROOT,
        env: { ...process.env, PYTHONPATH },
        detached: true,
        stdio: ["ignore", out, out],
      }
    );
    child.unref();
    fs.closeSync(out);
  } catch {
    spawnFailed = true;
  }

  const started = !spawnFailed && !!child?.pid;

  const job: JobDescriptor = {
    jobId,
    experimentId,
    configName,
    status: started ? "running" : "failed",
    pid: child?.pid ?? null,
    startedAt: Date.now(),
    endedAt: null,
    logFile,
    isSimulated: false,
    error: started ? undefined : "Python training process could not be started",
  };

  memoryJobs.set(jobId, job);
  try {
    fs.writeFileSync(path.join(jobsDir, `${jobId}.json`), JSON.stringify(job, null, 2));
  } catch {}

  return job;
}

/** Cancel a running job. */
export function cancelJob(jobId: string): boolean {
  const jobsDir = getJobsDir();
  const job = memoryJobs.get(jobId);
  if (job) {
    if (job.pid) {
      try {
        process.kill(job.pid);
      } catch {}
    }
    job.status = "cancelled";
    job.endedAt = Date.now();
    memoryJobs.set(jobId, job);
    try {
      fs.writeFileSync(path.join(jobsDir, `${jobId}.json`), JSON.stringify(job, null, 2));
    } catch {}
    return true;
  }
  return false;
}

/** Read a job descriptor and refresh its live status from the process table or simulation timeline. */
export function refreshJob(job: JobDescriptor): JobDescriptor {
  if (job.isSimulated) {
    return { ...job, status: "failed", endedAt: job.endedAt || Date.now(), error: "Simulated job record is not a measured run" };
  }

  const alive = job.pid ? isPidAlive(job.pid) : false;
  const resultExists = fs.existsSync(
    path.join(RUNS_DIR, job.experimentId, "result.json")
  );
  let status = job.status;
  if (job.status === "running" || job.status === "queued") {
    if (alive) status = "running";
    else if (resultExists) status = "completed";
    else status = "failed";
  }
  let error: string | undefined = job.error;
  if (status === "failed" && !error) {
    const log = readLogTail(job.logFile, 400);
    error = log || "training process exited without a result";
  }
  return { ...job, status, endedAt: status === "running" ? null : Date.now(), error };
}

export function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: unknown) {
    const code = (e as NodeJS.ErrnoException).code;
    return code === "EPERM"; // alive but not owned
  }
}

export function readLogTail(logFile: string, n = 1500): string | undefined {
  try {
    if (fs.existsSync(logFile)) {
      const text = fs.readFileSync(logFile, "utf-8");
      if (text.length > 0) return text.length > n ? text.slice(-n) : text;
    }
  } catch {}

  return undefined;
}

export function listJobs(): JobDescriptor[] {
  const jobsDir = getJobsDir();
  const seenIds = new Set<string>();
  const out: JobDescriptor[] = [];

  // 1. Scan filesystem jobs
  const dirs = [JOBS_DIR, jobsDir];
  for (const dir of dirs) {
    if (!fs.existsSync(dir)) continue;
    try {
      for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith(".json")) continue;
        try {
          const job = JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8"));
          if (job?.jobId && !seenIds.has(job.jobId)) {
            seenIds.add(job.jobId);
            out.push(refreshJob(job));
          }
        } catch {}
      }
    } catch {}
  }

  // 2. Scan memory jobs
  for (const job of memoryJobs.values()) {
    if (!seenIds.has(job.jobId)) {
      seenIds.add(job.jobId);
      out.push(refreshJob(job));
    }
  }

  return out.sort((a, b) => b.startedAt - a.startedAt);
}

/** Read per-fold progress written live by the runner or simulated timeline. */
export function readProgress(experimentId: string): Record<string, unknown> | null {
  try {
    const disk = path.join(RUNS_DIR, experimentId, "progress.json");
    if (fs.existsSync(disk)) {
      return JSON.parse(fs.readFileSync(disk, "utf-8"));
    }
  } catch {}

  return null;
}

export async function saveUploadedCsv(file: File): Promise<string> {
  const uploadsDir = getUploadsDir();
  const safe = file.name.replace(/[^a-zA-Z0-9_.-]+/g, "_").slice(0, 80);
  const dest = path.join(uploadsDir, `${randomUUID()}_${safe}`);
  const bytes = Buffer.from(await file.arrayBuffer());
  if (bytes.byteLength > MAX_UPLOAD_BYTES) throw new Error("File exceeds the upload size limit");
  fs.writeFileSync(dest, bytes);
  return dest;
}

export const DATA_REGISTRY_PATH = path.join(ROOT, "data", "registry", "datasets.json");

export function registerDatasetFallback(dataset: Record<string, unknown>): void {
  const name = String(dataset.name || `dataset_${Date.now()}`);
  memoryDatasets.set(name, dataset);
}

export function getExperimentsFallback(): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  const existingIds = new Set<string>();

  if (fs.existsSync(RUNS_DIR)) {
    try {
      for (const d of fs.readdirSync(RUNS_DIR)) {
        const dirPath = path.join(RUNS_DIR, d);
        if (!fs.statSync(dirPath).isDirectory()) continue;
        const resultPath = path.join(dirPath, "result.json");
        const certPath = path.join(dirPath, "certificate.json");
        const cfgPath = path.join(dirPath, "config.yaml");
        if (!fs.existsSync(resultPath)) continue;
        try {
          const result = JSON.parse(fs.readFileSync(resultPath, "utf-8"));
          const stat = fs.statSync(dirPath);
          const entry: Record<string, unknown> = {
            experiment_id: d,
            dataset: result.dataset,
            n_samples: result.n_samples,
            n_features: result.n_features,
            models: Object.keys(result.models || {}),
            protocol: result.protocol,
            runtime_s: result.runtime_s,
            created: stat.mtimeMs,
            has_certificate: fs.existsSync(certPath),
            config_file: fs.existsSync(cfgPath) ? cfgPath : null,
            validation_passed: result.validation?.passed,
          };
          if (fs.existsSync(certPath)) {
            try {
              const cert = JSON.parse(fs.readFileSync(certPath, "utf-8"));
              entry.evidence_classification = cert.final_evidence_classification;
            } catch {}
          }
          existingIds.add(d);
          out.push(entry);
        } catch {}
      }
    } catch {}
  }

  return out.sort((a, b) => Number(b.created || 0) - Number(a.created || 0));
}

export function getExperimentFallback(id: string): Record<string, unknown> | null {
  const resultPath = path.join(RUNS_DIR, id, "result.json");
  if (fs.existsSync(resultPath)) {
    try {
      return JSON.parse(fs.readFileSync(resultPath, "utf-8"));
    } catch {}
  }

  return null;
}

export function getCertificateFallback(id: string): { certificate: Record<string, unknown>; markdown: string | null } | null {
  const certPath = path.join(RUNS_DIR, id, "certificate.json");
  const reportPath = path.join(RUNS_DIR, id, "certificate_report.md");
  if (fs.existsSync(certPath)) {
    try {
      const certificate = JSON.parse(fs.readFileSync(certPath, "utf-8"));
      const markdown = fs.existsSync(reportPath) ? fs.readFileSync(reportPath, "utf-8") : null;
      return { certificate, markdown };
    } catch {}
  }

  return null;
}

export function getExplainFallback(id: string): Record<string, unknown> | null {
  const explainPath = path.join(RUNS_DIR, id, "explainability.json");
  if (fs.existsSync(explainPath)) {
    try {
      return JSON.parse(fs.readFileSync(explainPath, "utf-8"));
    } catch {}
  }

  return null;
}

export function getDatasetsFallback(): Array<Record<string, unknown>> {
  const list: Array<Record<string, unknown>> = [];
  if (fs.existsSync(DATA_REGISTRY_PATH)) {
    try {
      const data = JSON.parse(fs.readFileSync(DATA_REGISTRY_PATH, "utf-8")) as unknown;
      if (data && typeof data === "object" && !Array.isArray(data)) {
        list.push(...Object.values(data).filter(
          (value): value is Record<string, unknown> =>
            !!value && typeof value === "object" && !Array.isArray(value),
        ));
      }
    } catch {}
  }

  for (const d of memoryDatasets.values()) {
    list.push(d);
  }

  return list;
}

export function getValidationFallback(_dataset: string): null { return null; }

export { ROOT };


