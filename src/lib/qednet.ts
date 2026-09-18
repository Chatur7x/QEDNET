/**
 * QED-Net Python bridge.
 *
 * All backend intelligence lives in the Python package (backend/qednet).
 * Next.js API routes invoke it through `python3 -m qednet.cli ...` which
 * emits strict JSON on stdout. Long-running training jobs are spawned
 * detached and tracked through job descriptor files.
 */
import { execFile, spawn } from "child_process";
import fs from "fs";
import path from "path";

const ROOT = process.cwd();

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

export interface CliResult<T = unknown> {
  ok: boolean;
  data: T | null;
  error?: string;
  stderrTail?: string;
}

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
}

/** Spawn a detached training job; returns its descriptor. */
export function startTrainingJob(configName: string, experimentId: string): JobDescriptor {
  fs.mkdirSync(JOBS_DIR, { recursive: true });
  const jobId = `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
  const logFile = path.join(JOBS_DIR, `${jobId}.log`);
  const out = fs.openSync(logFile, "a");
  const child = spawn(
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

  const job: JobDescriptor = {
    jobId,
    experimentId,
    configName,
    status: "running",
    pid: child.pid ?? null,
    startedAt: Date.now(),
    endedAt: null,
    logFile,
  };
  fs.writeFileSync(path.join(JOBS_DIR, `${jobId}.json`), JSON.stringify(job, null, 2));
  return job;
}

/** Read a job descriptor and refresh its live status from the process table. */
export function refreshJob(job: JobDescriptor): JobDescriptor {
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
    const text = fs.readFileSync(logFile, "utf-8");
    return text.length > n ? text.slice(-n) : text;
  } catch {
    return undefined;
  }
}

export function listJobs(): JobDescriptor[] {
  fs.mkdirSync(JOBS_DIR, { recursive: true });
  const out: JobDescriptor[] = [];
  for (const f of fs.readdirSync(JOBS_DIR)) {
    if (!f.endsWith(".json")) continue;
    try {
      const job = JSON.parse(fs.readFileSync(path.join(JOBS_DIR, f), "utf-8"));
      out.push(refreshJob(job));
    } catch {
      /* skip corrupt descriptors */
    }
  }
  return out.sort((a, b) => b.startedAt - a.startedAt);
}

/** Read per-fold progress written live by the runner (if available). */
export function readProgress(experimentId: string): Record<string, unknown> | null {
  try {
    return JSON.parse(
      fs.readFileSync(path.join(RUNS_DIR, experimentId, "progress.json"), "utf-8")
    );
  } catch {
    return null;
  }
}

export async function saveUploadedCsv(file: File): Promise<string> {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  const safe = file.name.replace(/[^a-zA-Z0-9_.-]+/g, "_").slice(0, 80);
  const dest = path.join(UPLOADS_DIR, `${Date.now().toString(36)}_${safe}`);
  const bytes = Buffer.from(await file.arrayBuffer());
  fs.writeFileSync(dest, bytes);
  return dest;
}

export { ROOT };
