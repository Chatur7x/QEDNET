/**
 * QED-Net Python bridge.
 *
 * All backend intelligence lives in the Python package (backend/qednet).
 * Next.js API routes invoke it through `python -m qednet.cli ...` which
 * emits strict JSON on stdout. Long-running training jobs are spawned
 * detached and tracked through job descriptor files.
 *
 * Phase 0: fabricated fallback/simulation paths removed. Routes must surface
 * explicit unavailable/error behavior when the Python backend or artifacts
 * are not available.
 */
import { execFile, spawn } from "child_process";
import { randomUUID } from "crypto";
import fs from "fs";
import os from "os";
import path from "path";

// Next standalone can start from its generated directory; keep all Python,
// configuration, and writable runtime paths anchored at the container root.
export const ROOT = process.env.QEDNET_WORKSPACE_ROOT || process.cwd();

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

export const PYTHONPATH = path.join(ROOT, "backend");
export const CONFIGS_DIR = path.join(ROOT, "configs", "experiments");
export const RUNS_DIR = path.join(ROOT, "experiments", "runs");
export const JOBS_DIR = path.join(ROOT, "experiments", "jobs");
export const UPLOADS_DIR = path.join(ROOT, "data", "uploads");
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export function isSafeId(value: string): boolean {
  return /^[A-Za-z0-9_-]{1,128}$/.test(value);
}

export function getJobsDir(): string {
  return getWritableDir(path.join("experiments", "jobs"), "jobs");
}
export function getRunsDir(): string {
  return getWritableDir(path.join("experiments", "runs"), "runs");
}

for (const dir of [RUNS_DIR, JOBS_DIR, UPLOADS_DIR]) {
  try { fs.mkdirSync(dir, { recursive: true }); } catch {}
}

const PYTHON =
  process.platform === "win32"
    ? (fs.existsSync(path.join(ROOT, ".venv", "Scripts", "python.exe"))
        ? path.join(ROOT, ".venv", "Scripts", "python.exe")
        : "python")
    : "python3";

export interface CliResult {
  ok: boolean;
  data?: unknown;
  error?: string;
  stderr?: string;
  code?: number | null;
}

export function cli(args: string[], timeoutMs = 300_000): Promise<CliResult> {
  const isCloud = !!process.env.VERCEL;
  const cwd = isCloud ? "/tmp" : PYTHONPATH;

  return new Promise((resolve) => {
    execFile(
      PYTHON,
      ["-m", "qednet.cli", ...args],
      {
        cwd,
        env: { ...process.env, PYTHONPATH },
        maxBuffer: 50 * 1024 * 1024,
        timeout: timeoutMs,
      },
      (error, stdout, stderr) => {
        const out = stdout?.toString().trim() ?? "";
        let parsed: any = undefined;
        if (out) {
          try { parsed = JSON.parse(out); } catch { parsed = out; }
        }
        if (error) {
          resolve({
            ok: false,
            data: parsed,
            error: error.message,
            stderr: stderr?.toString(),
            code: (error as any).code ?? null,
          });
          return;
        }
        const ok = parsed && typeof parsed === "object" ? parsed.ok !== false : true;
        resolve({ ok, data: parsed, stderr: stderr?.toString() || undefined });
      }
    );
  });
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

export function startTrainingJob(configName: string, experimentId: string): JobDescriptor {
  const jobsDir = getJobsDir();
  const jobId = `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
  const logFile = path.join(jobsDir, `${jobId}.log`);
  let child: ReturnType<typeof spawn> | null = null;
  let fd: number | null = null;
  try {
    fd = fs.openSync(logFile, "a");
    child = spawn(
      PYTHON,
      ["-m", "qednet.cli", "run-experiment", "--config",
        path.join(CONFIGS_DIR, `${configName}.yaml`), "--exp-id", experimentId],
      {
        cwd: PYTHONPATH,
        env: { ...process.env, PYTHONPATH },
        detached: true,
        stdio: ["ignore", fd, fd],
      }
    );
    child.unref();
  } finally {
    if (fd !== null) {
      try { fs.closeSync(fd); } catch {}
    }
  }

  const job: JobDescriptor = {
    jobId,
    experimentId,
    configName,
    status: child?.pid ? "running" : "failed",
    pid: child?.pid ?? null,
    startedAt: Date.now(),
    endedAt: child?.pid ? null : Date.now(),
    logFile,
    error: child?.pid ? undefined : "Failed to spawn Python training process",
  };
  try {
    fs.writeFileSync(path.join(jobsDir, `${jobId}.json`), JSON.stringify(job, null, 2));
  } catch {}
  return job;
}

export function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: unknown) {
    return (e as NodeJS.ErrnoException).code === "EPERM";
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

export function readProgress(experimentId: string): Record<string, unknown> | null {
  if (!isSafeId(experimentId)) return null;
  try {
    const p = path.join(RUNS_DIR, experimentId, "progress.json");
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {}
  return null;
}

export function refreshJob(job: JobDescriptor): JobDescriptor {
  if (job.status === "cancelled") return job;
  if (!isSafeId(job.experimentId)) {
    return { ...job, status: "failed", endedAt: Date.now(), error: "Invalid experiment ID" };
  }
  const alive = job.pid ? isPidAlive(job.pid) : false;
  const resultExists = fs.existsSync(path.join(RUNS_DIR, job.experimentId, "result.json"));
  let status = job.status;
  if (job.status === "running" || job.status === "queued") {
    if (alive) status = "running";
    else if (resultExists) status = "completed";
    else status = "failed";
  }
  let error = job.error;
  if (status === "failed" && !error) {
    error = readLogTail(job.logFile, 400) || "training process exited without a result";
  }
  return {
    ...job,
    status,
    endedAt: status === "running" ? null : (job.endedAt ?? Date.now()),
    error,
  };
}

export function listJobs(): JobDescriptor[] {
  const seen = new Set<string>();
  const out: JobDescriptor[] = [];
  for (const dir of [JOBS_DIR, getJobsDir()]) {
    if (!fs.existsSync(dir)) continue;
    try {
      for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith(".json")) continue;
        try {
          const job = JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8")) as JobDescriptor;
          if (job?.jobId && !seen.has(job.jobId)) {
            seen.add(job.jobId);
            const refreshed = refreshJob(job);
            out.push(refreshed);
            try { fs.writeFileSync(path.join(dir, f), JSON.stringify(refreshed, null, 2)); } catch {}
          }
        } catch {}
      }
    } catch {}
  }
  return out.sort((a, b) => b.startedAt - a.startedAt);
}

export function saveUploadedCsv(filename: string, content: string): string {
  const safe = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
  if (Buffer.byteLength(content, "utf-8") > MAX_UPLOAD_BYTES) {
    throw new Error("File exceeds the upload size limit");
  }
  const dest = path.join(UPLOADS_DIR, `${randomUUID()}_${safe}`);
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  fs.writeFileSync(dest, content, "utf-8");
  return dest;
}
