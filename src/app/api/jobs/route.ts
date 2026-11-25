import { NextResponse } from "next/server";
import fs from "fs";
import { listJobs, startTrainingJob, CONFIGS_DIR, readProgress, refreshJob, cancelJob, isSafeId } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET() {
  const jobs = listJobs().map((job) => ({
    ...job,
    progress: readProgress(job.experimentId),
  }));
  return NextResponse.json({ ok: true, jobs });
}

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const configName = body?.configName;
  const experimentId = body?.experimentId;
  if (typeof configName !== "string" || !isSafeId(configName)
      || typeof experimentId !== "string" || !isSafeId(experimentId)) {
    return NextResponse.json(
      { ok: false, error: "configName and experimentId are required" },
      { status: 400 },
    );
  }
  const cfgPath = `${CONFIGS_DIR}/${configName}.yaml`;
  if (!fs.existsSync(cfgPath)) {
    return NextResponse.json({ ok: false, error: `Unknown config: ${configName}` }, { status: 404 });
  }
  // refuse duplicate live jobs for the same experiment id
  const running = listJobs().find(
    (j) => j.experimentId === experimentId && refreshJob(j).status === "running"
  );
  if (running) {
    return NextResponse.json(
      { ok: false, error: `Experiment ${experimentId} is already running (job ${running.jobId})` },
      { status: 409 },
    );
  }
  const job = startTrainingJob(configName, experimentId);
  if (job.status === "failed") {
    return NextResponse.json({ ok: false, job, error: job.error }, { status: 503 });
  }
  return NextResponse.json({ ok: true, job: { ...job, progress: readProgress(experimentId) } });
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const jobId = searchParams.get("jobId") || (await req.json().catch(() => ({}))).jobId;
  if (!jobId) {
    return NextResponse.json({ ok: false, error: "jobId is required" }, { status: 400 });
  }
  if (typeof jobId !== "string" || !isSafeId(jobId)) {
    return NextResponse.json({ ok: false, error: "Invalid job ID" }, { status: 400 });
  }
  const cancelled = cancelJob(String(jobId));
  return NextResponse.json({ ok: true, cancelled });
}

