import { NextResponse } from "next/server";
import fs from "fs";
import { listJobs, startTrainingJob, CONFIGS_DIR, readProgress, refreshJob } from "@/lib/qednet";

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
  const configName = String(body.configName || "").replace(/[^a-z0-9_-]/gi, "");
  const experimentId = String(body.experimentId || "").replace(/[^a-zA-Z0-9_-]/g, "");
  if (!configName || !experimentId) {
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
  return NextResponse.json({ ok: true, job: { ...job, progress: readProgress(experimentId) } });
}
