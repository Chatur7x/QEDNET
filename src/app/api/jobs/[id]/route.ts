import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { JOBS_DIR, readProgress, readLogTail, refreshJob } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const jobPath = path.join(JOBS_DIR, `${id}.json`);
  if (!fs.existsSync(jobPath)) {
    return NextResponse.json({ ok: false, error: "job not found" }, { status: 404 });
  }
  const job = refreshJob(JSON.parse(fs.readFileSync(jobPath, "utf-8")));
  fs.writeFileSync(jobPath, JSON.stringify(job, null, 2));
  return NextResponse.json({
    ok: true,
    job: {
      ...job,
      progress: readProgress(job.experimentId),
      logTail: job.status === "running" || job.status === "failed"
        ? readLogTail(job.logFile)
        : undefined,
    },
  });
}
