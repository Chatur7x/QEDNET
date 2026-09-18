import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { cli, RUNS_DIR, getExperimentFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const res = await cli(["get-experiment", "--id", id], 120_000);
  const reportPath = path.join(RUNS_DIR, id, "report.md");
  const report = fs.existsSync(reportPath) ? fs.readFileSync(reportPath, "utf-8") : null;

  if (res.ok && (res.data as any)?.result) {
    return NextResponse.json({
      ok: true,
      result: (res.data as any).result,
      report,
    });
  }

  const fallback = getExperimentFallback(id);
  if (fallback) {
    return NextResponse.json({
      ok: true,
      result: fallback,
      report,
    });
  }

  return NextResponse.json({
    ok: false,
    result: null,
    report,
    error: res.error ?? (res.data as any)?.error ?? "Experiment not found",
  }, { status: 404 });
}

