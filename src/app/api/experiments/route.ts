import { NextResponse } from "next/server";
import { cli, getExperimentsFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET() {
  const res = await cli(["list-experiments"], 60_000);
  if (res.ok && Array.isArray((res.data as any)?.experiments) && (res.data as any).experiments.length > 0) {
    return NextResponse.json({
      ok: true,
      experiments: (res.data as any).experiments,
    });
  }
  const fallback = getExperimentsFallback();
  if (res.ok) return NextResponse.json({ ok: true, experiments: fallback });
  return NextResponse.json({ ok: false, experiments: fallback, error: res.error || "Experiment registry unavailable" }, { status: 503 });
}

