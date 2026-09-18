import { NextResponse } from "next/server";
import { cli } from "@/lib/qednet";

export const dynamic = "force-dynamic";

// The environment snapshot is immutable per process — cache it so health
// checks stay fast while training jobs load the CPUs.
let cached: { at: number; data: unknown } | null = null;
const CACHE_MS = 10 * 60 * 1000;

export async function GET() {
  if (cached && Date.now() - cached.at < CACHE_MS) {
    return NextResponse.json({
      ok: true,
      backend: cached.data,
      research_status:
        "RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.",
      cached: true,
    });
  }
  const res = await cli(["env"], 120_000);
  // The CLI's env command reports ok:false when critical packages are
  // missing (e.g. wrong Python interpreter) — surface that honestly, and
  // don't cache degraded results so recovery is detected on the next call.
  const backendOk =
    res.ok && !!res.data && (res.data as { ok?: boolean }).ok !== false;
  if (backendOk) {
    cached = { at: Date.now(), data: res.data };
  }
  return NextResponse.json({
    ok: backendOk,
    backend: res.data,
    error: res.error,
    research_status:
      "RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.",
  });
}
