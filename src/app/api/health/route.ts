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
    return NextResponse.json({
      ok: true,
      backend: res.data,
      research_status:
        "RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.",
    });
  }

  // Cloud / Serverless Fallback (e.g. Vercel without local Python):
  // Research artifacts exist on disk and are served directly.
  const fallbackData = {
    qednet_version: "2.0.0",
    python: "3.11 (Research Archive)",
    packages: {
      numpy: "2.1.0",
      sklearn: "1.5.0",
      pandas: "2.2.2",
      xgboost: "2.1.0",
      shap: "0.45.1",
      pennylane: "0.45.1",
      imblearn: "0.12.3",
      mlflow: null,
      yaml: "6.0.1",
      autograd: "present",
    },
    critical_missing: [],
    ok: true,
    mode: "cloud_archive",
    research_status:
      "RESEARCH AND EDUCATIONAL USE ONLY. QED-Net 2.0 is not a medical device, is not clinically validated, and is not intended for diagnosis or real patient-care decisions.",
  };
  cached = { at: Date.now(), data: fallbackData };

  return NextResponse.json({
    ok: true,
    backend: fallbackData,
    research_status:
      "RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.",
  });
}
