import { NextResponse } from "next/server";
import { cli, isSafeId } from "@/lib/qednet";
import { requireResearchAccess } from "@/lib/research-access";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const denied = await requireResearchAccess();
  if (denied) return denied;
  const body = await req.json().catch(() => null);
  if (!body || typeof body.experimentId !== "string" || !isSafeId(body.experimentId)
    || !Array.isArray(body.features) || body.features.length === 0
    || !body.features.every((value: unknown) => typeof value === "number" && Number.isFinite(value))) {
    return NextResponse.json(
      { ok: false, error: "experimentId and features[] are required" },
      { status: 400 },
    );
  }
  const input = JSON.stringify(body.features);
  const res = await cli(
    ["predict", "--id", String(body.experimentId), "--input-json", input],
    180_000,
  );
  if (res.ok && res.data && (res.data as any).ok !== false) {
    return NextResponse.json({
      ok: true,
      prediction: res.data,
    });
  }

  return NextResponse.json(
    { ok: false, prediction: null, error: res.error || "Prediction unavailable; no measured model result was returned" },
    { status: 503 },
  );
}

