import { NextResponse } from "next/server";
import { cli, predictFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!body || !body.experimentId || !Array.isArray(body.features)) {
    return NextResponse.json(
      { ok: false, error: "experimentId and features[] are required" },
      { status: 400 },
    );
  }
  const input = JSON.stringify(body.features.map((v: unknown) => Number(v)));
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

  const fallback = predictFallback(String(body.experimentId), body.features.map(Number));
  return NextResponse.json({
    ok: true,
    prediction: fallback,
  });
}

