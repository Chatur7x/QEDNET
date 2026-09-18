import { NextResponse } from "next/server";
import { cli } from "@/lib/qednet";

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
  return NextResponse.json({
    ok: (res.data as any)?.ok ?? false,
    prediction: res.data,
    error: res.error ?? (res.data as any)?.error,
  });
}
