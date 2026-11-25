import { NextResponse } from "next/server";
import { cli } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const dataset = body?.dataset;
  if (typeof dataset !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(dataset)) {
    return NextResponse.json({ ok: false, error: "dataset is required" }, { status: 400 });
  }
  const res = await cli(["validate", "--dataset", dataset], 120_000);
  if (res.ok && res.data) {
    return NextResponse.json({ ok: true, report: res.data });
  }
  return NextResponse.json(
    { ok: false, report: null, error: res.error || "Dataset validation unavailable" },
    { status: 503 },
  );
}

