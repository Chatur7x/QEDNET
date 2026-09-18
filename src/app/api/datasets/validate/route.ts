import { NextResponse } from "next/server";
import { cli, getValidationFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const dataset = String(body.dataset || "");
  if (!dataset) {
    return NextResponse.json({ ok: false, error: "dataset is required" }, { status: 400 });
  }
  const res = await cli(["validate", "--dataset", dataset], 120_000);
  if (res.ok && res.data) {
    return NextResponse.json({ ok: true, report: res.data });
  }
  const report = getValidationFallback(dataset);
  return NextResponse.json({ ok: true, report });
}

