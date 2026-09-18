import { NextResponse } from "next/server";
import { cli } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const dataset = String(body.dataset || "");
  if (!dataset) {
    return NextResponse.json({ ok: false, error: "dataset is required" }, { status: 400 });
  }
  const res = await cli(["validate", "--dataset", dataset], 120_000);
  return NextResponse.json({ ok: res.ok, report: res.data, error: res.error });
}
