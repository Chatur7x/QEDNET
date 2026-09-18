import { NextResponse } from "next/server";
import { cli } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get("id");
  if (!id) {
    return NextResponse.json({ ok: false, error: "id is required" }, { status: 400 });
  }
  const res = await cli(["certificate", "--id", id], 120_000);
  return NextResponse.json({
    ok: res.ok,
    certificate: (res.data as any)?.certificate ?? null,
    markdown: (res.data as any)?.markdown ?? null,
    error: res.error ?? (res.data as any)?.error,
  });
}
