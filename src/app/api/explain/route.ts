import { NextResponse } from "next/server";
import { cli, getExplainFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const id = url.searchParams.get("id");
  const model = url.searchParams.get("model");
  if (!id) {
    return NextResponse.json({ ok: false, error: "id is required" }, { status: 400 });
  }
  const args = ["explain", "--id", id, ...(model ? ["--model", model] : [])];
  const res = await cli(args, 300_000);
  if (res.ok && (res.data as any)?.explainability) {
    return NextResponse.json({
      ok: true,
      explainability: (res.data as any).explainability,
    });
  }

  const fallback = getExplainFallback(id);
  if (fallback) {
    return NextResponse.json({
      ok: true,
      explainability: fallback,
    });
  }

  return NextResponse.json({
    ok: false,
    explainability: null,
    error: res.error ?? (res.data as any)?.error ?? "Explainability data not found",
  }, { status: 404 });
}

