import { NextResponse } from "next/server";
import { cli, getCertificateFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get("id");
  if (!id) {
    return NextResponse.json({ ok: false, error: "id is required" }, { status: 400 });
  }
  const res = await cli(["certificate", "--id", id], 120_000);
  if (res.ok && (res.data as any)?.certificate) {
    return NextResponse.json({
      ok: true,
      certificate: (res.data as any).certificate,
      markdown: (res.data as any).markdown,
    });
  }

  const fallback = getCertificateFallback(id);
  if (fallback) {
    return NextResponse.json({
      ok: true,
      certificate: fallback.certificate,
      markdown: fallback.markdown,
    });
  }

  return NextResponse.json({
    ok: false,
    certificate: null,
    markdown: null,
    error: res.error ?? (res.data as any)?.error ?? "Certificate not found",
  }, { status: 404 });
}

