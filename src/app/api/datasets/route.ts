import { NextResponse } from "next/server";
import fs from "fs";
import { cli, saveUploadedCsv, UPLOADS_DIR, getDatasetsFallback } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET() {
  const res = await cli(["list-datasets"], 60_000);
  if (res.ok && Array.isArray((res.data as any)?.datasets) && (res.data as any).datasets.length > 0) {
    return NextResponse.json({ ok: true, datasets: (res.data as any).datasets });
  }
  const fallback = getDatasetsFallback();
  return NextResponse.json({ ok: true, datasets: fallback });
}


export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json({ ok: false, error: "CSV file is required" }, { status: 400 });
    }
    if (file.size > 25 * 1024 * 1024) {
      return NextResponse.json({ ok: false, error: "File exceeds 25 MB limit" }, { status: 400 });
    }
    if (!/\.(csv|tsv)$/i.test(file.name)) {
      return NextResponse.json(
        { ok: false, error: "Only CSV/TSV files are accepted" },
        { status: 400 },
      );
    }
    const target = (form.get("target") as string) || undefined;
    const name = (form.get("name") as string) || undefined;
    const dest = await saveUploadedCsv(file);
    const res = await cli(
      ["upload", "--csv", dest, ...(target ? ["--target", target] : []), ...(name ? ["--name", name] : [])],
      120_000,
    );
    return NextResponse.json({
      ok: (res.data as any)?.ok ?? false,
      dataset: (res.data as any)?.dataset,
      validation: (res.data as any)?.validation,
      error: res.error ?? (res.data as any)?.error,
    });
  } catch (e) {
    return NextResponse.json({ ok: false, error: String(e) }, { status: 500 });
  }
}
