import { NextResponse } from "next/server";
import { cli, saveUploadedCsv, MAX_UPLOAD_BYTES } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET() {
  const res = await cli(["list-datasets"], 60_000);
  if (res.ok && Array.isArray((res.data as any)?.datasets)) {
    return NextResponse.json({ ok: true, datasets: (res.data as any).datasets });
  }
  return NextResponse.json(
    { ok: false, datasets: [], error: res.error || "Dataset registry unavailable" },
    { status: 503 },
  );
}

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json({ ok: false, error: "CSV/TSV file is required" }, { status: 400 });
    }
    if (!/\.(csv|tsv)$/i.test(file.name)) {
      return NextResponse.json({ ok: false, error: "Only CSV/TSV files are accepted" }, { status: 400 });
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return NextResponse.json({ ok: false, error: "File exceeds the upload size limit" }, { status: 413 });
    }

    const target = String(form.get("target") || "");
    const name = String(form.get("name") || "");
    const csvPath = await saveUploadedCsv(file);
    const args = ["upload", "--csv", csvPath];
    if (target) args.push("--target", target);
    if (name) args.push("--name", name);
    const res = await cli(args, 120_000);
    if (!res.ok || !res.data || (res.data as any).ok === false) {
      return NextResponse.json(
        { ok: false, error: res.error || (res.data as any)?.error || "Dataset registration unavailable" },
        { status: 503 },
      );
    }
    return NextResponse.json(res.data);
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Dataset upload failed" },
      { status: 400 },
    );
  }
}
