import { NextResponse } from "next/server";
import { cli } from "@/lib/qednet";

export const dynamic = "force-dynamic";

export async function GET() {
  const res = await cli(["list-experiments"], 60_000);
  return NextResponse.json({
    ok: res.ok,
    experiments: (res.data as any)?.experiments ?? [],
    error: res.error,
  });
}
