import { NextResponse } from "next/server";
import { hasProductionAuthConfig, isAllowedEmail } from "@/lib/auth-policy";

export async function POST(request: Request) {
  if (!hasProductionAuthConfig()) {
    return NextResponse.json({ ok: false, error: "Sign-up is not configured" }, { status: 503 });
  }

  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  if (!email || email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ ok: false, error: "Enter a valid email address." }, { status: 400 });
  }
  if (!isAllowedEmail(email)) {
    return NextResponse.json({ ok: false, error: "This email is not enabled for this private research workspace." }, { status: 403 });
  }

  return NextResponse.json({ ok: true });
}
