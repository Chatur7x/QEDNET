import { NextResponse, type NextRequest } from "next/server";
import type { EmailOtpType } from "@supabase/supabase-js";
import { createSupabaseServerClient } from "@/lib/supabase/server";

const emailOtpTypes = new Set<EmailOtpType>([
  "email", "signup", "invite", "magiclink", "recovery", "email_change",
]);

function safeNextPath(value: string | null) {
  if (!value?.startsWith("/") || value.startsWith("//")) return "/";
  const base = new URL("https://qednet.invalid");
  const target = new URL(value, base);
  return target.origin === base.origin ? `${target.pathname}${target.search}${target.hash}` : "/";
}

function publicUrl(path: string, request: NextRequest) {
  return new URL(path, process.env.NEXT_PUBLIC_SITE_URL ?? request.nextUrl.origin);
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get("code");
  const tokenHash = searchParams.get("token_hash");
  const type = searchParams.get("type") as EmailOtpType | null;
  const next = safeNextPath(request.nextUrl.searchParams.get("next"));
  if (!code && !(tokenHash && type && emailOtpTypes.has(type))) {
    return NextResponse.redirect(publicUrl("/signin?message=confirmation-failed", request));
  }

  const supabase = await createSupabaseServerClient();
  const { error } = tokenHash && type
    ? await supabase.auth.verifyOtp({ token_hash: tokenHash, type })
    : await supabase.auth.exchangeCodeForSession(code!);
  if (error) return NextResponse.redirect(publicUrl("/signin?message=confirmation-failed", request));
  return NextResponse.redirect(publicUrl(next, request));
}
