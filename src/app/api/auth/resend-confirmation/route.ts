import { NextResponse } from "next/server";
import { hasProductionAuthConfig, isAllowedEmail } from "@/lib/auth-policy";
import { createSupabaseServerClient } from "@/lib/supabase/server";

export async function POST(request: Request) {
  if (!hasProductionAuthConfig()) {
    return NextResponse.json({ error: "Sign-up is not configured." }, { status: 503 });
  }

  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  if (!email || email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ error: "Enter a valid email address." }, { status: 400 });
  }
  if (!isAllowedEmail(email)) {
    return NextResponse.json({ error: "This email is not enabled for this private research workspace." }, { status: 403 });
  }

  const supabase = await createSupabaseServerClient();
  const redirectTo = new URL("/auth/confirm?next=%2F", process.env.NEXT_PUBLIC_SITE_URL).toString();
  const { error } = await supabase.auth.resend({
    type: "signup",
    email,
    options: { emailRedirectTo: redirectTo },
  });

  if (error) {
    const rateLimited = error.status === 429 || /rate limit/i.test(error.message);
    return NextResponse.json(
      { error: rateLimited ? "Confirmation email limit reached. Wait before trying again." : "A confirmation email could not be sent." },
      { status: rateLimited ? 429 : 400 },
    );
  }

  return NextResponse.json({ ok: true });
}
