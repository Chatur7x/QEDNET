import { NextResponse } from "next/server";
import { hasProductionAuthConfig, isAllowedEmail } from "@/lib/auth-policy";
import { createSupabaseServerClient } from "@/lib/supabase/server";

/** Route-level authorization for research data and job APIs.
 * The Next.js Proxy is an early redirect only; sensitive handlers verify the
 * current Supabase user themselves before touching files or running jobs.
 */
export async function requireResearchAccess(): Promise<NextResponse | null> {
  if (process.env.NODE_ENV !== "production") return null;
  if (!hasProductionAuthConfig()) {
    return NextResponse.json(
      { ok: false, error: "Research workspace authentication is not configured." },
      { status: 503 },
    );
  }

  try {
    const supabase = await createSupabaseServerClient();
    const { data: { user }, error } = await supabase.auth.getUser();
    if (error || !user) {
      return NextResponse.json(
        { ok: false, error: "Authentication required." }, { status: 401 },
      );
    }
    if (!user.email_confirmed_at || !isAllowedEmail(user.email)) {
      return NextResponse.json(
        { ok: false, error: "This account is not authorized for the research workspace." },
        { status: 403 },
      );
    }
    return null;
  } catch {
    return NextResponse.json(
      { ok: false, error: "Research workspace authentication could not be verified." },
      { status: 503 },
    );
  }
}
