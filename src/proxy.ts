import { NextResponse, type NextRequest } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { hasProductionAuthConfig, isAllowedEmail } from "@/lib/auth-policy";

const publicPaths = new Set([
  "/signin", "/signup", "/access-denied",
  "/auth/confirm", "/forgot-password", "/api/auth/resend-confirmation",
]);

export async function proxy(request: NextRequest) {
  if (process.env.NODE_ENV !== "production") return NextResponse.next();

  const path = request.nextUrl.pathname;
  if (path === "/live" || path === "/setup-required") {
    return NextResponse.next();
  }

  if (!hasProductionAuthConfig()) {
    return NextResponse.redirect(new URL("/setup-required", request.url));
  }

  if (publicPaths.has(path) || path === "/api/auth/signup") return NextResponse.next();

  let response = NextResponse.next({ request });
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll: () => request.cookies.getAll(),
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value));
          response = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) => response.cookies.set(name, value, options));
        },
      },
    },
  );

  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    const signInUrl = new URL("/signin", request.url);
    signInUrl.searchParams.set("callbackUrl", `${path}${request.nextUrl.search}`);
    return NextResponse.redirect(signInUrl);
  }

  if (!user.email_confirmed_at || !isAllowedEmail(user.email)) {
    return NextResponse.redirect(new URL("/access-denied", request.url));
  }

  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
