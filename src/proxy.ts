import { getToken } from "next-auth/jwt";
import { NextResponse, type NextRequest } from "next/server";
import { hasProductionAuthConfig } from "@/lib/auth-policy";

const publicPaths = new Set(["/live", "/signin", "/setup-required"]);

export async function proxy(request: NextRequest) {
  if (process.env.NODE_ENV !== "production") return NextResponse.next();

  const path = request.nextUrl.pathname;
  if (publicPaths.has(path) || path === "/api/auth" || path.startsWith("/api/auth/")) {
    return NextResponse.next();
  }

  if (!hasProductionAuthConfig()) {
    return NextResponse.redirect(new URL("/setup-required", request.url));
  }

  const token = await getToken({ req: request, secret: process.env.NEXTAUTH_SECRET });
  if (token?.qednetAuthorized === true) {
    if (path === "/signin") return NextResponse.redirect(new URL("/", request.url));
    return NextResponse.next();
  }

  const signInUrl = new URL("/signin", request.url);
  signInUrl.searchParams.set("callbackUrl", `${path}${request.nextUrl.search}`);
  return NextResponse.redirect(signInUrl);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
