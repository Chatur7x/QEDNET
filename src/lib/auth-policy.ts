export function hasProductionAuthConfig(env: Record<string, string | undefined> = process.env) {
  if (!env.NEXT_PUBLIC_SUPABASE_ANON_KEY || !env.QEDNET_ALLOWED_EMAIL) return false;
  try {
    return new URL(env.NEXT_PUBLIC_SITE_URL ?? "").protocol === "https:"
      && new URL(env.NEXT_PUBLIC_SUPABASE_URL ?? "").protocol === "https:";
  } catch {
    return false;
  }
}

export function isAllowedEmail(
  email?: string | null,
  env: Record<string, string | undefined> = process.env,
) {
  return Boolean(email && env.QEDNET_ALLOWED_EMAIL
    && email.trim().toLowerCase() === env.QEDNET_ALLOWED_EMAIL.trim().toLowerCase());
}
