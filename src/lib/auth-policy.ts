import type { Profile } from "next-auth";

export function hasProductionAuthConfig(env: Record<string, string | undefined> = process.env) {
  if (!env.NEXTAUTH_SECRET || !env.OIDC_CLIENT_ID || !env.OIDC_CLIENT_SECRET || !env.OIDC_ALLOWED_GROUP) {
    return false;
  }
  try {
    return new URL(env.NEXTAUTH_URL ?? "").protocol === "https:"
      && new URL(env.OIDC_ISSUER ?? "").protocol === "https:";
  } catch {
    return false;
  }
}

export function isAllowedOrganizationProfile(
  profile?: Profile | null,
  env: Record<string, string | undefined> = process.env,
) {
  const allowedGroup = env.OIDC_ALLOWED_GROUP;
  const groupClaim = env.OIDC_GROUP_CLAIM || "groups";
  if (!allowedGroup || !profile) return false;
  const value = (profile as Profile & Record<string, unknown>)[groupClaim];
  return Array.isArray(value)
    ? value.includes(allowedGroup)
    : value === allowedGroup;
}
