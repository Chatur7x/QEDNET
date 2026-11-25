import { describe, expect, test } from "bun:test";
import { NextRequest } from "next/server";
import { encode } from "next-auth/jwt";
import type { Profile } from "next-auth";
import { hasProductionAuthConfig, isAllowedOrganizationProfile } from "../src/lib/auth-policy";
import { proxy } from "../src/proxy";

describe("organization authentication configuration", () => {
  test("production auth requires every credential and the organization group allowlist", () => {
    expect(hasProductionAuthConfig({})).toBe(false);
    expect(hasProductionAuthConfig({
      NEXTAUTH_URL: "http://research.example.org",
      NEXTAUTH_SECRET: "test-only-secret",
      OIDC_ISSUER: "https://id.example.org",
      OIDC_CLIENT_ID: "qednet",
      OIDC_CLIENT_SECRET: "test-only-client-secret",
    })).toBe(false);
    expect(hasProductionAuthConfig({
      NEXTAUTH_URL: "https://research.example.org",
      NEXTAUTH_SECRET: "test-only-secret",
      OIDC_ISSUER: "https://id.example.org",
      OIDC_CLIENT_ID: "qednet",
      OIDC_CLIENT_SECRET: "test-only-client-secret",
      OIDC_ALLOWED_GROUP: "research-team",
    })).toBe(true);
  });

  test("only exact organization group claims are allowed", () => {
    process.env.OIDC_ALLOWED_GROUP = "research-team";
    delete process.env.OIDC_GROUP_CLAIM;
    expect(isAllowedOrganizationProfile({ sub: "member", groups: ["research-team"] } as Profile)).toBe(true);
    expect(isAllowedOrganizationProfile({ sub: "outsider", groups: ["other-team"] } as Profile)).toBe(false);
    expect(isAllowedOrganizationProfile({ sub: "missing-claim" })).toBe(false);
    delete process.env.OIDC_ALLOWED_GROUP;
  });

  test("production proxy locks an unconfigured deployment", async () => {
    const keys = ["NEXTAUTH_URL", "NEXTAUTH_SECRET", "OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "OIDC_ALLOWED_GROUP"];
    const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
    const env = process.env as Record<string, string | undefined>;
    const previousNodeEnv = env.NODE_ENV;
    for (const key of keys) delete env[key];
    env.NODE_ENV = "production";
    try {
      const response = await proxy(new NextRequest("https://research.example.org/api/jobs"));
      expect(response.headers.get("location")).toBe("https://research.example.org/setup-required");
    } finally {
      for (const key of keys) {
        if (previous[key] === undefined) delete env[key];
        else env[key] = previous[key];
      }
      if (previousNodeEnv === undefined) delete env.NODE_ENV;
      else env.NODE_ENV = previousNodeEnv;
    }
  });

  test("production proxy requires an authorized session for research routes", async () => {
    const values: Record<string, string> = {
      NODE_ENV: "production",
      NEXTAUTH_URL: "https://research.example.org",
      NEXTAUTH_SECRET: "test-only-secret",
      OIDC_ISSUER: "https://id.example.org",
      OIDC_CLIENT_ID: "qednet",
      OIDC_CLIENT_SECRET: "test-only-client-secret",
      OIDC_ALLOWED_GROUP: "research-team",
    };
    const env = process.env as Record<string, string | undefined>;
    const previous = Object.fromEntries(Object.keys(values).map((key) => [key, env[key]]));
    Object.assign(env, values);
    try {
      const anonymous = await proxy(new NextRequest("https://research.example.org/api/jobs"));
      expect(anonymous.headers.get("location")).toBe("https://research.example.org/signin?callbackUrl=%2Fapi%2Fjobs");

      const token = await encode({
        token: { sub: "member", qednetAuthorized: true },
        secret: values.NEXTAUTH_SECRET,
        maxAge: 60,
      });
      const authenticated = await proxy(new NextRequest("https://research.example.org/api/jobs", {
        headers: { cookie: `__Secure-next-auth.session-token=${token}` },
      }));
      expect(authenticated.headers.get("x-middleware-next")).toBe("1");
    } finally {
      for (const [key, value] of Object.entries(previous)) {
        if (value === undefined) delete env[key];
        else env[key] = value;
      }
    }
  });
});
