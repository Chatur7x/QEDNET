import { describe, expect, test } from "bun:test";
import { NextRequest } from "next/server";
import { hasProductionAuthConfig, isAllowedEmail } from "../src/lib/auth-policy";
import { proxy } from "../src/proxy";

describe("private email account policy", () => {
  test("production auth needs HTTPS Supabase settings and one allowed email", () => {
    expect(hasProductionAuthConfig({})).toBe(false);
    expect(hasProductionAuthConfig({
      NEXT_PUBLIC_SITE_URL: "http://research.example.org",
      NEXT_PUBLIC_SUPABASE_URL: "https://project.supabase.co",
      NEXT_PUBLIC_SUPABASE_ANON_KEY: "test-key",
      QEDNET_ALLOWED_EMAIL: "owner@example.org",
    })).toBe(false);
    expect(hasProductionAuthConfig({
      NEXT_PUBLIC_SITE_URL: "https://research.example.org",
      NEXT_PUBLIC_SUPABASE_URL: "https://project.supabase.co",
      NEXT_PUBLIC_SUPABASE_ANON_KEY: "test-key",
      QEDNET_ALLOWED_EMAIL: "owner@example.org",
    })).toBe(true);
  });

  test("only the configured email is authorized, case-insensitively", () => {
    const env = { QEDNET_ALLOWED_EMAIL: "Owner@Example.org" };
    expect(isAllowedEmail("owner@example.org", env)).toBe(true);
    expect(isAllowedEmail("other@example.org", env)).toBe(false);
    expect(isAllowedEmail(null, env)).toBe(false);
    expect(isAllowedEmail("owner@example.org", {})).toBe(false);
  });

  test("production proxy locks the workspace until auth is configured", async () => {
    const keys = ["NEXT_PUBLIC_SITE_URL", "NEXT_PUBLIC_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_ANON_KEY", "QEDNET_ALLOWED_EMAIL"];
    const env = process.env as Record<string, string | undefined>;
    const previous = Object.fromEntries(keys.map((key) => [key, env[key]]));
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
});
