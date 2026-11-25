/**
 * Production-build smoke tests for the real QEDNET routes.
 * Run against a local server with QEDNET_BASE_URL set.
 * Protected research endpoints must remain locked unless auth is configured.
 */
import { describe, expect, test } from "bun:test";

const BASE = process.env.QEDNET_BASE_URL || "http://localhost:3000";

async function request(path: string, method = "GET") {
  return fetch(`${BASE}${path}`, {
    method,
    redirect: "manual",
    headers: { "content-type": "application/json" },
    body: method === "POST" ? "{}" : undefined,
  });
}

describe("QEDNET production route smoke", () => {
  test("public liveness endpoint identifies research-only state", async () => {
    const response = await request("/live");
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      status: "ok",
      research_only: true,
      model_loaded: false,
    });
  });

  test("workspace does not render without an authenticated session", async () => {
    const response = await request("/");
    expect([307, 308]).toContain(response.status);
    expect(response.headers.get("location")).toMatch(/\/signin|\/setup-required/);
  });

  for (const [path, method] of [
    ["/api/health", "GET"],
    ["/api/datasets", "GET"],
    ["/api/experiments", "GET"],
    ["/api/jobs", "GET"],
    ["/api/datasets/validate", "POST"],
    ["/api/predict", "POST"],
  ] as const) test(`protected route ${method} ${path} stays behind auth`, async () => {
    const response = await request(path, method);
    expect([307, 308, 401, 403]).toContain(response.status);
    if (response.status === 307 || response.status === 308) {
      expect(response.headers.get("location")).toMatch(/\/signin|\/setup-required/);
    }
  });
});
