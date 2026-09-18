/**
 * QED-Net 2.0 frontend integration tests (bun test).
 *
 * Exercises the real dashboard surface: page rendering, API integration,
 * loading/error behaviour and the Python-backend bridge. These tests run
 * against the dev server (port 3000) — they verify the full stack,
 * including the Python core. Python-backed calls get generous timeouts.
 *
 * Run: bun test tests/qednet-api.test.ts
 */
import { describe, expect, test } from "bun:test";

const BASE = process.env.QEDNET_BASE_URL || "http://localhost:3000";
const PY_TIMEOUT = 45_000; // python CLI bridge can be slow under training load

async function get(path: string) {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  const text = await res.text();
  return { status: res.status, text };
}

async function getJson(path: string) {
  const { status, text } = await get(path);
  try {
    return { status, json: JSON.parse(text) as any };
  } catch {
    return { status, json: null as any, text };
  }
}

describe("dashboard page rendering", () => {
  test(
    "GET / renders the QED-Net dashboard shell",
    async () => {
      const { status, text } = await get("/");
      expect(status).toBe(200);
      expect(text).toContain("QED-Net 2.0");
      expect(text).toContain("Hybrid Quantum");
    },
    30_000
  );

  test(
    "page contains all eight view sections in navigation",
    async () => {
      const { text } = await get("/");
      for (const view of [
        "Overview",
        "Datasets",
        "Training",
        "Prediction",
        "Explainability",
        "Benchmark",
        "Quantum Advantage",
        "Research Reports",
      ]) {
        expect(text).toContain(view);
      }
    },
    30_000
  );

  test(
    "research-only safety status is always visible",
    async () => {
      const { text } = await get("/");
      expect(text).toContain("research");
      expect(text.toLowerCase()).toContain("not a medical device");
    },
    30_000
  );
});

describe("API integration (Python backend bridge)", () => {
  test(
    "GET /api/health returns live backend environment",
    async () => {
      const { status, json } = await getJson("/api/health");
      expect(status).toBe(200);
      expect(json.ok).toBe(true);
      expect(json.backend?.python).toBeTruthy();
      expect(json.backend?.packages?.pennylane).toBeTruthy();
      expect(json.research_status).toContain("not a medical device");
    },
    PY_TIMEOUT
  );

  test(
    "GET /api/datasets lists registered open datasets",
    async () => {
      const { status, json } = await getJson("/api/datasets");
      expect(status).toBe(200);
      expect(json.ok).toBe(true);
      const names = (json.datasets ?? []).map((d: any) => d.name);
      expect(names).toContain("breast_cancer");
      expect(names).toContain("heart_disease");
      for (const d of json.datasets ?? []) {
        expect(d.n_samples).toBeGreaterThan(0);
        expect(d.n_features).toBeGreaterThan(0);
        expect(d.profile?.class_balance).toBeDefined();
      }
    },
    PY_TIMEOUT
  );

  test(
    "GET /api/experiments returns a list",
    async () => {
      const { status, json } = await getJson("/api/experiments");
      expect(status).toBe(200);
      expect(Array.isArray(json.experiments)).toBe(true);
    },
    PY_TIMEOUT
  );

  test(
    "GET /api/jobs returns tracked background jobs",
    async () => {
      const { status, json } = await getJson("/api/jobs");
      expect(status).toBe(200);
      expect(Array.isArray(json.jobs)).toBe(true);
      for (const job of json.jobs ?? []) {
        expect(["queued", "running", "completed", "failed", "cancelled"]).toContain(job.status);
      }
    },
    PY_TIMEOUT
  );

  test(
    "invalid experiment id returns a graceful error, not a stack trace",
    async () => {
      const { status, json } = await getJson("/api/experiments/does_not_exist");
      expect(status).toBe(200);
      expect(json.ok).toBe(false);
      expect(json.error).toBeTruthy();
      expect(String(json.error)).not.toContain("Traceback");
    },
    PY_TIMEOUT
  );
});

describe("dataset validation endpoint", () => {
  test(
    "validate rejects unknown dataset with a clear message",
    async () => {
      const res = await fetch(`${BASE}/api/datasets/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: "no_such_dataset" }),
      });
      const json = await res.json();
      expect(json.ok === false || json.report?.passed === false).toBe(true);
    },
    PY_TIMEOUT
  );
});

describe("predict endpoint input validation", () => {
  test(
    "predict rejects malformed requests",
    async () => {
      const res = await fetch(`${BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: "not-an-array" }),
      });
      expect(res.status).toBe(400);
      const json = await res.json();
      expect(json.ok).toBe(false);
    },
    PY_TIMEOUT
  );
});
