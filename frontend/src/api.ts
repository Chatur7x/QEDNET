const API_ORIGIN = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);
const BASE = `${API_ORIGIN}/research`;

async function req(path: string, opts?: RequestInit) {
  const r = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  health: () => req("/health"),
  datasets: () => req("/datasets"),
  models: () => req("/models"),
  train: (config: object, compare: string[]) =>
    req("/train", { method: "POST", body: JSON.stringify({ config, compare }) }),
  experiment: (id: string) => req(`/experiments/${id}`),
  experiments: () => req("/experiments"),
  predict: (experiment_id: string, features: number[]) =>
    req("/predict", { method: "POST", body: JSON.stringify({ experiment_id, features }) }),
  explain: (experiment_id: string, features: number[]) =>
    req("/explain", { method: "POST", body: JSON.stringify({ experiment_id, features }) }),
  benchmark: (id: string) => req(`/benchmark/${id}`),
  certificate: (id: string) => req(`/certificate/${id}`),
};
