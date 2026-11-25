/**
 * API client + shared types for the QED-Net 2.0 research dashboard.
 * Server state via TanStack Query; all data comes from REAL experiment
 * results produced by the Python backend (never fabricated).
 */
"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

/* ----------------------------- types ----------------------------- */

export interface DatasetProfile {
  n_samples: number;
  n_features: number;
  feature_names: string[];
  dtypes: Record<string, string>;
  class_balance: Record<string, number>;
  missing_per_column: Record<string, number>;
  duplicate_rows: number;
  constant_columns: string[];
  identifier_like_columns: string[];
  memory_mb: number;
}

export interface DatasetMeta {
  name: string;
  source: string;
  target: string;
  path?: string | null;
  sha256?: string | null;
  n_samples: number;
  n_features: number;
  profile: DatasetProfile;
  validation_status: string;
  notes?: string;
}

export interface MetricSummary {
  mean: number | null;
  std: number;
  n_obs: number;
}

export interface ModelResult {
  model: string;
  family: "quantum" | "classical" | "unknown";
  summary: Record<string, MetricSummary>;
  bootstrap_auroc: { ci_low: number; ci_high: number; point: number; n_resamples: number } | null;
  curve_roc: { fpr: number[]; tpr: number[] };
  curve_pr: { precision: number[]; recall: number[] };
  curve_calibration: { centers: number[]; accuracy: number[] };
  decision_curve: { thresholds: number[]; model: number[]; treat_all: number[] } | null;
  resources: Record<string, unknown> | null;
  model_meta: Record<string, unknown>;
  param_count: number | null;
  train_time_s: { mean: number; total: number };
  folds: { seed: number; fold: number; metrics: Record<string, number | null> }[];
  pooled_predictions: { y: number[]; p: number[]; p_calibrated: number[] };
  leakage_audit: { fitted_parts: string[]; contamination_detected: boolean };
}

export interface ExperimentSummary {
  experiment_id: string;
  dataset: string;
  n_samples: number;
  n_features: number;
  models: string[];
  protocol: Record<string, unknown>;
  runtime_s: number;
  created: number;
  has_certificate: boolean;
  evidence_classification?: string;
  validation_passed?: boolean;
}

export interface ExperimentResult {
  dataset: string;
  n_samples: number;
  n_features: number;
  feature_names: string[];
  validation: Record<string, unknown> & { passed?: boolean };
  protocol: Record<string, unknown>;
  models: Record<string, ModelResult>;
  statistics: Record<string, Record<string, unknown>>;
  runtime_s: number;
  dataset_fingerprint?: string;
  experiment_id?: string;
  research_status?: string;
  ablation?: Record<string, unknown> | null;
  dequantization?: Record<string, unknown> | null;
  bottleneck_honesty_meter?: Record<string, unknown> | null;
  cascade?: Record<string, unknown> | null;
  conformal?: Record<string, unknown> | null;
  explainability_summary?: Record<string, unknown>;
  reproducibility?: Record<string, unknown>;
  mlflow_logged?: boolean;
  total_runtime_s?: number;
}

export interface ExplainEntry {
  model: string;
  shap: {
    available: boolean;
    method?: string;
    global_mean_abs_shap?: number[];
    local_shap_values?: number[][];
    base_value?: number;
    background_size?: number;
    nsamples?: number;
  };
  permutation: {
    available: boolean;
    importances_mean?: number[];
    importances_std?: number[];
    method?: string;
    n_repeats?: number;
  };
  sensitivity: {
    available: boolean;
    feature_names?: string[];
    mean_abs_effect?: number[];
    method?: string;
  };
  quantum_diagnostics?: Record<string, unknown>;
  counterfactual?: Record<string, unknown>;
}

export interface PredictionResult {
  ok?: boolean;
  stage?: string;
  route?: "classical" | "quantum";
  classical_screening_probability?: number;
  model_used?: string;
  probability_used?: number;
  prediction?: number | null;
  confidence?: number;
  uncertainty_entropy?: number;
  abstention?: { abstain: boolean; reasons: Record<string, boolean> };
  status?: string;
  abstention_rule?: Record<string, unknown>;
  experiment_id?: string;
}

export interface JobView {
  jobId: string;
  experimentId: string;
  configName: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  pid: number | null;
  startedAt: number;
  endedAt: number | null;
  logFile: string;
  error?: string;
  progress?: {
    stage: string;
    seed?: number | null;
    fold?: number | null;
    model?: string | null;
    completed_fits?: number | null;
    auroc?: number;
    seeds_total?: number;
    fits_per_seed?: number;
  } | null;
  logTail?: string;
}

export interface HealthInfo {
  ok: boolean;
  backend?: {
    qednet_version: string;
    python: string;
    packages: Record<string, string | null>;
    research_status: string;
  };
  research_status: string;
}

/* ----------------------------- fetchers ----------------------------- */

async function jget<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${url} failed (${res.status})`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthInfo> {
  return jget("/api/health");
}
export async function fetchDatasets(): Promise<DatasetMeta[]> {
  const d = await jget<{ datasets: DatasetMeta[] }>("/api/datasets");
  return d.datasets ?? [];
}
export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const d = await jget<{ experiments: ExperimentSummary[] }>("/api/experiments");
  return d.experiments ?? [];
}
export async function fetchExperiment(id: string): Promise<{ result: ExperimentResult; report: string | null }> {
  return jget(`/api/experiments/${encodeURIComponent(id)}`);
}
export async function fetchExplain(id: string): Promise<Record<string, ExplainEntry>> {
  const d = await jget<{ explainability: Record<string, ExplainEntry> }>(
    `/api/explain?id=${encodeURIComponent(id)}`
  );
  return d.explainability ?? {};
}
export async function fetchCertificate(id: string): Promise<{ certificate: Record<string, any>; markdown: string | null }> {
  return jget(`/api/certificate?id=${encodeURIComponent(id)}`);
}
export async function fetchJobs(): Promise<JobView[]> {
  const d = await jget<{ jobs: JobView[] }>("/api/jobs");
  return d.jobs ?? [];
}

/* ----------------------------- hooks ----------------------------- */

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: fetchHealth, refetchInterval: 60_000 });
}
export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: fetchDatasets, refetchInterval: 30_000 });
}
export function useExperiments() {
  return useQuery({ queryKey: ["experiments"], queryFn: fetchExperiments, refetchInterval: 15_000 });
}
export function useExperiment(id: string | null) {
  return useQuery({
    queryKey: ["experiment", id],
    queryFn: () => fetchExperiment(id as string),
    enabled: !!id,
  });
}
export function useExplain(id: string | null) {
  return useQuery({
    queryKey: ["explain", id],
    queryFn: () => fetchExplain(id as string),
    enabled: !!id,
    staleTime: 5 * 60_000,
  });
}
export function useCertificate(id: string | null) {
  return useQuery({
    queryKey: ["certificate", id],
    queryFn: () => fetchCertificate(id as string),
    enabled: !!id,
    staleTime: 5 * 60_000,
  });
}
export function useJobs() {
  return useQuery({ queryKey: ["jobs"], queryFn: fetchJobs, refetchInterval: 3_000 });
}

export function useStartTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: { configName: string; experimentId: string }) => {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.json();
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const res = await fetch(`/api/jobs?jobId=${encodeURIComponent(jobId)}`, {
        method: "DELETE",
      });
      return res.json();
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}


export function usePredict() {
  return useMutation({
    mutationFn: async (body: { experimentId: string; features: number[] }) => {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.json() as Promise<{ ok: boolean; prediction?: PredictionResult; error?: string }>;
    },
  });
}

export function useUploadDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (form: FormData) => {
      const res = await fetch("/api/datasets", { method: "POST", body: form });
      return res.json();
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useValidateDataset() {
  return useMutation({
    mutationFn: async (dataset: string) => {
      const res = await fetch("/api/datasets/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset }),
      });
      return res.json();
    },
  });
}

/* ----------------------------- helpers ----------------------------- */

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  rbf_svm: "RBF-SVM",
  matched_mlp: "Matched MLP",
  qsvm: "QSVM",
  vqc: "VQC",
  reupload: "Data Re-Uploading",
  hybrid_qnn: "Hybrid QNN",
};

export function familyOf(model: string): "quantum" | "classical" {
  // Accept both raw ids ("vqc") and display labels ("VQC") — views pass
  // display labels into chart components, which color series by family.
  const QUANTUM_IDS = new Set(["qsvm", "vqc", "reupload", "hybrid_qnn"]);
  if (QUANTUM_IDS.has(model)) return "quantum";
  for (const [id, label] of Object.entries(MODEL_LABELS)) {
    if (label === model) return QUANTUM_IDS.has(id) ? "quantum" : "classical";
  }
  return "classical";
}

export function modelColor(model: string): string {
  return familyOf(model) === "quantum" ? "#7c3aed" : "#0f766e";
}

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function seconds(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (v < 1) return `${(v * 1000).toFixed(0)} ms`;
  if (v < 90) return `${v.toFixed(1)} s`;
  if (v < 5400) return `${(v / 60).toFixed(1)} min`;
  return `${(v / 3600).toFixed(1)} h`;
}
