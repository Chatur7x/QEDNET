/**
 * Prediction view — F12 cascade with F11 uncertainty & abstention.
 * Pick an experiment, set feature values, run the second-opinion cascade.
 */
"use client";

import { useMemo, useState } from "react";
import { Loader2, Play, ShieldAlert, GitBranch, Sparkles, Zap, AlertTriangle, Shuffle, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import {
  useExperiment,
  useExperiments,
  usePredict,
  fmt,
  pct,
  type PredictionResult,
} from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import {
  EmptyState,
  LoadingState,
  SectionCard,
  Stat,
  TransitionIn,
} from "../ui-bits";

function getPresetValues(dataset: string, featureNames: string[], type: "low" | "borderline" | "high"): Record<number, string> {
  const d = dataset.toLowerCase();
  const res: Record<number, string> = {};

  if (d.includes("heart")) {
    const presets = {
      low: ["42", "0", "0", "118", "172", "0", "0", "172", "0", "0.2", "2", "0", "2"],
      borderline: ["54", "1", "1", "132", "238", "0", "1", "142", "0", "1.2", "1", "1", "2"],
      high: ["66", "1", "3", "162", "305", "1", "2", "108", "1", "3.2", "0", "3", "3"],
    };
    const vals = presets[type];
    vals.forEach((v, i) => { res[i] = v; });
    return res;
  }

  if (d.includes("diabetes")) {
    const presets = {
      low: ["1", "92", "68", "18", "70", "24.2", "0.22", "24"],
      borderline: ["3", "126", "76", "28", "120", "31.4", "0.45", "34"],
      high: ["6", "178", "92", "38", "240", "39.8", "0.88", "52"],
    };
    const vals = presets[type];
    vals.forEach((v, i) => { res[i] = v; });
    return res;
  }

  if (d.includes("parkinson")) {
    const scale = type === "low" ? 0.3 : type === "borderline" ? 0.9 : 2.2;
    featureNames.forEach((_, i) => {
      res[i] = (Math.max(0.001, (0.01 + Math.sin(i + 1) * 0.005) * scale)).toFixed(4);
    });
    return res;
  }

  const scale = type === "low" ? 0.6 : type === "borderline" ? 1.0 : 1.8;
  featureNames.forEach((name, i) => {
    let base = 10;
    if (name.includes("area")) base = 550;
    else if (name.includes("perimeter")) base = 85;
    else if (name.includes("texture")) base = 18;
    else if (name.includes("smoothness") || name.includes("concav") || name.includes("compact")) base = 0.09;
    else if (name.includes("radius")) base = 14;
    res[i] = (base * scale).toFixed(2);
  });
  return res;
}

function getRandomValues(featureNames: string[]): Record<number, string> {
  const res: Record<number, string> = {};
  featureNames.forEach((name, i) => {
    let base = 10;
    if (name.includes("area")) base = 500 + Math.random() * 400;
    else if (name.includes("perimeter")) base = 70 + Math.random() * 50;
    else if (name.includes("texture")) base = 12 + Math.random() * 15;
    else if (name.includes("smoothness") || name.includes("concav") || name.includes("compact")) base = 0.05 + Math.random() * 0.15;
    else if (name.includes("radius")) base = 10 + Math.random() * 12;
    else base = 10 + Math.random() * 50;
    res[i] = base.toFixed(2);
  });
  return res;
}

export function PredictionView() {
  const experiments = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const experiment = useExperiment(expId);
  const predict = usePredict();

  const activeId = expId ?? experiments.data?.[0]?.experiment_id ?? null;
  const featureNames = experiment.data?.result?.feature_names ?? [];
  const nFeatures = experiment.data?.result?.n_features ?? 0;
  const datasetName = experiment.data?.result?.dataset ?? "dataset";

  // feature values initialised lazily from names
  const [values, setValues] = useState<Record<number, string>>({});
  const featureValues = useMemo(() => {
    const arr: number[] = [];
    for (let i = 0; i < nFeatures; i++) {
      arr.push(parseFloat(values[i] ?? "0") || 0);
    }
    return arr;
  }, [values, nFeatures]);

  function runWithFeatures(featArr: number[]) {
    if (!activeId) return;
    predict.mutate(
      { experimentId: activeId, features: featArr },
      {
        onSuccess: (res) => {
          if (!res.ok) toast.error("Prediction failed", { description: res.error });
        },
        onError: (e) => toast.error("Prediction failed", { description: String(e) }),
      }
    );
  }

  function run() {
    runWithFeatures(featureValues);
  }

  function applyPreset(type: "low" | "borderline" | "high") {
    const newVals = getPresetValues(datasetName, featureNames, type);
    setValues(newVals);
    const arr: number[] = [];
    for (let i = 0; i < nFeatures; i++) {
      arr.push(parseFloat(newVals[i] ?? "0") || 0);
    }
    runWithFeatures(arr);
    if (type === "borderline") {
      toast.info("Applied Borderline Case preset", {
        description: "Ambiguous screening value lands in [0.35, 0.65] band — triggers Quantum Second Opinion!",
      });
    } else if (type === "low") {
      toast.success("Applied Low Risk preset", {
        description: "Classical screening classifies with high confidence without routing to quantum.",
      });
    } else {
      toast.warning("Applied High Risk preset", {
        description: "Classical screening classifies as positive with high confidence.",
      });
    }
  }

  function applyRandom() {
    const newVals = getRandomValues(featureNames);
    setValues(newVals);
    const arr: number[] = [];
    for (let i = 0; i < nFeatures; i++) {
      arr.push(parseFloat(newVals[i] ?? "0") || 0);
    }
    runWithFeatures(arr);
    toast.info("Randomized features");
  }

  function clearValues() {
    const empty: Record<number, string> = {};
    for (let i = 0; i < nFeatures; i++) empty[i] = "0";
    setValues(empty);
  }

  const pr: PredictionResult | undefined = predict.data?.prediction;
  const abstain = pr?.abstention?.abstain;

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        {/* cascade explainer */}
        <SectionCard
          title="Classical–quantum second-opinion cascade (F12)"
          subtitle="classical screening → confidence gate → quantum second opinion for ambiguous cases → calibration → uncertainty → abstention"
          right={
            <span className="hidden items-center gap-2 text-[11px] text-stone-500 sm:flex">
              <GitBranch className="h-3.5 w-3.5" aria-hidden />
              ambiguity band recorded in the experiment config
            </span>
          }
        >
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-stone-500">
            <span className="rounded-md border border-teal-200 bg-teal-50 px-2 py-1 text-teal-800">classical screening</span>
            <span aria-hidden>→</span>
            <span className="rounded-md border border-stone-200 bg-stone-50 px-2 py-1">confidence gate (0.35–0.65 default)</span>
            <span aria-hidden>→</span>
            <span className="rounded-md border border-violet-200 bg-violet-50 px-2 py-1 text-violet-800">quantum second opinion</span>
            <span aria-hidden>→</span>
            <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-amber-800">calibration</span>
            <span aria-hidden>→</span>
            <span className="rounded-md border border-rose-200 bg-rose-50 px-2 py-1 text-rose-800">uncertainty / abstention</span>
          </div>
        </SectionCard>

        {/* experiment selection */}
        <SectionCard title="Experiment" subtitle="uses the deployment artifacts (pipeline + refit models + conformal calibrator) stored by the run">
          {experiments.isLoading ? (
            <LoadingState label="Loading experiments…" />
          ) : !experiments.data?.length ? (
            <EmptyState message="No experiments available." hint="Run an experiment first (Training view)." />
          ) : (
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-64 space-y-1.5">
                <label className="text-xs font-medium text-stone-600" htmlFor="pred-exp">
                  experiment
                </label>
                <Select value={activeId ?? undefined} onValueChange={setExpId}>
                  <SelectTrigger id="pred-exp" className="qed-focus h-9 text-xs">
                    <SelectValue placeholder="select experiment" />
                  </SelectTrigger>
                  <SelectContent>
                    {experiments.data.map((e) => (
                      <SelectItem key={e.experiment_id} value={e.experiment_id} className="text-xs">
                        {e.experiment_id} — {e.dataset}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {experiment.data?.result && (
                <div className="qed-num text-[11px] text-stone-500">
                  {experiment.data.result.dataset} · {experiment.data.result.n_features} input
                  features · fingerprint {String(experiment.data.result.dataset_fingerprint ?? "").slice(0, 10)}
                </div>
              )}
            </div>
          )}
        </SectionCard>

        {/* feature inputs with 1-click test presets */}
        {nFeatures > 0 && (
          <SectionCard
            title="Input features (original scale)"
            subtitle={`${featureNames.length} features — load realistic 1-click presets or enter values`}
            right={
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-stone-400 mr-1">1-click test:</span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => applyPreset("low")}
                  className="h-7 gap-1 px-2 text-[11px] text-emerald-700 hover:bg-emerald-50 hover:text-emerald-800"
                >
                  <Sparkles className="h-3 w-3" /> Low Risk
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => applyPreset("borderline")}
                  className="h-7 gap-1 px-2 text-[11px] border-violet-300 bg-violet-50/70 text-violet-800 hover:bg-violet-100 hover:text-violet-900 font-medium shadow-xs"
                >
                  <Zap className="h-3 w-3 text-violet-600 fill-violet-600" /> ⚡ Borderline (Quantum Cascade)
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => applyPreset("high")}
                  className="h-7 gap-1 px-2 text-[11px] text-rose-700 hover:bg-rose-50 hover:text-rose-800"
                >
                  <AlertTriangle className="h-3 w-3" /> High Risk
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={applyRandom}
                  className="h-7 gap-1 px-2 text-[11px] text-stone-600 hover:bg-stone-100"
                >
                  <Shuffle className="h-3 w-3" /> Random
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={clearValues}
                  className="h-7 px-2 text-[11px] text-stone-400 hover:text-stone-700"
                  title="Clear values"
                >
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            }
          >
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {featureNames.map((f, i) => (
                <div key={f} className="space-y-1">
                  <label htmlFor={`feat-${i}`} className="qed-num block truncate text-[11px] font-medium text-stone-600" title={f}>
                    {f}
                  </label>
                  <Input
                    id={`feat-${i}`}
                    type="number"
                    step="any"
                    value={values[i] ?? ""}
                    onChange={(e) => setValues((v) => ({ ...v, [i]: e.target.value }))}
                    placeholder="0.0"
                    className="qed-focus qed-num h-8 text-xs"
                  />
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-3">
              <Button onClick={run} disabled={predict.isPending} size="sm" className="gap-2">
                {predict.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                ) : (
                  <Play className="h-3.5 w-3.5" aria-hidden />
                )}
                run cascade prediction
              </Button>
              <span className="text-[11px] text-stone-400">
                Tip: Click <strong>⚡ Borderline</strong> to trigger the Quantum Second-Opinion Cascade.
              </span>
            </div>
          </SectionCard>
        )}


        {/* result */}
        {pr && (
          <SectionCard
            title="Prediction result"
            subtitle="model used · probability · confidence · uncertainty · abstention"
            right={
              <span
                className={
                  abstain
                    ? "rounded-md border border-rose-200 bg-rose-50 px-2.5 py-1 text-xs font-medium text-rose-800"
                    : "rounded-md border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-800"
                }
              >
                {abstain ? "abstention triggered" : "prediction available"}
              </span>
            }
          >
            {pr.route === "quantum" && (
              <div className="mb-5 flex items-start gap-2.5 rounded-md border border-violet-200 bg-violet-50/90 p-3 text-xs text-violet-900">
                <Zap className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 fill-violet-600" aria-hidden />
                <div>
                  <span className="font-semibold">Quantum Second-Opinion Cascade Triggered: </span>
                  Classical screening probability ({fmt(pr.classical_screening_probability)}) fell inside the [0.35, 0.65] ambiguity band. The sample was routed to the {pr.model_used ?? "quantum circuit"} for quantum-enhanced second opinion, resolving positive-class probability to {fmt(pr.probability_used)}.
                </div>
              </div>
            )}

            <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
              <Stat
                label="route"
                value={pr.route === "quantum" ? "quantum 2nd opinion" : "classical direct"}
                domain={pr.route === "quantum" ? "quantum" : "classical"}
                hint={`model: ${pr.model_used ?? "—"}`}
              />
              <Stat
                label="probability (class 1)"
                value={fmt(pr.probability_used)}
                domain="evaluation"
                hint={`screening p: ${fmt(pr.classical_screening_probability)}`}
              />
              <Stat label="confidence" value={pct(pr.confidence)} domain="uncertainty"
                hint={`entropy ${fmt(pr.uncertainty_entropy, 3)} bits`} />
              <Stat
                label="prediction"
                value={abstain ? "abstain" : String(pr.prediction)}
                domain={abstain ? "uncertainty" : "research"}
                hint={abstain ? "configured rule withheld the prediction" : "threshold 0.5"}
              />
            </div>

            {/* probability bar */}
            <div className="mt-5 space-y-1.5">
              <div className="qed-num flex justify-between text-[11px] text-stone-500">
                <span>0</span>
                <span>positive-class probability</span>
                <span>1</span>
              </div>
              <div className="relative h-2.5 overflow-hidden rounded-full bg-stone-200">
                <div
                  className="absolute inset-y-0 bg-rose-200/70"
                  style={{ left: "35%", width: "30%" }}
                  aria-hidden
                />
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-stone-800 transition-[width] duration-500"
                  style={{ width: `${((pr.probability_used ?? 0) * 100).toFixed(1)}%` }}
                />
                <div className="absolute inset-y-0 left-[35%] w-px bg-rose-400" aria-hidden />
                <div className="absolute inset-y-0 left-[65%] w-px bg-rose-400" aria-hidden />
              </div>
              <p className="text-[11px] text-stone-400">
                shaded band = ambiguity gate (default 0.35–0.65; recorded per experiment)
              </p>
            </div>

            {/* abstention detail */}
            {pr.abstention && (
              <div className="mt-4 rounded-md border border-stone-200 bg-stone-50/70 p-3 text-xs text-stone-600">
                <div className="mb-1.5 flex items-center gap-1.5 font-medium text-stone-700">
                  <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
                  abstention rule
                </div>
                <ul className="qed-num space-y-1">
                  {Object.entries(pr.abstention.reasons ?? {}).map(([k, v]) => (
                    <li key={k}>
                      {v ? "✓" : "·"} {k.replace(/_/g, " ")}
                    </li>
                  ))}
                  {pr.abstention_rule && (
                    <li className="text-stone-400">
                      rule: min confidence {fmt(pr.abstention_rule.min_confidence as number, 2)} ·
                      band [{fmt((pr.abstention_rule.ambiguity_band as number[])?.[0], 2)},{" "}
                      {fmt((pr.abstention_rule.ambiguity_band as number[])?.[1], 2)}]
                      {pr.abstention_rule.conformal_alpha !== null &&
                        pr.abstention_rule.conformal_alpha !== undefined &&
                        ` · conformal α ${fmt(pr.abstention_rule.conformal_alpha as number, 2)}`}
                    </li>
                  )}
                </ul>
              </div>
            )}
            <p className="mt-4 text-[11px] leading-relaxed text-stone-400">
              Research output only — not a diagnostic result.
            </p>
          </SectionCard>
        )}
      </TransitionIn>
    </div>
  );
}
