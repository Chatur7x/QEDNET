/**
 * Prediction view — F12 cascade with F11 uncertainty & abstention.
 * Pick an experiment, set feature values, run the second-opinion cascade.
 */
"use client";

import { useMemo, useState } from "react";
import { Loader2, Play, ShieldAlert, GitBranch } from "lucide-react";
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

export function PredictionView() {
  const experiments = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const experiment = useExperiment(expId);
  const predict = usePredict();

  const activeId = expId ?? experiments.data?.[0]?.experiment_id ?? null;
  const featureNames = experiment.data?.result?.feature_names ?? [];
  const nFeatures = experiment.data?.result?.n_features ?? 0;

  // feature values initialised lazily from names
  const [values, setValues] = useState<Record<number, string>>({});
  const featureValues = useMemo(() => {
    const arr: number[] = [];
    for (let i = 0; i < nFeatures; i++) {
      arr.push(parseFloat(values[i] ?? "0") || 0);
    }
    return arr;
  }, [values, nFeatures]);

  function run() {
    if (!activeId) return;
    predict.mutate(
      { experimentId: activeId, features: featureValues },
      {
        onSuccess: (res) => {
          if (!res.ok) toast.error("Prediction failed", { description: res.error });
        },
        onError: (e) => toast.error("Prediction failed", { description: String(e) }),
      }
    );
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

        {/* feature inputs */}
        {nFeatures > 0 && (
          <SectionCard
            title="Input features (original scale)"
            subtitle={`${featureNames.length} features — values in the dataset's native units`}
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
            <div className="mt-4">
              <Button onClick={run} disabled={predict.isPending} size="sm" className="gap-2">
                {predict.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                ) : (
                  <Play className="h-3.5 w-3.5" aria-hidden />
                )}
                run cascade prediction
              </Button>
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
