/**
 * Explainability view (F8) — SHAP global/local, permutation importance,
 * sensitivity analysis, counterfactuals and quantum diagnostics, mapped
 * back to ORIGINAL input features.
 */
"use client";

import { useMemo, useState } from "react";
import { FlaskConical, Loader2 } from "lucide-react";
import {
  useExplain,
  useExperiment,
  useExperiments,
  MODEL_LABELS,
  fmt,
  type ExplainEntry,
} from "@/lib/api-client";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DivergingBarChart, SensitivityScatter } from "../charts";
import {
  EmptyState,
  FamilyBadge,
  LoadingState,
  SectionCard,
  TransitionIn,
} from "../ui-bits";

export function ExplainabilityView() {
  const experiments = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const activeId = expId ?? experiments.data?.[0]?.experiment_id ?? null;
  const experiment = useExperiment(activeId);
  const explain = useExplain(activeId);

  const entries = useMemo(
    () => Object.values(explain.data ?? {}),
    [explain.data]
  );
  const featureNames = experiment.data?.result?.feature_names ?? [];
  const [modelKey, setModelKey] = useState<string | null>(null);
  const activeModel = modelKey ?? entries[0]?.model ?? null;
  const entry: ExplainEntry | undefined = entries.find((e) => e.model === activeModel);

  const [sampleIdx, setSampleIdx] = useState(0);

  if (experiments.isLoading) return <LoadingState />;
  if (!experiments.data?.length) {
    return <EmptyState message="No experiments available." hint="Run an experiment first." />;
  }

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-64 space-y-1.5">
            <label className="text-xs font-medium text-stone-600" htmlFor="exp-exp">
              experiment
            </label>
            <Select value={activeId ?? undefined} onValueChange={setExpId}>
              <SelectTrigger id="exp-exp" className="qed-focus h-9 w-80 text-xs">
                <SelectValue />
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
          {entries.length > 0 && (
            <div className="min-w-56 space-y-1.5">
              <label className="text-xs font-medium text-stone-600" htmlFor="exp-model">
                explained model
              </label>
              <Select value={activeModel ?? undefined} onValueChange={setModelKey}>
                <SelectTrigger id="exp-model" className="qed-focus h-9 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {entries.map((e) => (
                    <SelectItem key={e.model} value={e.model} className="text-xs">
                      {MODEL_LABELS[e.model] ?? e.model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {explain.isLoading ? (
          <LoadingState label="Loading explanations (SHAP computation is real and can take a minute)…" />
        ) : !entry ? (
          <EmptyState
            message="No explainability artifacts for this experiment."
            hint="Explainability runs automatically at the end of each experiment."
          />
        ) : (
          <>
            {/* global importance */}
            <SectionCard
              title="Global feature importance"
              subtitle="SHAP (KernelExplainer over the FULL pipeline) and permutation importance — both map to ORIGINAL input features"
              right={<FamilyBadge family={entry.model.startsWith("q") || ["vqc", "reupload", "hybrid_qnn", "qsvm"].includes(entry.model) ? "quantum" : "classical"} />}
            >
              <Tabs defaultValue="shap" className="w-full">
                <TabsList className="h-8 bg-stone-100 text-xs">
                  <TabsTrigger value="shap" className="text-xs">SHAP (global mean |value|)</TabsTrigger>
                  <TabsTrigger value="perm" className="text-xs">permutation importance</TabsTrigger>
                </TabsList>
                <TabsContent value="shap" className="mt-4">
                  {entry.shap.available && entry.shap.global_mean_abs_shap ? (
                    <DivergingBarChart
                      unit="SHAP"
                      items={zipNames(featureNames, entry.shap.global_mean_abs_shap)}
                    />
                  ) : (
                    <EmptyState message="SHAP unavailable for this model." hint={entry.shap.available === false ? undefined : "computed with real KernelExplainer runs"} />
                  )}
                  {entry.shap.available && (
                    <p className="qed-num mt-2 text-[11px] text-stone-400">
                      method: {entry.shap.method} · background {entry.shap.background_size} training
                      samples · {entry.shap.nsamples} samples/explanation · base value{" "}
                      {fmt(entry.shap.base_value, 4)}
                    </p>
                  )}
                </TabsContent>
                <TabsContent value="perm" className="mt-4">
                  {entry.permutation.available && entry.permutation.importances_mean ? (
                    <SensitivityScatter
                      labels={featureNames}
                      values={entry.permutation.importances_mean}
                      height={260}
                    />
                  ) : (
                    <EmptyState message="Permutation importance unavailable." />
                  )}
                  {entry.permutation.available && (
                    <p className="qed-num mt-2 text-[11px] text-stone-400">
                      {entry.permutation.method} · {entry.permutation.n_repeats} repeats
                    </p>
                  )}
                </TabsContent>
              </Tabs>
            </SectionCard>

            {/* local explanation */}
            {entry.shap.available && entry.shap.local_shap_values?.length ? (
              <SectionCard
                title="Local explanation (single sample)"
                subtitle="signed SHAP values for one explained row — positive pushes the risk score up"
                right={
                  <div className="w-40">
                    <Select value={String(sampleIdx)} onValueChange={(v) => setSampleIdx(Number(v))}>
                      <SelectTrigger className="qed-focus h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {entry.shap.local_shap_values.map((_, i) => (
                          <SelectItem key={i} value={String(i)} className="text-xs">
                            sample {i + 1}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                }
              >
                <DivergingBarChart
                  unit="SHAP"
                  items={zipNames(
                    featureNames,
                    entry.shap.local_shap_values[
                      Math.min(sampleIdx, entry.shap.local_shap_values.length - 1)
                    ] ?? []
                  )}
                />
              </SectionCard>
            ) : null}

            {/* sensitivity */}
            {entry.sensitivity.available && entry.sensitivity.mean_abs_effect ? (
              <SectionCard
                title="Sensitivity analysis"
                subtitle="mean |Δ prediction| when each feature shifts by ± 0.5σ (controlled perturbation)"
              >
                <SensitivityScatter
                  labels={entry.sensitivity.feature_names ?? featureNames}
                  values={entry.sensitivity.mean_abs_effect}
                />
                <p className="qed-num mt-2 text-[11px] text-stone-400">{entry.sensitivity.method}</p>
              </SectionCard>
            ) : null}

            {/* counterfactual */}
            {entry.counterfactual ? (
              <SectionCard
                title="Counterfactual analysis"
                subtitle="minimal feature changes that flip the prediction (greedy search — not claimed optimal)"
              >
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="qed-num space-y-1.5 text-xs text-stone-600">
                    <div>
                      original probability:{" "}
                      <strong className="text-stone-800">{fmt(entry.counterfactual.original_probability as number)}</strong>
                    </div>
                    <div>
                      counterfactual probability:{" "}
                      <strong className="text-stone-800">{fmt(entry.counterfactual.final_probability as number)}</strong>
                    </div>
                    <div>
                      flipped:{" "}
                      <strong className={entry.counterfactual.flipped ? "text-emerald-700" : "text-rose-700"}>
                        {String(entry.counterfactual.flipped)}
                      </strong>{" "}
                      · steps {String(entry.counterfactual.n_steps)} · total change L1{" "}
                      {fmt(entry.counterfactual.total_change_l1 as number, 3)}
                    </div>
                    <p className="text-[11px] text-stone-400">{String(entry.counterfactual.method)}</p>
                  </div>
                  <div className="qed-scroll max-h-56 overflow-y-auto rounded-md border border-stone-100">
                    <table className="w-full text-[11px]">
                      <thead className="sticky top-0 bg-stone-50 text-stone-500">
                        <tr>
                          <th className="px-2 py-1.5 text-left font-medium">feature</th>
                          <th className="qed-num px-2 py-1.5 text-right font-medium">original</th>
                          <th className="qed-num px-2 py-1.5 text-right font-medium">counterfactual</th>
                          <th className="qed-num px-2 py-1.5 text-right font-medium">Δ</th>
                        </tr>
                      </thead>
                      <tbody className="qed-num text-stone-600">
                        {((entry.counterfactual.changed_features as any[]) ?? []).map((c, i) => (
                          <tr key={i} className="border-t border-stone-100">
                            <td className="truncate px-2 py-1.5" title={String(c.feature)}>{String(c.feature)}</td>
                            <td className="px-2 py-1.5 text-right">{fmt(c.original, 3)}</td>
                            <td className="px-2 py-1.5 text-right">{fmt(c.counterfactual, 3)}</td>
                            <td className="px-2 py-1.5 text-right text-stone-500">
                              {c.delta >= 0 ? "+" : ""}
                              {fmt(c.delta, 3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </SectionCard>
            ) : null}

            {/* quantum diagnostics */}
            {entry.quantum_diagnostics && (
              <SectionCard
                title="Quantum diagnostics"
                subtitle="feature sensitivity through the encoder, parameter-shift gradients, circuit structure"
                right={<FamilyBadge family="quantum" />}
              >
                <QuantumDiagnosticsPanel diag={entry.quantum_diagnostics as Record<string, any>} />
              </SectionCard>
            )}
          </>
        )}
      </TransitionIn>
    </div>
  );
}

function QuantumDiagnosticsPanel({ diag }: { diag: Record<string, any> }) {
  if (!diag.available) {
    return <EmptyState message={diag.reason ?? "quantum diagnostics unavailable"} />;
  }
  const grads = diag.parameter_shift_gradient_mean_abs as number[] | undefined;
  const sens = diag.encoder_feature_sensitivity as number[] | undefined;
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div>
        <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-stone-500">
          circuit structure
        </h4>
        <ul className="qed-num space-y-1 text-xs text-stone-600">
          <li>qubits: {String(diag.n_qubits)}</li>
          <li>layers: {String(diag.layers)}</li>
          <li>trainable parameters: {String(diag.trainable_parameters)}</li>
          <li>
            entangling gates: {String((diag.circuit_structure as any)?.entangling_gates)} · depth{" "}
            {String((diag.circuit_structure as any)?.circuit_depth)}
          </li>
          <li className="text-stone-400">ansatz: {String((diag.circuit_structure as any)?.ansatz)}</li>
          {diag.parameter_gradient_norm !== undefined && (
            <li>mean |parameter-shift gradient| (L2): {fmt(diag.parameter_gradient_norm, 6)}</li>
          )}
        </ul>
        <p className="mt-2 text-[11px] text-stone-400">{String(diag.note ?? "")}</p>
        {Array.isArray(diag.kernel) && (
          <p className="qed-num mt-1 text-[11px] text-stone-400">kernel: {String(diag.kernel)}</p>
        )}
      </div>
      <div className="space-y-4">
        {Array.isArray(grads) && grads.length > 0 && (
          <div>
            <h4 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-stone-500">
              parameter-shift |∂f/∂θ| (per parameter)
            </h4>
            <div className="qed-num flex h-12 items-end gap-[2px]">
              {grads.map((g, i) => (
                <div
                  key={i}
                  className="flex-1 bg-violet-400/70"
                  style={{ height: `${Math.max(2, (g / Math.max(...grads)) * 100)}%` }}
                  title={`θ${i}: ${g}`}
                />
              ))}
            </div>
          </div>
        )}
        {Array.isArray(sens) && sens.length > 0 && (
          <div>
            <h4 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-stone-500">
              encoder feature sensitivity (compressed dims)
            </h4>
            <div className="qed-num flex h-12 items-end gap-1">
              {sens.map((g, i) => (
                <div
                  key={i}
                  className="flex-1 bg-emerald-400/70"
                  style={{ height: `${Math.max(2, (g / Math.max(...sens)) * 100)}%` }}
                  title={`z${i}: ${g}`}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function zipNames(names: string[], values: number[]): { label: string; value: number }[] {
  const n = Math.min(names.length, values.length);
  return Array.from({ length: n }, (_, i) => ({
    label: names[i] ?? `x${i}`,
    value: values[i],
  }));
}
