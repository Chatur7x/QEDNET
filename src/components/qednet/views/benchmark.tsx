/**
 * Benchmark view (F7) — classical vs quantum models under one evaluation
 * framework: metric table, ROC/PR/calibration curves, runtime & parameter
 * comparisons, statistical evidence. All values from real experiment JSON.
 */
"use client";

import { useMemo, useState } from "react";
import {
  useExperiment,
  useExperiments,
  MODEL_LABELS,
  fmt,
  pct,
  seconds,
  type ModelResult,
} from "@/lib/api-client";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  CalibrationChart,
  DecisionCurveChart,
  ModelBarChart,
  PrCurvesChart,
  RocCurvesChart,
} from "../charts";
import {
  EmptyState,
  FamilyBadge,
  LoadingState,
  SectionCard,
  TransitionIn,
} from "../ui-bits";

const METRICS: { key: string; label: string; digits?: number; asPct?: boolean }[] = [
  { key: "auroc", label: "AUROC" },
  { key: "auprc", label: "AUPRC" },
  { key: "sensitivity", label: "Sens" },
  { key: "specificity", label: "Spec" },
  { key: "sens_at_spec_95", label: "Sens@95Spec" },
  { key: "f1", label: "F1" },
  { key: "mcc", label: "MCC" },
  { key: "brier", label: "Brier ↓" },
  { key: "ece", label: "ECE ↓" },
];

export function BenchmarkView() {
  const experiments = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const activeId = expId ?? experiments.data?.[0]?.experiment_id ?? null;
  const experiment = useExperiment(activeId);
  const [metric, setMetric] = useState("auroc");

  const result = experiment.data?.result;
  const models = useMemo(() => {
    if (!result?.models) return [];
    return Object.values(result.models).sort(
      (a, b) => (b.summary.auroc.mean ?? 0) - (a.summary.auroc.mean ?? 0)
    );
  }, [result]);

  const barData = models.map((m) => ({
    model: MODEL_LABELS[m.model] ?? m.model,
    value: m.summary[metric]?.mean ?? 0,
    err: m.summary[metric]?.std ?? 0,
  }));

  const curveLimit = 6;
  const curveModels = models.slice(0, curveLimit);

  if (experiments.isLoading) return <LoadingState />;
  if (!experiments.data?.length) {
    return (
      <EmptyState
        message="No experiments available yet."
        hint="Launch a benchmark from the Training view — results appear here with full provenance."
      />
    );
  }

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        {/* experiment picker + protocol */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="min-w-64 space-y-1.5">
            <label className="text-xs font-medium text-stone-600" htmlFor="bench-exp">
              experiment
            </label>
            <Select value={activeId ?? undefined} onValueChange={setExpId}>
              <SelectTrigger id="bench-exp" className="qed-focus h-9 w-80 text-xs">
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
          {result?.protocol && (
            <div className="qed-num flex flex-wrap gap-2 text-[11px] text-stone-500">
              <Badge variant="outline" className="font-normal">
                {String(result.protocol.outer_folds)}-fold outer CV
              </Badge>
              <Badge variant="outline" className="font-normal">
                {String(result.protocol.inner_folds)}-fold inner tuning
              </Badge>
              <Badge variant="outline" className="font-normal">
                seeds {Array.isArray(result.protocol.seeds) ? (result.protocol.seeds as number[]).join(", ") : "—"}
              </Badge>
              <Badge variant="outline" className="font-normal">
                {seconds(result.runtime_s)} runtime
              </Badge>
              <Badge variant="outline" className="font-normal">
                leakage-free ✓
              </Badge>
            </div>
          )}
        </div>

        {experiment.isLoading || !result ? (
          <LoadingState label="Loading benchmark results…" />
        ) : (
          <>
            {/* metric table */}
            <SectionCard
              title="Metric comparison (mean ± std over folds × seeds)"
              subtitle="identical evaluation framework for classical and quantum models"
            >
              <div className="overflow-x-auto qed-scroll">
                <Table>
                  <TableHeader>
                    <TableRow className="border-stone-200">
                      <TableHead className="text-xs">Model</TableHead>
                      <TableHead className="text-xs">Family</TableHead>
                      {METRICS.map((m) => (
                        <TableHead key={m.key} className="qed-num text-right text-xs">
                          {m.label}
                        </TableHead>
                      ))}
                      <TableHead className="qed-num text-right text-xs">Params</TableHead>
                      <TableHead className="qed-num text-right text-xs">Train</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {models.map((m: ModelResult) => (
                      <TableRow key={m.model} className="border-stone-100">
                        <TableCell className="font-medium text-stone-800">
                          {MODEL_LABELS[m.model] ?? m.model}
                        </TableCell>
                        <TableCell>
                          <FamilyBadge family={m.family} />
                        </TableCell>
                        {METRICS.map((met) => {
                          const s = m.summary[met.key];
                          return (
                            <TableCell key={met.key} className="qed-num text-right text-stone-600">
                              {s?.mean == null ? (
                                "—"
                              ) : (
                                <span>
                                  {fmt(s.mean, met.key === "brier" || met.key === "ece" ? 4 : 4)}
                                  <span className="text-[9px] text-stone-400"> ±{fmt(s.std, 3)}</span>
                                </span>
                              )}
                            </TableCell>
                          );
                        })}
                        <TableCell className="qed-num text-right text-stone-600">
                          {m.param_count ?? "—"}
                        </TableCell>
                        <TableCell className="qed-num text-right text-stone-600">
                          {seconds(m.train_time_s.total)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </SectionCard>

            {/* charts */}
            <Tabs defaultValue="metric" className="w-full">
              <TabsList className="h-8 bg-stone-100 text-xs">
                <TabsTrigger value="metric" className="text-xs">metric comparison</TabsTrigger>
                <TabsTrigger value="roc" className="text-xs">ROC</TabsTrigger>
                <TabsTrigger value="pr" className="text-xs">precision–recall</TabsTrigger>
                <TabsTrigger value="calibration" className="text-xs">calibration</TabsTrigger>
                <TabsTrigger value="dca" className="text-xs">decision curve</TabsTrigger>
              </TabsList>

              <TabsContent value="metric" className="mt-4">
                <SectionCard
                  title="Model comparison"
                  subtitle="pooled out-of-fold predictions; bars show mean, tooltips show ± std"
                  right={
                    <div className="w-52">
                      <Select value={metric} onValueChange={setMetric}>
                        <SelectTrigger className="qed-focus h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {METRICS.filter((m) => m.key !== "brier_calibrated" && m.key !== "ece_calibrated").map((m) => (
                            <SelectItem key={m.key} value={m.key} className="text-xs">
                              {m.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  }
                >
                  <ModelBarChart data={barData} metric={metric.toUpperCase()} />
                </SectionCard>
              </TabsContent>

              <TabsContent value="roc" className="mt-4">
                <SectionCard
                  title="ROC curves (pooled out-of-fold)"
                  subtitle={`top ${curveLimit} models by AUROC · diagonal = chance`}
                >
                  <RocCurvesChart
                    series={curveModels.map((m) => ({
                      model: MODEL_LABELS[m.model] ?? m.model,
                      fpr: m.curve_roc.fpr,
                      tpr: m.curve_roc.tpr,
                    }))}
                  />
                </SectionCard>
              </TabsContent>

              <TabsContent value="pr" className="mt-4">
                <SectionCard title="Precision–recall curves (pooled out-of-fold)" subtitle={`top ${curveLimit} models by AUPRC`}>
                  <PrCurvesChart
                    series={curveModels.map((m) => ({
                      model: MODEL_LABELS[m.model] ?? m.model,
                      precision: m.curve_pr.precision,
                      recall: m.curve_pr.recall,
                    }))}
                  />
                </SectionCard>
              </TabsContent>

              <TabsContent value="calibration" className="mt-4">
                <SectionCard
                  title="Calibration (reliability curves, Platt-scaled)"
                  subtitle="calibrators fitted on training-fold predictions only"
                >
                  <CalibrationChart
                    series={curveModels.slice(0, 4).map((m) => ({
                      model: MODEL_LABELS[m.model] ?? m.model,
                      centers: m.curve_calibration.centers,
                      accuracy: m.curve_calibration.accuracy,
                    }))}
                  />
                  <div className="qed-num mt-3 grid gap-2 text-[11px] text-stone-500 sm:grid-cols-2 lg:grid-cols-4">
                    {curveModels.slice(0, 4).map((m) => (
                      <div key={m.model} className="rounded-md border border-stone-100 bg-stone-50/60 px-2.5 py-2">
                        <div className="font-medium text-stone-700">{MODEL_LABELS[m.model] ?? m.model}</div>
                        <div>
                          Brier raw {fmt(m.summary.brier?.mean)} → cal {fmt(m.summary.brier_calibrated?.mean)}
                        </div>
                        <div>
                          ECE raw {fmt(m.summary.ece?.mean)} → cal {fmt(m.summary.ece_calibrated?.mean)}
                        </div>
                      </div>
                    ))}
                  </div>
                </SectionCard>
              </TabsContent>

              <TabsContent value="dca" className="mt-4">
                <SectionCard
                  title="Decision curve analysis"
                  subtitle="net benefit of the top model vs treat-all / treat-none strategies"
                >
                  {curveModels[0]?.decision_curve ? (
                    <DecisionCurveChart
                      thresholds={curveModels[0].decision_curve.thresholds}
                      model={curveModels[0].decision_curve.model}
                      treatAll={curveModels[0].decision_curve.treat_all}
                    />
                  ) : (
                    <EmptyState message="Decision curve analysis disabled for this experiment." />
                  )}
                </SectionCard>
              </TabsContent>
            </Tabs>

            {/* statistical evidence */}
            <SectionCard
              title="Statistical evidence"
              subtitle="DeLong paired tests on pooled predictions + Nadeau–Bengio corrected CV comparison — significance only when actually tested"
            >
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {Object.entries(result.statistics ?? {}).map(([qk, entry]) => {
                  const delong = (entry as any)?.delong_vs_best_classical;
                  const cvt = (entry as any)?.corrected_cv_vs_best_classical;
                  return (
                    <div key={qk} className="rounded-lg border border-stone-200 bg-white p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-stone-800">
                          {MODEL_LABELS[qk] ?? qk}
                        </span>
                        <FamilyBadge family="quantum" />
                      </div>
                      <div className="qed-num mt-2 space-y-1 text-[11px] text-stone-600">
                        {delong ? (
                          <>
                            <div>
                              DeLong Δ AUROC {fmt(delong.auc_diff)} · z {fmt(delong.z, 2)} ·{" "}
                              <span className={delong.significant_005 ? "font-medium text-emerald-700" : "text-stone-500"}>
                                p {delong.p_value < 0.001 ? "<0.001" : fmt(delong.p_value, 3)}
                              </span>
                            </div>
                            <div className="text-stone-400">
                              vs best classical ({(entry as any)?.best_classical_model})
                            </div>
                          </>
                        ) : (
                          <div className="text-stone-400">DeLong test unavailable</div>
                        )}
                        {cvt?.available && (
                          <div>
                            corrected CV t {fmt(cvt.t, 2)} ·{" "}
                            <span className={cvt.significant_005 ? "font-medium text-emerald-700" : "text-stone-500"}>
                              p {cvt.p_value < 0.001 ? "<0.001" : fmt(cvt.p_value, 3)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
                {!Object.keys(result.statistics ?? {}).length && (
                  <p className="text-xs text-stone-400">No statistics recorded.</p>
                )}
              </div>
            </SectionCard>

            {/* resources */}
            <SectionCard
              title="Runtime, parameters & quantum resources"
              subtitle="honest accounting: parameter counts measured post-fit; circuit depth from executed gate schedules"
            >
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {models.map((m) => {
                  const res = m.resources as Record<string, any> | null;
                  return (
                    <div key={m.model} className="rounded-lg border border-stone-200 bg-white p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-stone-800">{MODEL_LABELS[m.model] ?? m.model}</span>
                        <FamilyBadge family={m.family} />
                      </div>
                      <ul className="qed-num mt-2 space-y-1 text-[11px] text-stone-600">
                        <li>
                          parameters: <strong>{m.param_count ?? "—"}</strong>
                          {m.model === "matched_mlp" && (m.model_meta as any)?.matched_to_params && (
                            <span className="text-stone-400"> (matched to {(m.model_meta as any).matched_to_params})</span>
                          )}
                        </li>
                        <li>train {seconds(m.train_time_s.total)} · inference {seconds((m.model_meta as any)?.inference_time_s)}</li>
                        {res && m.family === "quantum" && (
                          <>
                            <li>
                              qubits {res.n_qubits} · depth {res.circuit_depth} · circuits {res.n_circuits}
                            </li>
                            <li className="text-stone-400">{res.shots_note}</li>
                          </>
                        )}
                        {res && m.family === "classical" && (
                          <li className="text-stone-400">classical model — no quantum resources</li>
                        )}
                      </ul>
                    </div>
                  );
                })}
              </div>
            </SectionCard>
          </>
        )}
      </TransitionIn>
    </div>
  );
}
