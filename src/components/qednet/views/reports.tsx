/**
 * Research Reports view — experiment list, markdown research reports,
 * certificates, conformal coverage, cascade stats, reproducibility.
 */
"use client";

import { useState } from "react";
import { FileText, GitBranch, ShieldQuestion } from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  useExperiment,
  useExperiments,
  fmt,
  pct,
  seconds,
} from "@/lib/api-client";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  EmptyState,
  LoadingState,
  SectionCard,
  Stat,
  TransitionIn,
} from "../ui-bits";
import { ProbabilityHistogram as Histogram } from "../charts";

export function ReportsView() {
  const experiments = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const activeId = expId ?? experiments.data?.[0]?.experiment_id ?? null;
  const experiment = useExperiment(activeId);

  const result = experiment.data?.result;
  const report = experiment.data?.report;

  if (experiments.isLoading) return <LoadingState />;
  if (!experiments.data?.length) {
    return <EmptyState message="No experiments available." hint="Run an experiment first." />;
  }

  const cascade = result?.cascade as any;
  const conformal = result?.conformal as any;

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        <div className="min-w-64 space-y-1.5">
          <label className="text-xs font-medium text-stone-600" htmlFor="rep-exp">
            experiment
          </label>
          <Select value={activeId ?? undefined} onValueChange={setExpId}>
            <SelectTrigger id="rep-exp" className="qed-focus h-9 w-80 text-xs">
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

        {experiment.isLoading || !result ? (
          <LoadingState label="Loading report…" />
        ) : (
          <>
            {/* reliability: conformal + cascade */}
            <div className="grid gap-6 lg:grid-cols-2">
              <SectionCard
                title="Conformal prediction & abstention (F11)"
                subtitle="inductive LAC sets on pooled out-of-fold predictions — marginal coverage guarantee"
                right={<ShieldQuestion className="h-4 w-4 text-stone-400" aria-hidden />}
              >
                {conformal ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <Stat label="target coverage" value={pct(1 - conformal.alpha, 0)} domain="uncertainty" />
                      <Stat
                        label="empirical coverage"
                        value={pct(conformal.empirical_coverage)}
                        domain="uncertainty"
                        hint={`q̂ ${fmt(conformal.q_hat, 4)}`}
                      />
                      <Stat label="singleton sets" value={pct(conformal.frac_singleton, 0)} domain="research" />
                      <Stat label="empty sets" value={pct(conformal.frac_empty, 0)} domain="uncertainty" />
                    </div>
                    <p className="text-[11px] text-stone-400">{String(conformal.note ?? "")}</p>
                  </div>
                ) : (
                  <EmptyState message="conformal analysis not stored for this experiment" />
                )}
              </SectionCard>

              <SectionCard
                title="Second-opinion cascade performance (F12)"
                subtitle="classical screening with quantum routing for ambiguous cases"
                right={<GitBranch className="h-4 w-4 text-stone-400" aria-hidden />}
              >
                {cascade ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <Stat
                        label="routed to quantum"
                        value={pct(cascade.routing.coverage_routed_fraction, 1)}
                        domain="quantum"
                        hint={`${cascade.routing.n_routed_to_quantum}/${cascade.routing.n_total}`}
                      />
                      <Stat label="AUROC screening" value={fmt(cascade.performance.auroc_screening_alone)} domain="classical" />
                      <Stat label="AUROC quantum" value={fmt(cascade.performance.auroc_quantum_alone)} domain="quantum" />
                      <Stat label="AUROC cascade" value={fmt(cascade.performance.auroc_cascade)} domain="evaluation" />
                    </div>
                    <p className="qed-num text-[11px] text-stone-400">
                      band [{fmt(cascade.configuration.ambiguity_band?.[0], 2)},{" "}
                      {fmt(cascade.configuration.ambiguity_band?.[1], 2)}] ·{" "}
                      {String(cascade.configuration.note ?? "")}
                    </p>
                  </div>
                ) : (
                  <EmptyState message="cascade evaluation not stored for this experiment" />
                )}
              </SectionCard>
            </div>

            {/* uncertainty distribution */}
            {result.models && Object.keys(result.models).length > 0 && (
              <SectionCard
                title="Prediction uncertainty distribution"
                subtitle="pooled out-of-fold probabilities of the top model — shaded band marks the cascade ambiguity region"
              >
                <Histogram
                  probs={result.models[Object.keys(result.models).sort((a, b) =>
                    (result.models[b].summary.auroc.mean ?? 0) - (result.models[a].summary.auroc.mean ?? 0)
                  )[0]]?.pooled_predictions.p ?? []}
                  height={200}
                />
              </SectionCard>
            )}

            {/* markdown report + reproducibility */}
            <Tabs defaultValue="report" className="w-full">
              <TabsList className="h-8 bg-stone-100 text-xs">
                <TabsTrigger value="report" className="text-xs">
                  <span className="inline-flex items-center gap-1.5">
                    <FileText className="h-3 w-3" aria-hidden /> research report
                  </span>
                </TabsTrigger>
                <TabsTrigger value="repro" className="text-xs">reproducibility</TabsTrigger>
              </TabsList>

              <TabsContent value="report" className="mt-4">
                <SectionCard
                  title={`Research report — ${result.experiment_id ?? activeId}`}
                  subtitle="auto-generated from measured results only"
                >
                  {report ? (
                    <div className="qed-scroll max-h-[640px] overflow-y-auto pr-2 text-sm leading-relaxed text-stone-700 [&_h1]:qed-display [&_h1]:mb-3 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-stone-900 [&_h2]:mt-5 [&_h2]:mb-2 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-stone-900 [&_h3]:mt-3 [&_h3]:text-xs [&_h3]:font-semibold [&_table]:w-full [&_table]:text-xs [&_th]:border [&_th]:border-stone-200 [&_th]:bg-stone-50 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_td]:border [&_td]:border-stone-200 [&_td]:px-2 [&_td]:py-1 [&_code]:rounded [&_code]:bg-stone-100 [&_code]:px-1 [&_code]:text-[11px] [&_pre]:qed-num [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-stone-950 [&_pre]:p-3 [&_pre]:text-[11px] [&_pre]:text-stone-300 [&_blockquote]:border-l-2 [&_blockquote]:border-stone-300 [&_blockquote]:pl-3 [&_blockquote]:text-stone-500 [&_p]:mt-2 [&_li]:mt-1">
                      <ReactMarkdown>{report}</ReactMarkdown>
                    </div>
                  ) : (
                    <EmptyState message="report.md not found for this experiment" />
                  )}
                </SectionCard>
              </TabsContent>

              <TabsContent value="repro" className="mt-4">
                <SectionCard
                  title="Reproducibility record"
                  subtitle="everything needed to reproduce this experiment is stored with the run"
                >
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="qed-num space-y-1.5 text-xs text-stone-600">
                      <div>
                        python:{" "}
                        <strong>{String((result.reproducibility as any)?.python_version ?? "—")}</strong>
                      </div>
                      <div>
                        platform: {String((result.reproducibility as any)?.platform ?? "—")}
                      </div>
                      <div>
                        config hash:{" "}
                        <code className="rounded bg-stone-100 px-1">
                          {String((result.reproducibility as any)?.config_hash ?? "—")}
                        </code>
                      </div>
                      <div>
                        dataset fingerprint:{" "}
                        <code className="rounded bg-stone-100 px-1">{result.dataset_fingerprint ?? "—"}</code>
                      </div>
                      <div>mlflow logged: {String(result.mlflow_logged ?? false)}</div>
                      <div>total runtime: {seconds(result.total_runtime_s ?? result.runtime_s)}</div>
                    </div>
                    <div className="qed-num space-y-1.5 text-xs text-stone-600">
                      <div className="font-medium text-stone-700">package versions</div>
                      {Object.entries((result.reproducibility as any)?.package_versions ?? {}).map(
                        ([k, v]) => (
                          <div key={k} className="flex justify-between border-b border-stone-100 pb-0.5">
                            <span>{k}</span>
                            <span className="text-stone-500">{String(v)}</span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                  <div className="mt-4 rounded-md border border-stone-200 bg-stone-950 p-3">
                    <code className="qed-num block text-[11px] leading-relaxed text-stone-300">
                      python scripts/reproduce.py --config
                      configs/experiments/{result.dataset}.yaml
                    </code>
                  </div>
                </SectionCard>
              </TabsContent>
            </Tabs>
          </>
        )}
      </TransitionIn>
    </div>
  );
}
