/**
 * Quantum Advantage view (F9/F10) — certificate verdict, entanglement
 * ablation, dequantization, parameter efficiency, statistical evidence,
 * resource cost, and the bottleneck honesty meter.
 */
"use client";

import { useState } from "react";
import { Brain, FileCheck2, GitCompare, Scale } from "lucide-react";
import {
  useCertificate,
  useExperiment,
  useExperiments,
  MODEL_LABELS,
  fmt,
  seconds,
} from "@/lib/api-client";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  EmptyState,
  LoadingState,
  ResearchBadge,
  SectionCard,
  Stat,
  TransitionIn,
} from "../ui-bits";

export function AdvantageView() {
  const experiments = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const activeId = expId ?? experiments.data?.[0]?.experiment_id ?? null;
  const experiment = useExperiment(activeId);
  const cert = useCertificate(activeId);

  const result = experiment.data?.result;
  const c = cert.data?.certificate;

  if (experiments.isLoading) return <LoadingState />;
  if (!experiments.data?.length) {
    return <EmptyState message="No experiments available." hint="Run an experiment first." />;
  }

  const verdictClass =
    c?.final_evidence_classification === "No Demonstrated Advantage"
      ? "border-stone-300 bg-stone-50 text-stone-700"
      : c?.final_evidence_classification === "Parameter-Efficiency Advantage"
      ? "border-emerald-300 bg-emerald-50 text-emerald-800"
      : "border-amber-300 bg-amber-50 text-amber-800";

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="min-w-64 space-y-1.5">
            <label className="text-xs font-medium text-stone-600" htmlFor="adv-exp">
              experiment
            </label>
            <Select value={activeId ?? undefined} onValueChange={setExpId}>
              <SelectTrigger id="adv-exp" className="qed-focus h-9 w-80 text-xs">
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
          <ResearchBadge />
        </div>

        {cert.isLoading ? (
          <LoadingState label="Loading certificate…" />
        ) : !c ? (
          <EmptyState
            message="No Quantum Advantage Certificate for this experiment."
            hint="Certificates are generated automatically when certificate.enabled is true in the experiment config."
          />
        ) : (
          <>
            {/* verdict */}
            <SectionCard
              title="Quantum Advantage Certificate (F9)"
              subtitle={`generated from actual experiment data · experiment ${c.experiment_id} · config hash ${c.configuration_hash}`}
              right={<FileCheck2 className="h-4 w-4 text-stone-400" aria-hidden />}
            >
              <div className={`rounded-lg border p-4 ${verdictClass}`}>
                <div className="flex items-center gap-2.5">
                  <Brain className="h-5 w-5 shrink-0" aria-hidden />
                  <span className="qed-display text-lg font-semibold">
                    {c.final_evidence_classification}
                  </span>
                </div>
                <p className="mt-1.5 text-sm leading-relaxed opacity-90">{c.final_evidence_rationale}</p>
              </div>

              <div className="mt-5 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                <Stat
                  label={`best quantum (${MODEL_LABELS[c.metrics?.quantum_best?.model] ?? c.model})`}
                  value={fmt(c.metrics?.quantum_best?.auroc)}
                  domain="quantum"
                  hint={`95% CI [${fmt(c.metrics?.quantum_best?.ci95?.[0], 3)}, ${fmt(c.metrics?.quantum_best?.ci95?.[1], 3)}]`}
                />
                <Stat
                  label={`best classical (${MODEL_LABELS[c.metrics?.classical_best?.model] ?? "—"})`}
                  value={fmt(c.metrics?.classical_best?.auroc)}
                  domain="classical"
                  hint={`95% CI [${fmt(c.metrics?.classical_best?.ci95?.[0], 3)}, ${fmt(c.metrics?.classical_best?.ci95?.[1], 3)}]`}
                />
                <Stat
                  label="quantum / matched-MLP params"
                  value={fmt(c.parameter_count?.ratio_quantum_over_mlp, 3)}
                  domain="research"
                  hint={`${c.parameter_count?.quantum} quantum vs ${c.parameter_count?.matched_mlp} MLP parameters`}
                />
                <Stat
                  label="resources"
                  value={`${c.qubit_count} qubits`}
                  domain="quantum"
                  hint={`depth ${c.circuit_depth} · ${seconds(c.training_runtime?.[c.model])} train`}
                />
              </div>

              <ul className="mt-4 space-y-1.5 text-[11px] leading-relaxed text-stone-500">
                {(c.methodology_notes ?? []).map((n: string, i: number) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-stone-300">·</span>
                    {n}
                  </li>
                ))}
              </ul>
            </SectionCard>

            {/* ablation + dequantization */}
            <div className="grid gap-6 lg:grid-cols-2">
              <SectionCard
                title="Entanglement ablation (parameter-matched)"
                subtitle="entangled ansatz vs non-entangled control with identical trainable parameter count"
                right={<GitCompare className="h-4 w-4 text-stone-400" aria-hidden />}
              >
                {c.ablation_result?.available ? (
                  <div className="space-y-3 text-xs">
                    <div className="qed-num grid grid-cols-2 gap-3">
                      <div className="rounded-md border border-violet-200 bg-violet-50/50 p-3">
                        <div className="text-[11px] font-medium text-violet-800">entangled</div>
                        <div className="mt-1 text-stone-700">
                          AUROC {fmt(c.ablation_result.entangled.auroc)}
                        </div>
                        <div className="text-stone-500">
                          {c.ablation_result.entangled.n_parameters} params ·{" "}
                          {c.ablation_result.entangled.entangling_gates} CNOTs
                        </div>
                      </div>
                      <div className="rounded-md border border-stone-200 bg-stone-50/60 p-3">
                        <div className="text-[11px] font-medium text-stone-700">non-entangled (matched)</div>
                        <div className="mt-1 text-stone-700">
                          AUROC {fmt(c.ablation_result.non_entangled_parameter_matched.auroc)}
                        </div>
                        <div className="text-stone-500">
                          {c.ablation_result.non_entangled_parameter_matched.n_parameters} params · 0
                          CNOTs
                        </div>
                      </div>
                    </div>
                    <div className="qed-num text-stone-600">
                      Δ AUROC (entangled − non-entangled):{" "}
                      <strong className="text-stone-800">
                        {fmt(c.ablation_result.auroc_delta_entangled_minus_nonentangled)}
                      </strong>{" "}
                      · capacity preserved: {String(c.ablation_result.capacity_preserved)}
                    </div>
                    <Badge variant="outline" className="text-[11px] font-normal">
                      entanglement contribution: {String(c.ablation_result.entanglement_contribution)}
                    </Badge>
                    <p className="text-[11px] text-stone-400">{String(c.ablation_result.interpretation)}</p>
                  </div>
                ) : (
                  <EmptyState message={String(c.ablation_result?.reason ?? "ablation unavailable")} />
                )}
              </SectionCard>

              <SectionCard
                title="Classical dequantization"
                subtitle="can classical approximations reproduce the quantum kernel's performance?"
                right={<Scale className="h-4 w-4 text-stone-400" aria-hidden />}
              >
                {c.dequantization_result ? (
                  <div className="space-y-3 text-xs">
                    <div className="qed-num rounded-md border border-stone-200 bg-stone-50/60 p-3 text-stone-700">
                      quantum kernel reference AUROC:{" "}
                      <strong>{fmt(c.dequantization_result.quantum_kernel_auroc)}</strong> (ZZ/angle
                      fidelity kernel, exact statevector)
                    </div>
                    {c.dequantization_result.rff && (
                      <div className="qed-num rounded-md border border-teal-200 bg-teal-50/40 p-3">
                        <div className="font-medium text-teal-800">
                          Random Fourier Features: AUROC {fmt(c.dequantization_result.rff.auroc)}
                        </div>
                        <div className="text-[11px] text-stone-500">
                          {String(c.dequantization_result.rff.method)} ·{" "}
                          {String(c.dequantization_result.rff.n_components)} components
                        </div>
                      </div>
                    )}
                    {c.dequantization_result.nystrom && (
                      <div className="qed-num rounded-md border border-teal-200 bg-teal-50/40 p-3">
                        <div className="font-medium text-teal-800">
                          Nyström: AUROC {fmt(c.dequantization_result.nystrom.auroc)}
                        </div>
                        <div className="text-[11px] text-stone-500">
                          {String(c.dequantization_result.nystrom.method)} · kernel reconstruction
                          error {fmt(c.dequantization_result.nystrom.kernel_relative_reconstruction_error)}
                        </div>
                      </div>
                    )}
                    <p className="text-[11px] leading-relaxed text-stone-400">
                      If the classical surrogates match the quantum kernel model, no kernel-space
                      quantum advantage is demonstrated on this dataset — and the certificate says
                      exactly that.
                    </p>
                  </div>
                ) : (
                  <EmptyState message="dequantization unavailable (requires QSVM in the suite)" />
                )}
              </SectionCard>
            </div>

            {/* honesty meter */}
            {c.bottleneck_honesty_meter && (
              <SectionCard
                title="Bottleneck honesty meter (F10)"
                subtitle="how much performance comes from classical compression vs the quantum layer"
              >
                <div className="grid gap-5 sm:grid-cols-3">
                  <Stat
                    label="classical on compressed features"
                    value={fmt(c.bottleneck_honesty_meter.compressed_space_baseline?.auroc)}
                    domain="classical"
                    hint={`${c.bottleneck_honesty_meter.compressed_space_baseline?.n_dims} dims (logistic regression)`}
                  />
                  <Stat
                    label="quantum on same features"
                    value={fmt(c.bottleneck_honesty_meter.quantum_score?.auroc)}
                    domain="quantum"
                    hint={String(c.bottleneck_honesty_meter.quantum_score?.model)}
                  />
                  <Stat
                    label="gap (quantum − baseline)"
                    value={fmt(c.bottleneck_honesty_meter.performance_gap_quantum_minus_baseline)}
                    domain="research"
                    hint={String(c.bottleneck_honesty_meter.reading)}
                  />
                </div>
              </SectionCard>
            )}

            {/* statistical evidence detail */}
            <SectionCard
              title="Statistical evidence (F9 input)"
              subtitle="DeLong paired z-test + Nadeau–Bengio corrected resampled t-test + bootstrap CIs"
            >
              <div className="qed-scroll overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-stone-500">
                    <tr className="border-b border-stone-200">
                      <th className="px-2 py-2 text-left font-medium">test</th>
                      <th className="px-2 py-2 text-left font-medium">statistic</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">p-value</th>
                      <th className="px-2 py-2 text-left font-medium">significant (α=0.05)</th>
                    </tr>
                  </thead>
                  <tbody className="qed-num text-stone-600">
                    {c.statistical_result?.delong_test?.delong_available && (
                      <tr className="border-b border-stone-100">
                        <td className="px-2 py-2">DeLong (paired AUROC)</td>
                        <td className="px-2 py-2">z {fmt(c.statistical_result.delong_test.z, 3)}</td>
                        <td className="px-2 py-2 text-right">
                          {fmt(c.statistical_result.delong_test.p_value, 4)}
                        </td>
                        <td className="px-2 py-2">
                          {String(c.statistical_result.delong_test.significant_005)}
                        </td>
                      </tr>
                    )}
                    {c.statistical_result?.corrected_cv_test?.available && (
                      <tr className="border-b border-stone-100">
                        <td className="px-2 py-2">corrected CV (Nadeau–Bengio)</td>
                        <td className="px-2 py-2">t {fmt(c.statistical_result.corrected_cv_test.t, 3)}</td>
                        <td className="px-2 py-2 text-right">
                          {fmt(c.statistical_result.corrected_cv_test.p_value, 4)}
                        </td>
                        <td className="px-2 py-2">
                          {String(c.statistical_result.corrected_cv_test.significant_005)}
                        </td>
                      </tr>
                    )}
                    {Object.entries(c.statistical_result?.bootstrap ?? {}).map(([k, b]: any) =>
                      b ? (
                        <tr key={k} className="border-b border-stone-100">
                          <td className="px-2 py-2">bootstrap AUROC ({MODEL_LABELS[k] ?? k})</td>
                          <td className="px-2 py-2">
                            CI [{fmt(b.ci_low, 4)}, {fmt(b.ci_high, 4)}]
                          </td>
                          <td className="px-2 py-2 text-right">
                            {String(b.n_resamples)} resamples
                          </td>
                          <td className="px-2 py-2 text-stone-400">—</td>
                        </tr>
                      ) : null
                    )}
                  </tbody>
                </table>
              </div>
            </SectionCard>

            {/* resource cost */}
            <SectionCard
              title="Computational cost & resources"
              subtitle="measured runtimes and quantum resource accounting"
            >
              <div className="qed-scroll overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-stone-500">
                    <tr className="border-b border-stone-200">
                      <th className="px-2 py-2 text-left font-medium">model</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">train (total)</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">inference</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">params</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">qubits</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">depth</th>
                      <th className="qed-num px-2 py-2 text-right font-medium">circuits</th>
                    </tr>
                  </thead>
                  <tbody className="qed-num text-stone-600">
                    {Object.entries(result?.models ?? {}).map(([k, m]) => {
                      const res = m.resources as Record<string, any> | null;
                      return (
                        <tr key={k} className="border-b border-stone-100">
                          <td className="px-2 py-2 font-medium text-stone-700">{MODEL_LABELS[k] ?? k}</td>
                          <td className="px-2 py-2 text-right">{seconds(c.training_runtime?.[k])}</td>
                          <td className="px-2 py-2 text-right">{seconds(c.inference_runtime?.[k])}</td>
                          <td className="px-2 py-2 text-right">{String(c.parameter_count && k === "matched_mlp" ? c.parameter_count.matched_mlp : m.param_count ?? "—")}</td>
                          <td className="px-2 py-2 text-right">{res?.n_qubits ?? "—"}</td>
                          <td className="px-2 py-2 text-right">{res?.circuit_depth ?? "—"}</td>
                          <td className="px-2 py-2 text-right">{res?.n_circuits ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="qed-num mt-3 text-[11px] text-stone-400">
                {String((c.quantum_resource_data ?? {}).shots_note ?? "")}
              </p>
            </SectionCard>
          </>
        )}
      </TransitionIn>
    </div>
  );
}
