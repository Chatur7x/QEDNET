/**
 * Overview — project status, datasets, experiments, recent benchmarks,
 * research summary, safety status (Section 45).
 */
"use client";

import { Link2, ShieldCheck } from "lucide-react";
import {
  useDatasets,
  useExperiments,
  useHealth,
  fmt,
  pct,
  seconds,
} from "@/lib/api-client";
import type { ViewId } from "../shell";
import {
  DomainDot,
  EmptyState,
  ErrorState,
  FamilyBadge,
  LoadingState,
  ResearchBadge,
  SectionCard,
  Stat,
  TransitionIn,
} from "../ui-bits";

export function OverviewView({ onNavigate }: { onNavigate: (v: ViewId) => void }) {
  const health = useHealth();
  const datasets = useDatasets();
  const experiments = useExperiments();

  const latest = experiments.data?.[0];
  const totalModels = new Set(experiments.data?.flatMap((e) => e.models) ?? []).size;
  const certified = experiments.data?.filter((e) => e.has_certificate).length ?? 0;

  return (
    <div className="space-y-6">
      <section className="overview-masthead relative overflow-hidden rounded-3xl border border-white/70 p-6 shadow-xl shadow-slate-900/5 sm:p-8">
        <div className="overview-masthead-orb" aria-hidden />
        <div className="relative z-[1] flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200/80 bg-white/75 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-900">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-600" /> Whole-body health research
            </div>
            <h2 className="qed-display mt-4 text-3xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-4xl">
              Research the whole person. Report only what the evidence supports.
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
              Women’s health, including menstrual and life-course research, is a first-class track. The atlas distinguishes registered benchmark data from planned studies; this workspace does not collect personal cycle records.
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate("bodyHealth")}
            className="qed-focus inline-flex items-center gap-2 rounded-xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-950/15 transition hover:-translate-y-0.5 hover:bg-slate-800"
          >
            Open body health atlas <Link2 className="h-4 w-4" aria-hidden />
          </button>
        </div>
      </section>

      {/* status strip */}
      <TransitionIn className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-6">
        <Stat label="Backend" value={health.data?.ok ? "online" : "offline"} domain="classical"
          hint={health.data?.backend?.python ? `Python ${health.data.backend.python}` : undefined} />
        <Stat label="Datasets" value={String(datasets.data?.length ?? "—")} domain="classical"
          hint="registered & profiled" />
        <Stat label="Experiments" value={String(experiments.data?.length ?? "—")} domain="evaluation"
          hint={`${certified} with certificates`} />
        <Stat label="Models benchmarked" value={String(totalModels || "—")} domain="evaluation"
          hint="classical + quantum families" />
        <Stat label="Latest run" value={latest ? seconds(latest.runtime_s) : "—"} domain="evaluation"
          hint={latest?.dataset} />
        <Stat label="Quantum engine" value={health.data?.backend?.packages?.pennylane ? "statevector + PL" : "statevector"}
          domain="quantum" hint="exact simulation" />
      </TransitionIn>

      {/* research summary */}
      <SectionCard
        title="Research summary"
        subtitle="measured quantum value — never assumed"
        right={<ResearchBadge />}
      >
        {experiments.isLoading ? (
          <LoadingState label="Loading experiments…" />
        ) : experiments.isError ? (
          <ErrorState message={(experiments.error as Error).message} />
        ) : !experiments.data?.length ? (
          <EmptyState
            message="No experiments recorded yet."
            hint="Launch one from the Training view; results appear here with full provenance."
          />
        ) : (
          <div className="space-y-4">
            <p className="max-w-3xl text-sm leading-relaxed text-stone-600">
              {experiments.data.length} experiment(s) have saved run artifacts. The protocol,
              datasets, models, splits and seeds are specific to each run; inspect its recorded
              configuration before comparing results. Certificates summarize the saved evidence
              and do not establish quantum advantage or clinical performance.
            </p>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {experiments.data.slice(0, 4).map((e) => (
                <button
                  key={e.experiment_id}
                  onClick={() => onNavigate("benchmark")}
                  className="qed-focus group rounded-lg border border-stone-200 bg-white p-4 text-left transition-colors hover:border-stone-400"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="qed-num text-xs font-medium text-stone-500">{e.experiment_id}</span>
                    {e.evidence_classification && (
                      <span className="rounded-full border border-stone-200 bg-stone-50 px-2 py-0.5 text-[10px] text-stone-600">
                        {e.evidence_classification}
                      </span>
                    )}
                  </div>
                  <div className="mt-2 text-sm font-medium text-stone-800">{e.dataset}</div>
                  <div className="qed-num mt-1 text-[11px] text-stone-500">
                    {e.n_samples} samples · {e.n_features} features · {e.models.length} models ·{" "}
                    {seconds(e.runtime_s)}
                  </div>
                  <div className="mt-2 inline-flex items-center gap-1 text-[11px] text-stone-400 group-hover:text-stone-600">
                    <Link2 className="h-3 w-3" aria-hidden /> open benchmark
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </SectionCard>

      {/* datasets + features */}
      <div className="grid gap-6 lg:grid-cols-2">
        <SectionCard
          title="Datasets"
          subtitle="public, de-identified benchmark data (Tier 1 & 2)"
          right={
            <button onClick={() => onNavigate("datasets")} className="qed-focus text-xs text-stone-500 underline-offset-2 hover:underline">
              view all →
            </button>
          }
        >
          {datasets.isLoading ? (
            <LoadingState label="Loading datasets…" />
          ) : (
            <ul className="space-y-2.5">
              {(datasets.data ?? []).slice(0, 5).map((d) => (
                <li key={d.name} className="flex items-center justify-between gap-3 rounded-md border border-stone-100 bg-stone-50/50 px-3 py-2.5">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-sm font-medium text-stone-800">
                      <DomainDot domain="classical" />
                      {d.name}
                    </div>
                    <div className="qed-num text-[11px] text-stone-500">
                      {d.n_samples} × {d.n_features} · target: {d.target} · {d.source}
                    </div>
                  </div>
                  <div className="qed-num text-right text-[11px] text-stone-500">
                    <div>pos {pct(d.profile?.class_balance?.["1"] ?? NaN, 0)}</div>
                    <div>{d.validation_status}</div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>

        <SectionCard title="Research roadmap" subtitle="whole-body scope; availability is grounded in the registry">
          <div className="space-y-4">
            <p className="text-sm leading-6 text-stone-600">
              Current benchmarks are tabular. Cycle tracking, wearable streams, imaging, genomics, EHR, and clinical workflows need separate approved data and validation before implementation.
            </p>
            <button
              type="button"
              onClick={() => onNavigate("bodyHealth")}
              className="qed-focus inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-800 transition hover:border-cyan-300 hover:bg-cyan-50"
            >
              Explore whole-body and women’s-health research <Link2 className="h-3.5 w-3.5" aria-hidden />
            </button>
          </div>
        </SectionCard>
      </div>

      {/* safety status */}
      <SectionCard
        title="Safety status"
        subtitle="explicit research positioning — shown throughout the platform"
        right={<FamilyBadge family="classical" />}
      >
        <div className="grid gap-4 text-sm leading-relaxed text-stone-600 md:grid-cols-3">
          <div className="flex gap-2.5">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden />
            <p>
              <strong className="text-stone-800">Research & educational platform.</strong> Public or
              de-identified benchmark data only; no real identifiable patient information is
              processed during development.
            </p>
          </div>
          <div className="flex gap-2.5">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden />
            <p>
              <strong className="text-stone-800">Not a medical device.</strong> Not clinically
              validated, not intended for diagnosis or real patient-care decisions. The UI,
              certificates and reports state this explicitly.
            </p>
          </div>
          <div className="flex gap-2.5">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden />
            <p>
              <strong className="text-stone-800">Scientific integrity.</strong> No fabricated
              results: every metric traces to a logged experiment run with seeds, folds and a
              configuration hash. Negative results are reported as-is.
            </p>
          </div>
        </div>
        {health.data?.backend?.research_status && (
          <p className="mt-4 rounded-md border border-stone-200 bg-stone-50 px-3 py-2 text-[11px] text-stone-500">
            {health.data.backend.research_status}
          </p>
        )}
      </SectionCard>
    </div>
  );
}
