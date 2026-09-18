/**
 * QED-Net 2.0 dashboard shell — sidebar navigation + view switching.
 * Single-route SPA (sandbox constraint: only "/" is user-visible).
 * Motion: critically-damped springs, reduced-motion aware, GPU-friendly
 * (transform/opacity only).
 */
"use client";

import { useMemo, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  Activity,
  Beaker,
  Brain,
  FileText,
  FlaskConical,
  LayoutDashboard,
  Microscope,
  Table2,
  Scale,
  ShieldAlert,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SPRING, ResearchBadge } from "./ui-bits";
import { useHealth } from "@/lib/api-client";
import { OverviewView } from "./views/overview";
import { DatasetsView } from "./views/datasets";
import { TrainingView } from "./views/training";
import { PredictionView } from "./views/prediction";
import { ExplainabilityView } from "./views/explainability";
import { BenchmarkView } from "./views/benchmark";
import { AdvantageView } from "./views/advantage";
import { ReportsView } from "./views/reports";

export type ViewId =
  | "overview"
  | "datasets"
  | "training"
  | "prediction"
  | "explainability"
  | "benchmark"
  | "advantage"
  | "reports";

const NAV: { id: ViewId; label: string; icon: React.ComponentType<{ className?: string }>; hint: string }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, hint: "project status & research summary" },
  { id: "datasets", label: "Datasets", icon: Table2, hint: "F1/F2 — ingestion & validation" },
  { id: "training", label: "Training", icon: FlaskConical, hint: "F3–F7 — configure & launch experiments" },
  { id: "prediction", label: "Prediction", icon: Activity, hint: "F11/F12 — cascade, uncertainty, abstention" },
  { id: "explainability", label: "Explainability", icon: Microscope, hint: "F8 — SHAP, sensitivity, counterfactuals" },
  { id: "benchmark", label: "Benchmark", icon: Scale, hint: "F7 — quantum vs classical results" },
  { id: "advantage", label: "Quantum Advantage", icon: Brain, hint: "F9/F10 — certificate & honesty meter" },
  { id: "reports", label: "Research Reports", icon: FileText, hint: "experiment reports & certificates" },
];

const VIEW_TITLES: Record<ViewId, { title: string; sub: string }> = {
  overview: { title: "Overview", sub: "project status, datasets, experiments, research summary, safety" },
  datasets: { title: "Datasets", sub: "registered biomedical datasets, validation & leakage guard status" },
  training: { title: "Training", sub: "configure and launch benchmark experiments (background jobs)" },
  prediction: { title: "Prediction", sub: "classical–quantum cascade with calibration, uncertainty & abstention" },
  explainability: { title: "Explainability", sub: "SHAP, feature importance, sensitivity, counterfactuals, quantum diagnostics" },
  benchmark: { title: "Benchmark", sub: "classical vs quantum models under one evaluation framework" },
  advantage: { title: "Quantum Advantage", sub: "certificate, ablation, dequantization, statistical evidence" },
  reports: { title: "Research Reports", sub: "per-experiment reports, certificates, reproducibility" },
};

export function QEDNetDashboard() {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 5_000, retry: 1, refetchOnWindowFocus: false },
        },
      })
  );
  return (
    <QueryClientProvider client={client}>
      <DashboardInner />
    </QueryClientProvider>
  );
}

function DashboardInner() {
  const [view, setView] = useState<ViewId>("overview");
  const reduce = useReducedMotion();
  const health = useHealth();

  const backendOk = health.data?.ok ?? false;
  const pyVer = health.data?.backend?.python;
  const pennylane = health.data?.backend?.packages?.pennylane;

  const views = useMemo(
    () => ({
      overview: <OverviewView onNavigate={(v: ViewId) => setView(v)} />,
      datasets: <DatasetsView />,
      training: <TrainingView />,
      prediction: <PredictionView />,
      explainability: <ExplainabilityView />,
      benchmark: <BenchmarkView />,
      advantage: <AdvantageView />,
      reports: <ReportsView />,
    }),
    []
  );

  return (
    <div className="qed-app-bg flex min-h-screen w-full text-stone-900">
      {/* ------------------------------------------------ sidebar */}
      <nav
        aria-label="Dashboard sections"
        className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-stone-200 bg-white/70 backdrop-blur-sm md:flex"
      >
        <div className="flex items-center gap-2.5 border-b border-stone-200 px-5 py-5">
          <Beaker className="h-5 w-5 text-violet-700" aria-hidden />
          <div>
            <div className="qed-display text-[15px] font-semibold leading-tight text-stone-900">
              QED-Net <span className="text-stone-400">2.0</span>
            </div>
            <div className="text-[10px] uppercase tracking-wider text-stone-400">
              hybrid quantum–classical ML
            </div>
          </div>
        </div>

        <ul className="flex-1 space-y-0.5 overflow-y-auto px-3 py-4 qed-scroll">
          {NAV.map((item) => {
            const active = view === item.id;
            const Icon = item.icon;
            return (
              <li key={item.id}>
                <button
                  onClick={() => setView(item.id)}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "qed-focus group flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[13px] font-medium transition-colors",
                    active
                      ? "bg-stone-900 text-white"
                      : "text-stone-600 hover:bg-stone-100 hover:text-stone-900"
                  )}
                >
                  <Icon className={cn("h-4 w-4 shrink-0", active ? "text-white" : "text-stone-400 group-hover:text-stone-600")} aria-hidden />
                  <span className="truncate">{item.label}</span>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="space-y-3 border-t border-stone-200 px-4 py-4">
          <div className="flex items-center gap-2 text-[11px] text-stone-500">
            <span
              className={cn("inline-block h-1.5 w-1.5 rounded-full", backendOk ? "bg-emerald-500" : "bg-rose-500")}
              aria-hidden
            />
            <span className="truncate">
              {backendOk
                ? `backend online · Python ${pyVer ?? "—"}`
                : "backend offline"}
            </span>
          </div>
          {pennylane && (
            <div className="flex items-center gap-2 text-[11px] text-stone-500">
              <Cpu className="h-3 w-3 text-violet-500" aria-hidden />
              PennyLane {String(pennylane)}
            </div>
          )}
          <ResearchBadge compact />
          <p className="text-[10px] leading-relaxed text-stone-400">
            Not a medical device. Not clinically validated. Not for diagnosis.
          </p>
        </div>
      </nav>

      {/* ------------------------------------------------ main */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 border-b border-stone-200 bg-[#fafaf9]/85 backdrop-blur-md">
          <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 md:px-8">
            <div className="min-w-0">
              <h1 className="qed-display text-lg font-semibold text-stone-900">
                {VIEW_TITLES[view].title}
              </h1>
              <p className="truncate text-xs text-stone-500">{VIEW_TITLES[view].sub}</p>
            </div>
            <div className="hidden items-center gap-2 sm:flex">
              <ResearchBadge />
            </div>
          </div>
          {/* mobile nav */}
          <div className="flex gap-1 overflow-x-auto border-t border-stone-100 px-3 py-2 md:hidden qed-scroll">
            {NAV.map((item) => (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={cn(
                  "qed-focus whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs font-medium",
                  view === item.id ? "bg-stone-900 text-white" : "text-stone-500 hover:bg-stone-100"
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </header>

        <main id="main" className="min-w-0 flex-1 px-5 py-6 md:px-8" aria-live="polite">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={view}
              initial={reduce ? { opacity: 0 } : { opacity: 0, y: 8 }}
              animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
              exit={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }}
              transition={SPRING.smooth}
              className="min-w-0"
            >
              {views[view]}
            </motion.div>
          </AnimatePresence>
        </main>

        <footer className="mt-auto border-t border-stone-200 bg-white/60 px-5 py-4 text-[11px] leading-relaxed text-stone-400 md:px-8">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              QED-Net 2.0 — Hybrid Quantum–Classical ML research platform. All
              metrics shown are real experimental results with logged provenance.
            </span>
            <span className="inline-flex items-center gap-1.5">
              <ShieldAlert className="h-3 w-3" aria-hidden />
              research &amp; educational use only
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
