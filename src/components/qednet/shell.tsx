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
  HeartPulse,
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
import { BodyHealthView } from "./views/body-health";
import { BodyAtlasView } from "./views/body-atlas";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SignOutButton } from "@/components/auth/forms";

export type ViewId =
  | "overview"
  | "datasets"
  | "training"
  | "prediction"
  | "explainability"
  | "benchmark"
  | "advantage"
  | "bodyHealth"
  | "reports";

function BodyHealthWorkspace() {
  return (
    <Tabs defaultValue="research" className="w-full">
      <TabsList className="h-auto w-full justify-start overflow-x-auto rounded-xl bg-stone-100 p-1 sm:w-fit">
        <TabsTrigger value="research" className="min-h-10 px-4 text-sm">Health research</TabsTrigger>
        <TabsTrigger value="atlas" className="min-h-10 px-4 text-sm">3D body atlas</TabsTrigger>
      </TabsList>
      <TabsContent value="research" className="mt-5"><BodyHealthView /></TabsContent>
      <TabsContent value="atlas" className="mt-5"><BodyAtlasView /></TabsContent>
    </Tabs>
  );
}

const NAV: { id: ViewId; label: string; icon: React.ComponentType<{ className?: string }>; hint: string }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, hint: "project status & research summary" },
  { id: "datasets", label: "Datasets", icon: Table2, hint: "F1/F2 — ingestion & validation" },
  { id: "training", label: "Training", icon: FlaskConical, hint: "F3–F7 — configure & launch experiments" },
  { id: "prediction", label: "Prediction", icon: Activity, hint: "F11/F12 — cascade, uncertainty, abstention" },
  { id: "explainability", label: "Explainability", icon: Microscope, hint: "F8 — SHAP, sensitivity, counterfactuals" },
  { id: "benchmark", label: "Benchmark", icon: Scale, hint: "F7 — quantum vs classical results" },
  { id: "advantage", label: "Quantum Advantage", icon: Brain, hint: "F9/F10 — certificate & honesty meter" },
  { id: "bodyHealth", label: "Body Health Atlas", icon: HeartPulse, hint: "whole-body and women’s health research scope" },
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
  bodyHealth: { title: "Body Health Research Atlas", sub: "whole-body evidence map · women’s health · planned research domains" },
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
      bodyHealth: <BodyHealthWorkspace />,
      reports: <ReportsView />,
    }),
    []
  );

  return (
    <div className="qed-app-bg flex min-h-screen w-full text-stone-900">
      {/* ------------------------------------------------ sidebar */}
      <nav
        aria-label="Dashboard sections"
        className="research-sidebar sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-slate-800/70 text-slate-100 md:flex"
      >
        <div className="flex items-center gap-3 border-b border-white/10 px-5 py-5">
          <span className="research-brand-mark"><Beaker className="h-5 w-5" aria-hidden /></span>
          <div>
            <div className="qed-display text-[15px] font-semibold leading-tight text-white">
              QED-Net <span className="text-cyan-300">Research</span>
            </div>
            <div className="mt-1 text-[9px] uppercase tracking-[0.18em] text-slate-400">
              human health · QML
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
                    "qed-focus group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-all",
                    active
                      ? "bg-white text-slate-950 shadow-lg shadow-black/20"
                      : "text-slate-300 hover:bg-white/8 hover:text-white"
                  )}
                >
                  <Icon className={cn("h-4 w-4 shrink-0", active ? "text-cyan-700" : "text-slate-400 group-hover:text-cyan-200")} aria-hidden />
                  <span className="truncate">{item.label}</span>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="space-y-3 border-t border-white/10 px-4 py-4">
          <div className="flex items-center gap-2 text-[11px] text-slate-300">
            <span
              className={cn("inline-block h-2 w-2 rounded-full ring-4", backendOk ? "bg-emerald-400 ring-emerald-400/10" : "bg-rose-400 ring-rose-400/10")}
              aria-hidden
            />
            <span className="truncate">
              {backendOk
                ? `backend online · Python ${pyVer ?? "—"}`
                : "backend offline"}
            </span>
          </div>
          {pennylane && (
              <div className="flex items-center gap-2 text-[11px] text-slate-300">
              <Cpu className="h-3 w-3 text-violet-300" aria-hidden />
              PennyLane {String(pennylane)}
            </div>
          )}
          <ResearchBadge compact />
          <p className="text-[10px] leading-relaxed text-slate-400">
            Not a medical device. Not clinically validated. Not for diagnosis.
          </p>
        </div>
      </nav>

      {/* ------------------------------------------------ main */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 border-b border-slate-200/80 bg-white/80 shadow-sm shadow-slate-900/[0.025] backdrop-blur-xl">
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
            <SignOutButton />
          </div>
          {/* mobile nav */}
          <div className="flex gap-1 overflow-x-auto border-t border-stone-100 px-3 py-2 md:hidden qed-scroll">
            {NAV.map((item) => (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                aria-current={view === item.id ? "page" : undefined}
                className={cn(
                  "qed-focus whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium",
                  view === item.id ? "bg-stone-900 text-white" : "text-stone-500 hover:bg-stone-100"
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </header>

        <div role="note" className="border-b border-cyan-100 bg-cyan-50/80 px-5 py-2 text-xs leading-relaxed text-slate-700 md:px-8">
          Research preview · use public or authorized de-identified data only · not validated for personal medical or surgical decisions.
        </div>

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

        <footer className="mt-auto border-t border-slate-200 bg-white/70 px-5 py-4 text-[11px] leading-relaxed text-slate-500 md:px-8">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              QED-Net 2.0 — Hybrid Quantum–Classical ML research platform.
              Saved experiment metrics come from run artifacts with provenance; example inputs are illustrative.
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
