/**
 * Training view — configure dataset/model/encoding/compression/seeds/CV,
 * launch background jobs, watch live progress (queued/running/completed/
 * failed) with elapsed time and real per-fold feedback.
 */
"use client";

import { useEffect, useMemo, useState } from "react";
import { Loader2, Play, Timer, XCircle, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import {
  useDatasets,
  useExperiments,
  useJobs,
  useStartTraining,
  seconds,
} from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  EmptyState,
  LoadingState,
  SectionCard,
  StatusBadge,
  TransitionIn,
} from "../ui-bits";

const CONFIG_NAMES = [
  { value: "breast_cancer", label: "Breast Cancer (Tier 1)" },
  { value: "heart_disease", label: "Heart Disease (Tier 1)" },
  { value: "parkinsons", label: "Parkinson's (Tier 1)" },
  { value: "pima_diabetes", label: "Pima Diabetes (Tier 2)" },
];

export function TrainingView() {
  const jobs = useJobs();
  const experiments = useExperiments();
  const datasets = useDatasets();
  const start = useStartTraining();

  const [configName, setConfigName] = useState("breast_cancer");
  const [expId, setExpId] = useState("exp_custom_1");

  const liveJob = useMemo(
    () => jobs.data?.find((j) => j.status === "running") ?? null,
    [jobs.data]
  );

  useEffect(() => {
    if (liveJob && jobs.data) {
      // toast once per job start
      const seen = sessionStorage.getItem(`seen-${liveJob.jobId}`);
      if (!seen) {
        sessionStorage.setItem(`seen-${liveJob.jobId}`, "1");
        toast.info(`Training ${liveJob.experimentId} started`, {
          description: "background job — progress updates every 3 s",
        });
      }
    }
  }, [liveJob, jobs.data]);

  function launch() {
    const clean = expId.replace(/[^a-zA-Z0-9_-]/g, "_") || `exp_${Date.now()}`;
    start.mutate(
      { configName, experimentId: clean },
      {
        onSuccess: (res) => {
          if (res.ok) {
            toast.success(`Experiment ${clean} launched`, {
              description: `${configName}.yaml · job ${res.job?.jobId}`,
            });
          } else {
            toast.error("Could not launch experiment", { description: res.error });
          }
        },
        onError: (e) => toast.error("Launch failed", { description: String(e) }),
      }
    );
  }

  const p = liveJob?.progress;

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        {/* configuration */}
        <SectionCard
          title="Experiment configuration"
          subtitle="dataset · compression · model · cross-validation · seeds — everything is YAML-driven and logged"
        >
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            <div className="space-y-1.5">
              <Label className="text-xs">Dataset (config)</Label>
              <Select value={configName} onValueChange={setConfigName}>
                <SelectTrigger className="qed-focus h-9 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {CONFIG_NAMES.map((c) => (
                    <SelectItem key={c.value} value={c.value} className="text-xs">
                      {c.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Experiment ID</Label>
              <Input
                value={expId}
                onChange={(e) => setExpId(e.target.value)}
                placeholder="exp_custom_1"
                className="qed-focus h-9 text-xs"
              />
            </div>
            <div className="space-y-1.5 xl:col-span-2">
              <Label className="text-xs flex justify-between">
                <span>Protocol (from YAML)</span>
                <span className="text-stone-400">5-fold outer · 3-fold inner · 3 seeds</span>
              </Label>
              <div className="rounded-md border border-stone-200 bg-stone-50/60 px-3 py-2 text-[11px] leading-relaxed text-stone-500">
                The launcher runs the full F1→F12 pipeline: ingestion → validation →
                preprocessing/compression (PCA→8) → classical suite + quantum zoo →
                nested-CV benchmarking → explainability → certificate → honesty meter → cascade.
                Edit <code className="text-stone-700">configs/experiments/{configName}.yaml</code> for
                custom protocols.
              </div>
            </div>
          </div>
          <div className="mt-5 flex items-center gap-3">
            <Button
              onClick={launch}
              disabled={start.isPending || !!liveJob}
              size="sm"
              className="gap-2"
            >
              {start.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
              ) : (
                <Play className="h-3.5 w-3.5" aria-hidden />
              )}
              launch experiment
            </Button>
            {liveJob && (
              <span className="text-xs text-stone-500">
                <StatusBadge status="running" /> {liveJob.experimentId} — one experiment at a time
              </span>
            )}
          </div>
        </SectionCard>

        {/* live job */}
        {liveJob && (
          <SectionCard
            title={`Running: ${liveJob.experimentId}`}
            subtitle={`job ${liveJob.jobId} · PID ${liveJob.pid ?? "—"}`}
            right={
              <span className="qed-num inline-flex items-center gap-1.5 text-xs text-stone-500">
                <Timer className="h-3.5 w-3.5" aria-hidden />
                {seconds((Date.now() - liveJob.startedAt) / 1000)}
              </span>
            }
          >
            {p ? (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-stone-600">
                  <span>
                    stage: <strong className="text-stone-800">{p.stage}</strong>
                  </span>
                  {p.seed !== null && p.seed !== undefined && (
                    <span className="qed-num">seed {p.seed} · fold {p.fold}</span>
                  )}
                  {p.model && (
                    <span className="qed-num font-medium text-stone-800">{p.model}</span>
                  )}
                  {p.auroc !== undefined && (
                    <span className="qed-num text-stone-500">last fold AUROC {p.auroc.toFixed(4)}</span>
                  )}
                </div>
                {p.completed_fits != null && p.fits_per_seed != null && p.seeds_total != null && (
                  <div>
                    <div className="qed-num mb-1 flex justify-between text-[11px] text-stone-500">
                      <span>model fits completed</span>
                      <span>
                        {p.completed_fits} / {(p.seeds_total ?? 0) * (p.fits_per_seed ?? 0)}
                      </span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-stone-200">
                      <div
                        className="h-full rounded-full bg-stone-800 transition-[width] duration-500"
                        style={{
                          width: `${Math.min(100, (p.completed_fits / Math.max(1, (p.seeds_total ?? 0) * (p.fits_per_seed ?? 0))) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                )}
                {liveJob.logTail && (
                  <pre className="qed-scroll qed-num max-h-32 overflow-y-auto rounded-md border border-stone-200 bg-stone-950 p-3 text-[10px] leading-relaxed text-stone-300">
                    {liveJob.logTail.split("\n").slice(-8).join("\n")}
                  </pre>
                )}
              </div>
            ) : (
              <LoadingState label="Waiting for first progress update…" />
            )}
          </SectionCard>
        )}

        {/* job history */}
        <SectionCard title="Background jobs" subtitle="long-running quantum jobs never block the UI">
          {jobs.isLoading ? (
            <LoadingState label="Loading jobs…" />
          ) : !jobs.data?.length ? (
            <EmptyState
              message="No training jobs yet."
              hint="Launch an experiment above — jobs run detached and report live progress."
            />
          ) : (
            <ul className="space-y-2">
              {jobs.data.slice(0, 8).map((j) => (
                <li
                  key={j.jobId}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-stone-100 bg-stone-50/50 px-3 py-2.5"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    {j.status === "completed" ? (
                      <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" aria-hidden />
                    ) : j.status === "failed" ? (
                      <XCircle className="h-4 w-4 shrink-0 text-rose-600" aria-hidden />
                    ) : (
                      <Loader2 className="h-4 w-4 shrink-0 animate-spin text-amber-600" aria-hidden />
                    )}
                    <div className="min-w-0">
                      <div className="qed-num truncate text-xs font-medium text-stone-800">
                        {j.experimentId}
                      </div>
                      <div className="qed-num text-[11px] text-stone-400">
                        {j.configName}.yaml · started{" "}
                        {new Date(j.startedAt).toLocaleTimeString()}
                        {j.endedAt && ` · ended ${new Date(j.endedAt).toLocaleTimeString()}`}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {j.progress?.auroc !== undefined && (
                      <Badge variant="outline" className="qed-num text-[10px] font-normal">
                        AUROC {j.progress.auroc.toFixed(4)}
                      </Badge>
                    )}
                    <StatusBadge status={j.status} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>

        {/* completed experiments */}
        <SectionCard
          title="Completed experiments"
          subtitle="each row is a full F1→F12 run with stored artifacts"
        >
          {experiments.isLoading ? (
            <LoadingState />
          ) : !experiments.data?.length ? (
            <EmptyState message="No experiments recorded." />
          ) : (
            <ul className="grid gap-2 md:grid-cols-2">
              {experiments.data.map((e) => (
                <li
                  key={e.experiment_id}
                  className="flex items-center justify-between gap-3 rounded-md border border-stone-100 bg-white px-3 py-2.5"
                >
                  <div className="min-w-0">
                    <div className="qed-num text-xs font-medium text-stone-800">{e.experiment_id}</div>
                    <div className="qed-num text-[11px] text-stone-500">
                      {e.dataset} · {e.models.length} models · {seconds(e.runtime_s)}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {e.evidence_classification && (
                      <Badge variant="outline" className="max-w-44 truncate text-[10px] font-normal text-stone-600">
                        {e.evidence_classification}
                      </Badge>
                    )}
                    <StatusBadge status={e.validation_passed ? "passed" : "failed"} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>
      </TransitionIn>
    </div>
  );
}
