/**
 * Compare view — side-by-side descriptive comparison of two saved experiments.
 * Every displayed value comes from useExperiments / useExperiment; nothing is
 * synthesised. Cross-run differences are NOT a controlled statistical test.
 */
"use client";

import { useMemo, useState } from "react";
import { AlertTriangle, ArrowLeftRight } from "lucide-react";
import {
  useExperiment,
  useExperiments,
  MODEL_LABELS,
  fmt,
  type ExperimentResult,
  type ModelResult,
} from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  EmptyState,
  ErrorState,
  FamilyBadge,
  LoadingState,
  ResearchBadge,
  SectionCard,
  TransitionIn,
} from "../ui-bits";

function text(value: unknown): string {
  return value === undefined || value === null || value === "" ? "—" : String(value);
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function configHashOf(r: ExperimentResult): string {
  return text(r.reproducibility?.config_hash);
}

function seedsOf(r: ExperimentResult): string {
  const seeds = r.protocol?.seeds;
  return Array.isArray(seeds) ? seeds.map(String).join(", ") : text(seeds);
}

function outerFoldsOf(r: ExperimentResult): string {
  return text(r.protocol?.outer_folds ?? r.protocol?.folds);
}

function protocolSignature(r: ExperimentResult): string {
  return [outerFoldsOf(r), text(r.protocol?.inner_folds), seedsOf(r)].join("|");
}

function hasMetric(result: ExperimentResult | undefined, key: string): boolean {
  if (!result) return false;
  return Object.values(result.models ?? {}).some((m) => m.summary?.[key] != null);
}

function MetricCell({ model, metric }: { model: ModelResult | undefined; metric: string }) {
  const s = model?.summary?.[metric];
  if (!s || s.mean == null) return <span className="qed-num text-stone-400">—</span>;
  return (
    <span className="qed-num text-stone-700" title={`n_obs = ${s.n_obs}`}>
      {fmt(s.mean, 4)}
      <span className="text-stone-400"> ±{fmt(s.std, 3)}</span>
    </span>
  );
}

function ProvenanceRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-stone-100 pb-1">
      <dt className="qed-label shrink-0">{label}</dt>
      <dd className="qed-num min-w-0 break-all text-right text-sm text-stone-700">{value}</dd>
    </div>
  );
}

function Provenance({ id, result }: { id: string; result: ExperimentResult }) {
  return (
    <div className="min-w-0 space-y-2">
      <div className="qed-label border-b border-stone-200 pb-1.5">run {id}</div>
      <dl className="space-y-1">
        <ProvenanceRow label="dataset" value={text(result.dataset)} />
        <ProvenanceRow label="n_samples" value={text(result.n_samples)} />
        <ProvenanceRow label="n_features" value={text(result.n_features)} />
        <ProvenanceRow label="seeds" value={seedsOf(result)} />
        <ProvenanceRow label="outer folds" value={outerFoldsOf(result)} />
        <ProvenanceRow label="inner folds" value={text(result.protocol?.inner_folds)} />
        <ProvenanceRow label="config hash" value={configHashOf(result)} />
      </dl>
    </div>
  );
}

export function CompareView() {
  const experiments = useExperiments();
  const [idA, setIdA] = useState<string | null>(null);
  const [idB, setIdB] = useState<string | null>(null);

  const list = experiments.data ?? [];
  const selectedA = idA ?? list[0]?.experiment_id ?? null;
  const selectedB = idB ?? list[1]?.experiment_id ?? list[0]?.experiment_id ?? null;

  // Both hooks are called unconditionally; useExperiment is disabled on null.
  const queryA = useExperiment(selectedA);
  const queryB = useExperiment(selectedB);
  const resultA = queryA.data?.result;
  const resultB = queryB.data?.result;

  const modelKeys = useMemo(() => {
    const keys = new Set([
      ...Object.keys(resultA?.models ?? {}),
      ...Object.keys(resultB?.models ?? {}),
    ]);
    return [...keys].sort((x, y) => x.localeCompare(y));
  }, [resultA, resultB]);

  if (experiments.isError) {
    return <ErrorState message={`Could not load experiments: ${messageOf(experiments.error)}`} />;
  }
  if (queryA.isError || queryB.isError) {
    const errors: string[] = [];
    if (queryA.isError) errors.push(`A: ${messageOf(queryA.error)}`);
    if (queryB.isError) errors.push(`B: ${messageOf(queryB.error)}`);
    return <ErrorState message={`Could not load selected experiment — ${errors.join(" · ")}`} />;
  }
  if (experiments.isLoading) return <LoadingState label="Loading experiments…" />;
  if (list.length < 2) {
    return (
      <EmptyState
        message="At least two saved experiments are needed to compare."
        hint="Run experiments from the Training view; completed results appear here with full provenance."
      />
    );
  }
  if (queryA.isLoading || queryB.isLoading) {
    return <LoadingState label="Loading both experiment results…" />;
  }
  if (!resultA || !resultB) {
    return <ErrorState message="One or both selected experiments returned no result payload." />;
  }

  const datasetDiffers =
    resultA.dataset !== resultB.dataset ||
    (!!resultA.dataset_fingerprint &&
      !!resultB.dataset_fingerprint &&
      resultA.dataset_fingerprint !== resultB.dataset_fingerprint);
  const protocolDiffers =
    configHashOf(resultA) !== configHashOf(resultB) ||
    protocolSignature(resultA) !== protocolSignature(resultB);
  const showAuprc = hasMetric(resultA, "auprc") || hasMetric(resultB, "auprc");

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        {/* selection */}
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-64 flex-1 space-y-1.5">
            <label className="qed-label" htmlFor="cmp-a">experiment A</label>
            <Select value={selectedA ?? undefined} onValueChange={setIdA}>
              <SelectTrigger id="cmp-a" className="qed-focus h-9 w-full text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {list.map((e) => (
                  <SelectItem
                    key={e.experiment_id}
                    value={e.experiment_id}
                    disabled={e.experiment_id === selectedB}
                    className="text-sm"
                  >
                    {e.experiment_id} — {e.dataset}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon"
            aria-label="Swap experiments A and B"
            title="Swap experiments A and B"
            onClick={() => {
              setIdA(selectedB);
              setIdB(selectedA);
            }}
            className="qed-focus h-9 w-9 shrink-0"
          >
            <ArrowLeftRight className="h-4 w-4 text-stone-500" aria-hidden />
          </Button>
          <div className="min-w-64 flex-1 space-y-1.5">
            <label className="qed-label" htmlFor="cmp-b">experiment B</label>
            <Select value={selectedB ?? undefined} onValueChange={setIdB}>
              <SelectTrigger id="cmp-b" className="qed-focus h-9 w-full text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {list.map((e) => (
                  <SelectItem
                    key={e.experiment_id}
                    value={e.experiment_id}
                    disabled={e.experiment_id === selectedA}
                    className="text-sm"
                  >
                    {e.experiment_id} — {e.dataset}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* honest framing */}
        <p className="max-w-3xl text-sm leading-relaxed text-stone-500">
          This is a descriptive comparison of two stored runs — not a controlled statistical
          comparison. The runs were not executed as paired arms of one experiment, so any
          difference may reflect the dataset, protocol, or configuration rather than model
          behaviour. Do not read it as evidence of superiority.
        </p>
        {(datasetDiffers || protocolDiffers) && (
          <div
            role="status"
            className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50/60 px-3 py-2 text-sm text-amber-900"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>
              {datasetDiffers && protocolDiffers
                ? "Dataset (or dataset fingerprint) and run configuration/protocol differ between A and B."
                : datasetDiffers
                  ? "Dataset (or dataset fingerprint) differs between A and B."
                  : "Run configuration/protocol differs between A and B (config hash, folds, or seeds)."}
              {" "}These runs are not directly comparable.
            </span>
          </div>
        )}

        {/* provenance */}
        <SectionCard
          title="Run provenance"
          subtitle="values read from each experiment result; config hash from its reproducibility record"
        >
          <div className="grid gap-6 md:grid-cols-2">
            <Provenance id={selectedA ?? "—"} result={resultA} />
            <Provenance id={selectedB ?? "—"} result={resultB} />
          </div>
        </SectionCard>

        {/* metrics */}
        <SectionCard
          title="Model metrics — mean ± std"
          subtitle="as stored in each result (over folds × seeds); hover a value for its observation count"
        >
          {modelKeys.length === 0 ? (
            <EmptyState message="Neither selected result contains model metrics." />
          ) : (
            <div className="qed-scroll overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-stone-200">
                    <TableHead className="text-sm">Model</TableHead>
                    <TableHead className="text-sm">Family</TableHead>
                    <TableHead className="qed-num text-right text-sm">AUROC A</TableHead>
                    <TableHead className="qed-num text-right text-sm">AUROC B</TableHead>
                    {showAuprc && (
                      <>
                        <TableHead className="qed-num text-right text-sm">AUPRC A</TableHead>
                        <TableHead className="qed-num text-right text-sm">AUPRC B</TableHead>
                      </>
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {modelKeys.map((key) => {
                    const mA = resultA.models[key];
                    const mB = resultB.models[key];
                    const family = mA?.family ?? mB?.family ?? "unknown";
                    return (
                      <TableRow key={key} className="border-stone-100">
                        <TableCell className="font-medium text-stone-800">
                          {MODEL_LABELS[key] ?? key}
                        </TableCell>
                        <TableCell>
                          <FamilyBadge family={family} />
                        </TableCell>
                        <TableCell className="qed-num text-right">
                          <MetricCell model={mA} metric="auroc" />
                        </TableCell>
                        <TableCell className="qed-num text-right">
                          <MetricCell model={mB} metric="auroc" />
                        </TableCell>
                        {showAuprc && (
                          <>
                            <TableCell className="qed-num text-right">
                              <MetricCell model={mA} metric="auprc" />
                            </TableCell>
                            <TableCell className="qed-num text-right">
                              <MetricCell model={mB} metric="auprc" />
                            </TableCell>
                          </>
                        )}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </SectionCard>

        <div className="flex flex-wrap items-center gap-2 rounded-md border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-600">
          <ResearchBadge compact />
          <span>Research and educational use only — not a medical device, not clinically validated, not for diagnosis.</span>
        </div>
      </TransitionIn>
    </div>
  );
}
