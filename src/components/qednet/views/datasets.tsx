/**
 * Datasets view (F1/F2) — registration table, profiling, validation &
 * leakage-guard reports, CSV upload.
 */
"use client";

import { useState } from "react";
import { CheckCircle2, CloudUpload, FileWarning, Loader2 } from "lucide-react";
import { toast } from "sonner";
import {
  useDatasets,
  useUploadDataset,
  useValidateDataset,
  pct,
  type DatasetMeta,
} from "@/lib/api-client";
import { Button } from "@/components/ui/button";
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
  LoadingState,
  SectionCard,
  StatusBadge,
  TransitionIn,
} from "../ui-bits";

export function DatasetsView() {
  const datasets = useDatasets();
  const upload = useUploadDataset();
  const validate = useValidateDataset();
  const [report, setReport] = useState<Record<string, any> | null>(null);
  const [validating, setValidating] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [name, setName] = useState("");

  function onValidate(dataset: string) {
    setValidating(dataset);
    validate.mutate(dataset, {
      onSuccess: (res) => {
        setReport(res.report ?? null);
        setValidating(null);
        toast.success(`Validation complete for ${dataset}`, {
          description: res.report?.passed ? "All checks passed" : "See the validation report below",
        });
      },
      onError: (e) => {
        setValidating(null);
        toast.error("Validation failed", { description: String(e) });
      },
    });
  }

  function onUpload() {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    if (target.trim()) form.append("target", target.trim());
    if (name.trim()) form.append("name", name.trim());
    upload.mutate(form, {
      onSuccess: (res) => {
        setFile(null);
        setTarget("");
        setName("");
        if (res.ok) {
          toast.success(`Registered dataset "${res.dataset?.name}"`, {
            description: `${res.dataset?.n_samples} samples × ${res.dataset?.n_features} features`,
          });
        } else {
          toast.error("Upload rejected", { description: res.error ?? "invalid file" });
        }
      },
      onError: (e) => toast.error("Upload failed", { description: String(e) }),
    });
  }

  return (
    <div className="space-y-6">
      <TransitionIn className="space-y-6">
        {/* upload card */}
        <SectionCard
          title="Register a dataset (F1 — Biomedical Data Ingestion)"
          subtitle="CSV/TSV upload with schema discovery, datatype inspection and profiling. Identifier-like columns are flagged."
        >
          <div className="flex flex-col gap-3 md:flex-row md:items-end">
            <div className="flex-1 space-y-1">
              <label htmlFor="csv-file" className="text-xs font-medium text-stone-600">
                CSV / TSV file (≤ 25 MB)
              </label>
              <input
                id="csv-file"
                type="file"
                accept=".csv,.tsv,text/csv,text/tab-separated-values"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="qed-focus w-full cursor-pointer rounded-md border border-stone-200 bg-white px-3 py-2 text-xs text-stone-600 file:mr-3 file:rounded file:border-0 file:bg-stone-100 file:px-2.5 file:py-1 file:text-xs file:font-medium file:text-stone-700 hover:file:bg-stone-200"
              />
            </div>
            <div className="w-44 space-y-1">
              <label htmlFor="csv-target" className="text-xs font-medium text-stone-600">
                target column (optional)
              </label>
              <input
                id="csv-target"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="last column"
                className="qed-focus w-full rounded-md border border-stone-200 bg-white px-3 py-2 text-xs text-stone-700"
              />
            </div>
            <div className="w-44 space-y-1">
              <label htmlFor="csv-name" className="text-xs font-medium text-stone-600">
                dataset name (optional)
              </label>
              <input
                id="csv-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="from filename"
                className="qed-focus w-full rounded-md border border-stone-200 bg-white px-3 py-2 text-xs text-stone-700"
              />
            </div>
            <Button onClick={onUpload} disabled={!file || upload.isPending} size="sm" className="gap-2">
              {upload.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> : <CloudUpload className="h-3.5 w-3.5" aria-hidden />}
              register &amp; profile
            </Button>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-stone-400">
            Invalid files, empty datasets, missing target columns and malformed rows are rejected
            with explicit errors. Use public or de-identified research data only — direct
            identifiers (name, email, phone, patient ID) are flagged during validation.
          </p>
        </SectionCard>

        {/* registry table */}
        <SectionCard
          title="Registered datasets"
          subtitle="name · samples · features · class balance · missing values · validation status"
        >
          {datasets.isLoading ? (
            <LoadingState label="Loading dataset registry…" />
          ) : datasets.isError ? (
            <ErrorState message={(datasets.error as Error).message} />
          ) : !datasets.data?.length ? (
            <EmptyState message="No datasets registered." hint="Upload a CSV above or run the bundled dataset setup." />
          ) : (
            <div className="overflow-x-auto qed-scroll">
              <Table>
                <TableHeader>
                  <TableRow className="border-stone-200">
                    <TableHead className="text-xs">Dataset</TableHead>
                    <TableHead className="text-xs">Source</TableHead>
                    <TableHead className="qed-num text-xs text-right">Samples</TableHead>
                    <TableHead className="qed-num text-xs text-right">Features</TableHead>
                    <TableHead className="qed-num text-xs text-right">Positive class</TableHead>
                    <TableHead className="qed-num text-xs text-right">Missing</TableHead>
                    <TableHead className="qed-num text-xs text-right">Duplicates</TableHead>
                    <TableHead className="text-xs">Validation</TableHead>
                    <TableHead className="text-xs text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {datasets.data.map((d: DatasetMeta) => {
                    const missing = Object.values(d.profile?.missing_per_column ?? {}).reduce((a, b) => a + (b as number), 0);
                    return (
                      <TableRow key={d.name} className="border-stone-100">
                        <TableCell className="font-medium text-stone-800">{d.name}</TableCell>
                        <TableCell className="text-xs text-stone-500">{d.source}</TableCell>
                        <TableCell className="qed-num text-right text-stone-600">{d.n_samples}</TableCell>
                        <TableCell className="qed-num text-right text-stone-600">{d.n_features}</TableCell>
                        <TableCell className="qed-num text-right text-stone-600">
                          {pct(d.profile?.class_balance?.["1"] ?? NaN, 1)}
                        </TableCell>
                        <TableCell className="qed-num text-right text-stone-600">{missing}</TableCell>
                        <TableCell className="qed-num text-right text-stone-600">{d.profile?.duplicate_rows ?? 0}</TableCell>
                        <TableCell>
                          <StatusBadge status={d.validation_status} />
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={validating === d.name}
                            onClick={() => onValidate(d.name)}
                            className="h-7 gap-1.5 text-xs"
                          >
                            {validating === d.name ? (
                              <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
                            ) : (
                              <CheckCircle2 className="h-3 w-3" aria-hidden />
                            )}
                            validate
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </SectionCard>

        {/* validation report */}
        {report && (
          <SectionCard
            title={`Validation report — ${report.dataset ?? ""}`}
            subtitle="F2 — schema, datatypes, missing values, duplicates, target & privacy checks"
            right={<StatusBadge status={report.passed ? "passed" : "failed"} />}
          >
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">Checks</h4>
                <ul className="space-y-1.5 text-xs">
                  {Object.entries(report.checks ?? {}).map(([k, v]) => (
                    <li key={k} className="flex items-center gap-2">
                      {v ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" aria-hidden />
                      ) : (
                        <FileWarning className="h-3.5 w-3.5 text-rose-600" aria-hidden />
                      )}
                      <span className="text-stone-600">{k.replace(/_/g, " ")}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">Statistics</h4>
                <ul className="qed-num space-y-1 text-xs text-stone-600">
                  <li>total missing values: {String(report.stats?.total_missing ?? "—")}</li>
                  <li>missing fraction: {pct(report.stats?.missing_fraction ?? NaN, 2)}</li>
                  <li>duplicate rows: {String(report.stats?.duplicate_rows ?? "—")}</li>
                  <li>non-finite values: {String(report.stats?.nonfinite_values ?? "—")}</li>
                  <li>minority class: {pct(Math.min(...Object.values<number>(report.stats?.class_balance ?? { x: 0 })), 1)}</li>
                </ul>
              </div>
            </div>
            {(report.errors?.length || report.warnings?.length) && (
              <div className="mt-4 space-y-2">
                {report.errors?.map((e: string, i: number) => (
                  <p key={i} className="rounded-md border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs text-rose-800">
                    {e}
                  </p>
                ))}
                {report.warnings?.map((w: string, i: number) => (
                  <p key={i} className="rounded-md border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
                    {w}
                  </p>
                ))}
              </div>
            )}
          </SectionCard>
        )}
      </TransitionIn>
    </div>
  );
}
