/**
 * Scientific charts (recharts) — every series is REAL experiment data from
 * the Python backend; nothing here is hard-coded or fabricated.
 *
 * Chart language: hairline grids, small markers, domain-colored strokes,
 * legend outside the plot area (anti-overlap), tabular numerics.
 */
"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  Bar,
  BarChart,
  Cell,
  LabelList,
} from "recharts";
import { modelColor } from "@/lib/api-client";

const GRID = "var(--qed-grid)";
const TICK = { fontSize: 10, fill: "#78716c" } as const;

function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string; color?: string }>;
  label?: number | string;
  formatter?: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-stone-200 bg-white/95 px-3 py-2 text-xs shadow-sm backdrop-blur">
      {label !== undefined && (
        <div className="qed-num mb-1 text-stone-500">
          {typeof label === "number" ? label.toFixed(3) : label}
        </div>
      )}
      {payload.map((p, i) => (
        <div key={i} className="qed-num flex items-center gap-2 text-stone-700">
          <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: p.color }} />
          <span>{p.name}</span>
          <span className="font-medium text-stone-900">
            {typeof p.value === "number" ? (formatter ? formatter(p.value) : p.value.toFixed(3)) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------ ROC / PR ------------------------------ */

function interpMonotone(xs: number[], ys: number[], x: number): number {
  if (xs.length === 0) return x;
  if (x <= xs[0]) return ys[0];
  const last = xs.length - 1;
  if (x >= xs[last]) return ys[last];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

export function RocCurvesChart({
  series,
  height = 240,
}: {
  series: { model: string; fpr: number[]; tpr: number[] }[];
  height?: number;
}) {
  // Resample every model's ROC onto a shared UNIFORM FPR grid by linear
  // interpolation: curves from different models stay aligned by FPR value
  // (not by index), and the chance reference is an exact diagonal.
  const N = 120;
  const data: Record<string, number>[] = [];
  for (let i = 0; i <= N; i++) {
    const x = i / N;
    const row: Record<string, number> = { x, chance: x };
    for (const s of series) {
      if (s.fpr.length && s.tpr.length) row[s.model] = interpMonotone(s.fpr, s.tpr, x);
    }
    data.push(row);
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -18 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis dataKey="x" type="number" domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} label={{ value: "false positive rate", position: "insideBottom", offset: -2, fontSize: 10, fill: "#78716c" }} />
        <YAxis domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={42} />
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} iconType="plainline" iconSize={10} />
        {series.map((s) => (
          <Line
            key={s.model}
            type="monotone"
            dataKey={s.model}
            stroke={modelColor(s.model)}
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
          />
        ))}
        <Line type="linear" dataKey="chance" name="chance" stroke="#d6d3d1" strokeWidth={1} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PrCurvesChart({
  series,
  height = 240,
}: {
  series: { model: string; precision: number[]; recall: number[] }[];
  height?: number;
}) {
  const data: Record<string, number | null>[] = [];
  for (const s of series) {
    const step = Math.max(1, Math.floor(s.precision.length / 100));
    for (let i = 0; i < s.precision.length; i += step) {
      data.push({ x: s.recall[i], [s.model]: s.precision[i] } as Record<string, number | null>);
    }
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -18 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis dataKey="x" type="number" domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} allowDuplicatedCategory={false} label={{ value: "recall", position: "insideBottom", offset: -2, fontSize: 10, fill: "#78716c" }} />
        <YAxis domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={42} />
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} iconType="plainline" iconSize={10} />
        {series.map((s) => (
          <Line
            key={s.model}
            type="monotone"
            dataKey={s.model}
            stroke={modelColor(s.model)}
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------ calibration ------------------------------ */

export function CalibrationChart({
  series,
  height = 240,
}: {
  series: { model: string; centers: number[]; accuracy: number[] }[];
  height?: number;
}) {
  const data: Record<string, number | null>[] = [];
  for (const s of series) {
    for (let i = 0; i < s.centers.length; i++) {
      data.push({ x: s.centers[i], [s.model]: s.accuracy[i] } as Record<string, number | null>);
    }
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -18 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis dataKey="x" type="number" domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} allowDuplicatedCategory={false} label={{ value: "predicted probability", position: "insideBottom", offset: -2, fontSize: 10, fill: "#78716c" }} />
        <YAxis domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={42} />
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} iconType="plainline" iconSize={10} />
        <Line type="linear" data={data.map((d) => ({ x: d.x, ideal: (d.x as number) }))} dataKey="ideal" name="perfect" stroke="#d6d3d1" strokeWidth={1} dot={false} isAnimationActive={false} />
        {series.map((s) => (
          <Line
            key={s.model}
            type="monotone"
            dataKey={s.model}
            stroke={modelColor(s.model)}
            strokeWidth={1.6}
            dot={{ r: 2, fill: modelColor(s.model) }}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------ model comparison ------------------------------ */

export function ModelBarChart({
  data,
  metric = "AUROC",
  height = 260,
  valueFormatter,
}: {
  data: { model: string; value: number; err?: number }[];
  metric?: string;
  height?: number;
  valueFormatter?: (v: number) => string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 14, right: 12, bottom: 4, left: -16 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="model" tick={{ ...TICK, fontSize: 9 }} tickLine={false} axisLine={{ stroke: GRID }} interval={0} angle={-28} textAnchor="end" height={54} />
        <YAxis tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={44} domain={[0, 1]} />
        <Tooltip
          content={({ active, payload }) =>
            active && payload?.length ? (
              <div className="rounded-md border border-stone-200 bg-white/95 px-3 py-2 text-xs shadow-sm">
                <div className="font-medium text-stone-800">{payload[0].payload.model}</div>
                <div className="qed-num text-stone-600">
                  {metric}: <span className="font-medium text-stone-900">{payload[0].payload.value.toFixed(4)}</span>
                  {payload[0].payload.err !== undefined && (
                    <span className="text-stone-400"> ± {payload[0].payload.err.toFixed(4)}</span>
                  )}
                </div>
              </div>
            ) : null
          }
        />
        <Bar dataKey="value" radius={[2, 2, 0, 0]} maxBarSize={34}>
          {data.map((d) => (
            <Cell key={d.model} fill={modelColor(d.model)} fillOpacity={0.82} />
          ))}
          <LabelList
            dataKey="value"
            position="top"
            formatter={(v: number) => (valueFormatter ? valueFormatter(v) : v.toFixed(3))}
            style={{ fontSize: 9, fill: "#57534e", fontVariantNumeric: "tabular-nums" }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------ SHAP-style diverging bars ------------------------------ */

export function DivergingBarChart({
  items,
  height = 300,
  labelWidth = 150,
  unit = "SHAP",
}: {
  items: { label: string; value: number }[];
  height?: number;
  labelWidth?: number;
  unit?: string;
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  const barMax = (280 - labelWidth) / 2;
  return (
    <div className="w-full" style={{ minHeight: height }}>
      <div className="mb-2 text-[10px] uppercase tracking-wide text-stone-400">
        {unit} value (feature contribution; right = increases risk score)
      </div>
      <div className="relative">
        <div className="absolute bottom-0 top-0 border-l border-dashed border-stone-300" style={{ left: labelWidth + barMax }} />
        {items.map((it, idx) => {
          const w = (Math.abs(it.value) / max) * barMax;
          const positive = it.value >= 0;
          return (
            <div key={it.label} className="flex h-6 items-center text-[11px]">
              <div className="truncate pr-2 text-right text-stone-600" style={{ width: labelWidth }} title={it.label}>
                {it.label}
              </div>
              <div className="relative h-3.5" style={{ width: barMax * 2 }}>
                <div
                  className="absolute top-0 h-3.5 rounded-[2px]"
                  style={{
                    width: Math.max(w, 1),
                    left: positive ? barMax : barMax - Math.max(w, 1),
                    background: positive ? "var(--qed-uncertainty)" : "var(--qed-classical)",
                  }}
                />
              </div>
              <div className="qed-num pl-2 text-stone-500">{it.value >= 0 ? "+" : ""}{it.value.toFixed(3)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------ uncertainty histogram ------------------------------ */

export function ProbabilityHistogram({
  probs,
  height = 200,
  bins = 10,
}: {
  probs: number[];
  height?: number;
  bins?: number;
}) {
  const counts = new Array(bins).fill(0);
  for (const p of probs) counts[Math.min(bins - 1, Math.floor(p * bins))]++;
  const data = counts.map((c, i) => ({
    bin: (i + 0.5) / bins,
    count: c,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -22 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="bin" type="number" domain={[0, 1]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} label={{ value: "predicted probability", position: "insideBottom", offset: -2, fontSize: 10, fill: "#78716c" }} />
        <YAxis tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={40} allowDecimals={false} />
        <Tooltip content={<ChartTooltip formatter={(v) => String(Math.round(v))} />} />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={d.bin > 0.35 && d.bin < 0.65 ? "var(--qed-uncertainty)" : "var(--qed-classical)"}
              fillOpacity={0.75}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------ decision curve ------------------------------ */

export function DecisionCurveChart({
  thresholds,
  model,
  treatAll,
  height = 220,
}: {
  thresholds: number[];
  model: number[];
  treatAll: number[];
  height?: number;
}) {
  const data = thresholds.map((t, i) => ({ t, model: model[i], all: treatAll[i] }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -18 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis dataKey="t" type="number" domain={[0.05, 0.95]} tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} label={{ value: "threshold probability", position: "insideBottom", offset: -2, fontSize: 10, fill: "#78716c" }} />
        <YAxis tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={42} />
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} iconType="plainline" iconSize={10} />
        <Line type="monotone" dataKey="model" name="model net benefit" stroke="var(--qed-evaluation)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="all" name="treat all" stroke="#a8a29e" strokeWidth={1.2} dot={false} strokeDasharray="4 3" isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------ scatter (sensitivity) ------------------------------ */

export function SensitivityScatter({
  labels,
  values,
  height = 220,
}: {
  labels: string[];
  values: number[];
  height?: number;
}) {
  const data = values.map((v, i) => ({ label: labels[i], value: v }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis dataKey="label" type="category" tick={{ ...TICK, fontSize: 8 }} tickLine={false} axisLine={{ stroke: GRID }} interval={0} angle={-35} textAnchor="end" height={62} />
        <YAxis dataKey="value" type="number" tick={TICK} tickLine={false} axisLine={{ stroke: GRID }} width={42} />
        <Tooltip content={<ChartTooltip />} />
        <Scatter data={data} dataKey="value" name="mean |Δp|" fill="var(--qed-explain)" fillOpacity={0.7} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
