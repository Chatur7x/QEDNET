"use client";

import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import {
  Activity,
  Bone,
  Brain,
  Check,
  CircleDot,
  Database,
  Download,
  HeartPulse,
  Leaf,
  Plus,
  ShieldCheck,
  Trash2,
  Wind,
} from "lucide-react";
import { useDatasets } from "@/lib/api-client";
import {
  BODY_SYSTEMS,
  registeredDatasetsForSystem,
  type BodySystemId,
} from "@/lib/body-research";
import { cn } from "@/lib/utils";
import {
  cycleEntriesToCsv,
  validateCycleJournalDraft,
  type CycleJournalDraft,
  type CycleJournalEntry,
} from "@/lib/cycle-journal";

const ICONS = {
  neurological: Brain,
  cardiovascular: HeartPulse,
  respiratory: Wind,
  metabolic: Activity,
  digestive: Leaf,
  renal: CircleDot,
  musculoskeletal: Bone,
  reproductive: Activity,
  oncology: CircleDot,
  immune: ShieldCheck,
} satisfies Record<BodySystemId, typeof Activity>;

const LENSES = [
  { id: "whole", label: "Whole person" },
  { id: "women", label: "Women’s health" },
  { id: "men", label: "Men’s health" },
] as const;

type Lens = (typeof LENSES)[number]["id"];
type StudyBlueprint = "cycle" | "wearable" | "surgical";

const BLUEPRINTS: Record<StudyBlueprint, { title: string; fields: string[]; note: string }> = {
  cycle: {
    title: "Menstrual health study design",
    fields: ["Bleeding dates & spotting", "Symptoms, pain & context", "Optional device/source metadata"],
    note: "A diary schema preview only. No personal cycle data is collected or saved here.",
  },
  wearable: {
    title: "Wearable study design",
    fields: ["Device & firmware provenance", "Timestamp, units & quality flags", "Gaps, drift & participant-level splits"],
    note: "No device is connected. A study needs an approved source and validation protocol.",
  },
  surgical: {
    title: "Perioperative study design",
    fields: ["Prespecified preoperative index time", "Procedure-specific outcomes", "Site, patient & time-separated evaluation"],
    note: "Research workflow only. No operative guidance or treatment recommendations.",
  },
};

export function BodyHealthView() {
  const datasets = useDatasets();
  const reduceMotion = useReducedMotion();
  const [selected, setSelected] = useState<BodySystemId>("reproductive");
  const [lens, setLens] = useState<Lens>("whole");
  const [blueprint, setBlueprint] = useState<StudyBlueprint>("cycle");
  const activeSystem = BODY_SYSTEMS.find((system) => system.id === selected)!;
  const names = datasets.data?.map((dataset) => dataset.name) ?? [];
  const available = registeredDatasetsForSystem(selected, names);
  const ActiveIcon = ICONS[selected];

  function chooseLens(nextLens: Lens) {
    setLens(nextLens);
    if (nextLens === "women") {
      setSelected("reproductive");
      setBlueprint("cycle");
    }
    if (nextLens === "men") setSelected("reproductive");
  }

  return (
    <div className="space-y-7">
      <section className="health-atlas-hero relative overflow-hidden rounded-[1.75rem] p-6 text-white shadow-2xl shadow-slate-950/10 sm:p-9 xl:p-11">
        <div className="health-atlas-glow" aria-hidden />
        <div className="relative z-[1] grid items-center gap-8 xl:grid-cols-[1.12fr_0.88fr]">
          <div className="max-w-2xl">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/8 px-3 py-1.5 text-[11px] font-medium text-cyan-100 backdrop-blur">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />
              RESEARCH ATLAS · NOT A CLINICAL TOOL
            </div>
            <h2 className="qed-display max-w-xl text-3xl font-semibold leading-[1.08] tracking-[-0.035em] sm:text-5xl">
              Human health is connected. Research should be, too.
            </h2>
            <p className="mt-5 max-w-xl text-sm leading-7 text-slate-300 sm:text-base">
              Explore a whole-body research roadmap grounded in real dataset availability. Select a system to see what can be studied now and what still needs governed data.
            </p>
            <div className="mt-7 flex flex-wrap gap-2" role="group" aria-label="Research population lens">
              {LENSES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={lens === item.id}
                  onClick={() => chooseLens(item.id)}
                  className={cn(
                    "qed-focus rounded-full border px-3.5 py-2 text-xs font-medium transition-colors",
                    lens === item.id
                      ? "border-cyan-200/70 bg-cyan-100 text-slate-950"
                      : "border-white/15 bg-white/5 text-slate-200 hover:bg-white/10",
                  )}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <p className="mt-3 max-w-lg text-xs leading-5 text-slate-400">
              {lens === "women"
                ? "Women’s health is a full-body research lens. The cycle track is a design blueprint; no cycle records are being collected."
                : lens === "men"
                  ? "Men’s health research can span every body system. No sex or gender is inferred from a dataset or person’s name."
                  : "Population and life-course variables belong in a study protocol; this view does not infer sex, gender, or health status."}
            </p>
          </div>

          <div className="health-atlas-stage mx-auto w-full max-w-[470px]" aria-label="Interactive body systems map">
            <div className="atlas-orbit atlas-orbit-one" aria-hidden />
            <div className="atlas-orbit atlas-orbit-two" aria-hidden />
            <motion.div
              className="atlas-figure"
              animate={reduceMotion ? undefined : { y: [0, -5, 0] }}
              transition={reduceMotion ? undefined : { duration: 7, repeat: Infinity, ease: "easeInOut" }}
            >
              <svg viewBox="0 0 240 500" className="h-[390px] w-[190px] sm:h-[440px] sm:w-[220px]" role="img" aria-label="Abstract human body silhouette with selectable research areas">
                <defs>
                  <linearGradient id="body-fill" x1="0" x2="1" y1="0" y2="1">
                    <stop offset="0" stopColor="#e0f2fe" stopOpacity="0.98" />
                    <stop offset="0.55" stopColor="#bae6fd" stopOpacity="0.72" />
                    <stop offset="1" stopColor="#a5b4fc" stopOpacity="0.48" />
                  </linearGradient>
                  <radialGradient id="body-core">
                    <stop stopColor="#67e8f9" stopOpacity="0.82" />
                    <stop offset="1" stopColor="#67e8f9" stopOpacity="0" />
                  </radialGradient>
                  <filter id="body-shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="8" />
                  </filter>
                </defs>
                <ellipse cx="120" cy="475" rx="58" ry="10" fill="#020617" opacity="0.48" filter="url(#body-shadow)" />
                <g fill="url(#body-fill)" stroke="#e0f2fe" strokeOpacity="0.72" strokeWidth="1.5">
                  <ellipse cx="120" cy="49" rx="27" ry="34" />
                  <path d="M108 80h24v31h-24z" />
                  <path d="M92 112c-17 6-26 21-26 41l-1 58c0 12 17 14 18 1l5-49 5 73-13 91c-2 14 19 18 23 4l17-74 17 74c4 14 25 10 23-4l-13-91 5-73 5 49c1 13 18 11 18-1l-1-58c0-20-9-35-26-41-18-7-43-7-60 0z" />
                  <path d="M68 127c-10 3-16 12-19 25l-16 79c-3 13 15 18 19 5l23-75 4-22z" />
                  <path d="M172 127c10 3 16 12 19 25l16 79c3 13-15 18-19 5l-23-75-4-22z" />
                  <path d="M88 306l-9 92c-1 13 18 15 21 2l21-80 21 80c3 13 22 11 21-2l-9-92z" />
                </g>
                <g fill="url(#body-core)">
                  <circle cx="120" cy="49" r="28" />
                  <circle cx="120" cy="170" r="48" />
                  <circle cx="120" cy="260" r="48" />
                  <circle cx="120" cy="352" r="45" />
                </g>
                <path d="M120 94v276" fill="none" stroke="#cffafe" strokeDasharray="3 9" strokeOpacity="0.65" />
                <g fill="#e0f2fe" fillOpacity="0.9">
                  <circle cx="120" cy="145" r="4" />
                  <circle cx="120" cy="193" r="4" />
                  <circle cx="120" cy="244" r="4" />
                  <circle cx="120" cy="288" r="4" />
                  <circle cx="120" cy="342" r="4" />
                </g>
              </svg>
            </motion.div>

            {BODY_SYSTEMS.map((system) => {
              const isSelected = system.id === selected;
              const hasDataset = registeredDatasetsForSystem(system.id, names).length > 0;
              return (
                <motion.button
                  key={system.id}
                  type="button"
                  aria-label={`${system.label}${hasDataset ? ", dataset registered" : ", no dataset registered"}`}
                  aria-pressed={isSelected}
                  title={system.label}
                  onClick={() => setSelected(system.id)}
                  whileHover={reduceMotion ? undefined : { scale: 1.16, z: 12 }}
                  whileTap={reduceMotion ? undefined : { scale: 0.94 }}
                  className={cn("atlas-hotspot qed-focus", isSelected && "is-active", hasDataset && "has-data")}
                  style={{ top: system.point.top, left: system.point.left }}
                >
                  <span className="sr-only">{system.label}</span>
                </motion.button>
              );
            })}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-slate-950/45 px-3 py-1 text-[10px] text-slate-300 backdrop-blur">
              select a system to inspect its evidence
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[0.82fr_1.18fr]">
        <div className="health-panel rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-700">Selected research area</p>
              <h3 className="mt-2 text-xl font-semibold tracking-tight text-slate-950">{activeSystem.label}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{activeSystem.description}</p>
            </div>
            <div className="rounded-xl bg-cyan-50 p-3 text-cyan-800"><ActiveIcon className="h-5 w-5" aria-hidden /></div>
          </div>
          <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50/90 p-4">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-800">
              <Database className="h-4 w-4 text-cyan-700" aria-hidden />
              Dataset registry
            </div>
            {datasets.isLoading ? (
              <p className="mt-2 text-xs text-slate-500">Checking registered datasets…</p>
            ) : datasets.isError ? (
              <p className="mt-2 text-xs text-amber-800">Registry unavailable. No availability claim is shown.</p>
            ) : available.length ? (
              <div className="mt-3 space-y-2">
                {available.map((name) => (
                  <div key={name} className="flex items-center justify-between gap-2 rounded-lg bg-white px-3 py-2 text-xs text-slate-700 ring-1 ring-slate-200">
                    <span className="font-medium">{name}</span><span className="text-emerald-700">registered benchmark</span>
                  </div>
                ))}
                <p className="text-[11px] leading-5 text-slate-500">A registered benchmark does not establish early detection or clinical validity.</p>
              </div>
            ) : (
              <p className="mt-2 text-xs leading-5 text-slate-600">No dataset for this area is currently registered. The study needs a permitted source, protocol, and reference standard before modeling.</p>
            )}
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="rounded-full bg-slate-100 px-3 py-1.5 text-[11px] font-medium text-slate-600">Classical baselines first</span>
            <span className="rounded-full bg-violet-50 px-3 py-1.5 text-[11px] font-medium text-violet-700">Quantum models are research challengers</span>
          </div>
        </div>

        <div className="health-panel rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-violet-700">Study blueprint</p>
              <h3 className="mt-2 text-xl font-semibold tracking-tight text-slate-950">From health question to reproducible evidence</h3>
            </div>
            <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-amber-900">design preview</span>
          </div>
          <div className="mt-5 flex flex-wrap gap-2" role="tablist" aria-label="Research blueprint">
            {([
              ["cycle", "Women’s health"],
              ["wearable", "Wearables"],
              ["surgical", "Surgical outcomes"],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={blueprint === id}
                onClick={() => setBlueprint(id)}
                className={cn("qed-focus rounded-lg px-3 py-2 text-xs font-medium transition-colors", blueprint === id ? "bg-slate-950 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200")}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="mt-4 rounded-xl bg-gradient-to-br from-slate-50 to-cyan-50/60 p-4 sm:p-5" role="tabpanel">
            <h4 className="text-sm font-semibold text-slate-900">{BLUEPRINTS[blueprint].title}</h4>
            <ol className="mt-4 grid gap-2 sm:grid-cols-3">
              {BLUEPRINTS[blueprint].fields.map((field, index) => (
                <li key={field} className="flex min-h-16 items-start gap-2 rounded-lg border border-white/80 bg-white/75 p-3 text-xs leading-5 text-slate-700 shadow-sm">
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-cyan-100 text-[10px] font-bold text-cyan-900">{index + 1}</span>
                  {field}
                </li>
              ))}
            </ol>
            <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-200/80 bg-amber-50/80 p-3 text-[11px] leading-5 text-amber-950">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {BLUEPRINTS[blueprint].note}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4 text-xs text-slate-500">
            <span className="inline-flex items-center gap-2"><Check className="h-4 w-4 text-emerald-600" aria-hidden /> Protocol-led, participant-safe research</span>
            <span className="text-slate-600">Plan: docs/WHOLE_BODY_HEALTH_RESEARCH_PLAN.md</span>
          </div>
        </div>
      </section>

      {lens === "women" && <CycleJournalPrototype />}

      <section aria-labelledby="body-system-list" className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Evidence map</p>
            <h3 id="body-system-list" className="mt-1 text-lg font-semibold tracking-tight text-slate-950">Research areas across the body</h3>
          </div>
          <p className="text-xs text-slate-500">Availability comes from the live dataset registry.</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {BODY_SYSTEMS.map((system) => {
            const systemDatasets = registeredDatasetsForSystem(system.id, names);
            const Icon = ICONS[system.id];
            return (
              <button
                key={system.id}
                type="button"
                aria-pressed={selected === system.id}
                onClick={() => setSelected(system.id)}
                className={cn("qed-focus group rounded-xl border p-4 text-left transition-all hover:-translate-y-0.5 hover:shadow-md", selected === system.id ? "border-cyan-300 bg-cyan-50/80 shadow-sm" : "border-slate-200 bg-white hover:border-slate-300")}
              >
                <div className="flex items-center justify-between gap-2">
                  <Icon className={cn("h-4 w-4", selected === system.id ? "text-cyan-800" : "text-slate-400 group-hover:text-slate-600")} aria-hidden />
                  <span className={cn("rounded-full px-2 py-1 text-[9px] font-semibold uppercase tracking-wide", systemDatasets.length ? "bg-emerald-50 text-emerald-800" : "bg-slate-100 text-slate-500")}>
                    {systemDatasets.length ? "dataset" : "planned"}
                  </span>
                </div>
                <p className="mt-3 text-xs font-semibold leading-5 text-slate-900">{system.label}</p>
                <p className="mt-1 line-clamp-2 text-[10px] leading-4 text-slate-500">{systemDatasets.length ? systemDatasets.join(", ") : "No matching dataset registered"}</p>
              </button>
            );
          })}
        </div>
      </section>

      <p className="rounded-xl border border-slate-200 bg-white/70 px-4 py-3 text-[11px] leading-5 text-slate-500">
        Research and educational use only. This atlas does not collect health data, calculate personal risk, diagnose disease, or direct clinical or surgical decisions.
      </p>
    </div>
  );
}

const SYMPTOM_OPTIONS = ["Cramps", "Headache", "Fatigue", "Bloating", "Mood changes"] as const;

const EMPTY_DRAFT: CycleJournalDraft = {
  bleeding: "unknown",
  eventDate: "",
  endDate: "",
  flow: "",
  symptoms: [],
  pain: "",
  notes: "",
};

function CycleJournalPrototype() {
  const [entries, setEntries] = useState<CycleJournalEntry[]>([]);
  const [draft, setDraft] = useState<CycleJournalDraft>(EMPTY_DRAFT);
  const [error, setError] = useState<string | null>(null);
  const bleedingRecords = entries.filter((entry) => entry.bleeding === "bleeding").length;
  const noBleedingRecords = entries.filter((entry) => entry.bleeding === "none").length;
  const unknownRecords = entries.filter((entry) => entry.bleeding === "unknown").length;

  function updateDraft<K extends keyof CycleJournalDraft>(key: K, value: CycleJournalDraft[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
  }

  function toggleSymptom(symptom: string) {
    setDraft((current) => ({
      ...current,
      symptoms: current.symptoms.includes(symptom)
        ? current.symptoms.filter((item) => item !== symptom)
        : [...current.symptoms, symptom],
    }));
  }

  function changeBleedingStatus(value: CycleJournalDraft["bleeding"]) {
    setDraft((current) => ({
      ...current,
      bleeding: value,
      eventDate: "",
      endDate: "",
      flow: "",
    }));
    setError(null);
  }

  function addEntry(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const validationError = validateCycleJournalDraft(draft);
    if (validationError) {
      setError(validationError);
      return;
    }
    setEntries((current) => [{
      ...draft,
      id: crypto.randomUUID(),
      recordedAt: new Date().toISOString(),
      symptoms: [...draft.symptoms],
    }, ...current]);
    setDraft(EMPTY_DRAFT);
    setError(null);
  }

  function exportEntries() {
    const blob = new Blob([cycleEntriesToCsv(entries)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "qednet-cycle-journal.csv";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return (
    <section aria-labelledby="cycle-journal-title" className="health-panel rounded-2xl border border-violet-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-violet-700">Interactive prototype · no account data saved</p>
          <h3 id="cycle-journal-title" className="mt-2 text-xl font-semibold tracking-tight text-slate-950">Women’s health cycle journal</h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">A personal record-keeping demo for a future research study. It does not estimate cycle dates, identify conditions, or send entries to the server.</p>
        </div>
        {entries.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={exportEntries} className="qed-focus inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50">
              <Download className="h-4 w-4" aria-hidden /> Export CSV
            </button>
            <button type="button" onClick={() => setEntries([])} className="qed-focus inline-flex items-center gap-2 rounded-lg border border-rose-200 px-3 py-2 text-xs font-medium text-rose-700 hover:bg-rose-50">
              <Trash2 className="h-4 w-4" aria-hidden /> Clear journal
            </button>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-950">
        Prototype privacy boundary: entries stay in page memory and disappear when you reload or leave this view. Do not enter identifying or sensitive real health information in this demo.
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <form onSubmit={addEntry} className="space-y-4 rounded-xl bg-slate-50 p-4 sm:p-5">
          <div>
            <label htmlFor="cycle-bleeding" className="text-xs font-semibold text-slate-800">Bleeding record</label>
            <select id="cycle-bleeding" value={draft.bleeding} onChange={(event) => changeBleedingStatus(event.target.value as CycleJournalDraft["bleeding"])} className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800">
              <option value="unknown">Unknown / not recorded</option>
              <option value="bleeding">Bleeding occurred</option>
              <option value="none">No bleeding on this record</option>
            </select>
          </div>

          <div className={cn("grid gap-3", draft.bleeding === "bleeding" && "sm:grid-cols-2")}>
            <div>
              <label htmlFor="cycle-event-date" className="text-xs font-medium text-slate-700">
                {draft.bleeding === "bleeding" ? "Period start date" : draft.bleeding === "none" ? "Observation date" : "Date"}
                {draft.bleeding === "unknown" && <span className="font-normal text-slate-500"> (optional)</span>}
              </label>
              <input id="cycle-event-date" type="date" required={draft.bleeding !== "unknown"} value={draft.eventDate} onChange={(event) => updateDraft("eventDate", event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm" />
            </div>
            {draft.bleeding === "bleeding" && (
              <>
                <div>
                  <label htmlFor="cycle-end" className="text-xs font-medium text-slate-700">End date <span className="font-normal text-slate-500">(optional)</span></label>
                  <input id="cycle-end" type="date" value={draft.endDate} onChange={(event) => updateDraft("endDate", event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm" />
                </div>
                <div className="sm:col-span-2">
                <label htmlFor="cycle-flow" className="text-xs font-medium text-slate-700">Flow description <span className="font-normal text-slate-500">(optional)</span></label>
                <select id="cycle-flow" value={draft.flow} onChange={(event) => updateDraft("flow", event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm">
                  <option value="">Not recorded</option><option value="light">Light</option><option value="moderate">Moderate</option><option value="heavy">Heavy</option><option value="variable">Variable</option>
                </select>
                </div>
              </>
            )}
          </div>

          <fieldset>
            <legend className="text-xs font-semibold text-slate-800">Optional symptoms</legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {SYMPTOM_OPTIONS.map((symptom) => (
                <button key={symptom} type="button" aria-pressed={draft.symptoms.includes(symptom)} onClick={() => toggleSymptom(symptom)} className={cn("qed-focus rounded-full border px-3 py-1.5 text-xs transition-colors", draft.symptoms.includes(symptom) ? "border-violet-300 bg-violet-100 text-violet-950" : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100")}>
                  {symptom}
                </button>
              ))}
            </div>
          </fieldset>

          <div>
            <label htmlFor="cycle-pain" className="text-xs font-medium text-slate-700">Pain score <span className="font-normal text-slate-500">(optional, self-reported 0–10)</span></label>
            <input id="cycle-pain" type="number" min="0" max="10" step="1" value={draft.pain} onChange={(event) => updateDraft("pain", event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm" />
          </div>
          <div>
            <label htmlFor="cycle-notes" className="text-xs font-medium text-slate-700">Optional note</label>
            <textarea id="cycle-notes" value={draft.notes} onChange={(event) => updateDraft("notes", event.target.value)} rows={2} maxLength={500} className="mt-1.5 w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm" placeholder="Keep this non-identifying in the demo." />
          </div>
          {error && <p role="alert" className="text-xs font-medium text-rose-700">{error}</p>}
          <button type="submit" className="qed-focus inline-flex items-center gap-2 rounded-lg bg-slate-950 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-slate-800">
            <Plus className="h-4 w-4" aria-hidden /> Add journal entry
          </button>
        </form>

        <div aria-live="polite" className="min-h-40">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-slate-900">Journal entries</h4>
            <span className="text-[11px] text-slate-500">Temporary · this view only</span>
          </div>
          {entries.length > 0 && (
            <dl aria-label="Counts from entries in this session" className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {[
                ["Total entries", entries.length],
                ["Bleeding", bleedingRecords],
                ["No bleeding", noBleedingRecords],
                ["Unknown", unknownRecords],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <dt className="text-[10px] uppercase tracking-wide text-slate-500">{label}</dt>
                  <dd className="mt-1 text-lg font-semibold tabular-nums text-slate-900">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          {entries.length === 0 ? (
            <div className="mt-3 flex min-h-36 flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50/60 px-5 text-center">
              <p className="text-sm font-medium text-slate-700">No entries in this session</p>
              <p className="mt-1 max-w-sm text-xs leading-5 text-slate-500">Try the form with fictional information to preview how a participant could record their own history.</p>
            </div>
          ) : (
            <ul className="mt-3 space-y-2">
              {entries.map((entry) => (
                <li key={entry.id} className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold capitalize text-slate-900">{entry.bleeding === "none" ? "No bleeding recorded" : entry.bleeding === "unknown" ? "Bleeding status unknown" : "Bleeding recorded"}</p>
                      {entry.eventDate && <p className="mt-1 text-xs text-slate-600">{entry.bleeding === "bleeding" ? `Period start: ${entry.eventDate}` : `Observation date: ${entry.eventDate}`}{entry.endDate ? ` · through ${entry.endDate}` : ""}</p>}
                      <p className="mt-1 text-[10px] text-slate-400">Entered {new Date(entry.recordedAt).toLocaleString()}</p>
                      <p className="mt-2 text-xs leading-5 text-slate-600">{[entry.flow && `Flow: ${entry.flow}`, entry.symptoms.length && `Symptoms: ${entry.symptoms.join(", ")}`, entry.pain && `Pain: ${entry.pain}/10`, entry.notes].filter(Boolean).join(" · ") || "No optional details recorded."}</p>
                    </div>
                    <button type="button" aria-label="Remove journal entry" onClick={() => setEntries((current) => current.filter((item) => item.id !== entry.id))} className="qed-focus rounded-md p-2 text-slate-500 hover:bg-rose-50 hover:text-rose-700">
                      <Trash2 className="h-4 w-4" aria-hidden />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
