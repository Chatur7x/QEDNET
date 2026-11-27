/**
 * Body Atlas — schematic navigation aid for anatomy and dataset associations.
 * The 3D scene is client-only; an accessible 2D table mirrors every selection
 * and serves browsers without WebGL or reduced-power environments.
 */
"use client";

import {
  Component,
  useMemo,
  useState,
  useSyncExternalStore,
  type ComponentType,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import dynamic from "next/dynamic";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Box, Info, List, Search, ShieldAlert, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { ORGANS, type Organ } from "../atlas/anatomy-data";
import {
  DomainDot,
  ResearchBadge,
  SectionCard,
  TransitionIn,
} from "../ui-bits";

type SelectHandler = Dispatch<SetStateAction<string | null>>;

/** Local contract for the independently authored scene module. */
type AtlasSceneProps = {
  selectedId: string | null;
  onSelect: SelectHandler;
  layers: Record<string, boolean>;
  /** Scene expects 0 = intact, 1 = full cross-section. */
  clip: number;
  reduceMotion: boolean;
};

const organs = ORGANS;

const LAYERS = [
  "skin",
  "skeleton",
  "cardiovascular",
  "respiratory",
  "nervous",
  "digestive",
  "renal",
  "endocrine",
] as const;

const systems = [...new Set(organs.map((organ) => organ.system))];

/** System color is used only as a meaning-bearing marker inside the atlas. */
const SYSTEM_COLOR: Record<string, string> = {
  nervous: "var(--qed-nervous)",
  cardiovascular: "var(--qed-cardio)",
  respiratory: "var(--qed-resp)",
  digestive: "var(--qed-digestive)",
  renal: "var(--qed-renal)",
  endocrine: "var(--qed-endocrine)",
  skeletal: "var(--qed-skeletal)",
  integumentary: "var(--qed-skeletal)",
};

function SystemDot({ system }: { system: string }) {
  return (
    <span
      aria-hidden
      className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
      style={{ backgroundColor: SYSTEM_COLOR[system] ?? "var(--qed-research)" }}
    />
  );
}

let webglSupport: boolean | undefined;

function getWebGLSupport(): boolean {
  if (webglSupport === undefined) {
    try {
      const canvas = document.createElement("canvas");
      webglSupport = Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
    } catch {
      webglSupport = false;
    }
  }
  return webglSupport;
}

const subscribeNoop = () => () => {};
const serverWebGLSupport = () => null;

function StageLoading() {
  return (
    <div className="flex h-full min-h-[420px] items-center justify-center pt-10 text-sm text-stone-500">
      Loading 3D atlas…
    </div>
  );
}

const AnatomyCanvas = dynamic(
  () => import("../atlas/AnatomyCanvas").then((m) => m.AnatomyCanvas),
  { ssr: false, loading: StageLoading }
) as unknown as ComponentType<AtlasSceneProps>;

/** Keeps a failed WebGL scene from taking down the dashboard. */
class SceneBoundary extends Component<
  { children: ReactNode; onError: () => void },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch() {
    this.props.onError();
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function CoverageChip({ datasetAssociation, system }: { datasetAssociation: Organ["datasetAssociation"]; system?: string }) {
  return (
    <span className="qed-num inline-flex items-center gap-1.5 rounded-full border border-stone-200 bg-stone-50 px-2 py-0.5 text-sm text-stone-600">
      {system ? <SystemDot system={system} /> : <DomainDot domain="research" />}
      {datasetAssociation === "associated" ? "dataset associated" : "planned"}
    </span>
  );
}

/** Mirrors the 3D selection in a form screen readers and keyboards can use. */
function OrganTable({
  items,
  selectedId,
  onSelect,
  className,
}: {
  items: readonly Organ[];
  selectedId: string | null;
  onSelect: SelectHandler;
  className?: string;
}) {
  return (
    <Table className={className}>
      <caption className="sr-only">
        Anatomical regions. Selecting a row also selects the matching region in the 3D atlas.
      </caption>
      <TableHeader>
        <TableRow className="border-stone-200">
          <TableHead className="text-sm">Region</TableHead>
          <TableHead className="text-sm">System</TableHead>
          <TableHead className="text-sm">Dataset association</TableHead>
          <TableHead className="text-sm">Related dataset</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((organ) => (
          <TableRow
            key={organ.id}
            className={cn(
              "border-stone-100",
              organ.id === selectedId && "bg-stone-100/70"
            )}
          >
            <TableCell>
              <button
                type="button"
                onClick={() => onSelect(organ.id)}
                aria-pressed={organ.id === selectedId}
                className="qed-focus rounded text-left text-sm font-medium text-stone-800 underline-offset-2 hover:underline"
              >
                {organ.name}
              </button>
            </TableCell>
            <TableCell className="text-sm capitalize text-stone-500">{organ.system}</TableCell>
            <TableCell><CoverageChip datasetAssociation={organ.datasetAssociation} system={organ.system} /></TableCell>
            <TableCell className="qed-num text-sm text-stone-500">
              {organ.linkedDataset ?? "—"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export function BodyAtlasView() {
  const reduceMotion = useReducedMotion() ?? false;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [layers, setLayers] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(LAYERS.map((layer) => [layer, true]))
  );
  const [clip, setClip] = useState(0);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  const [sceneError, setSceneError] = useState(false);
  const [sceneKey, setSceneKey] = useState(0);
  const webgl = useSyncExternalStore(subscribeNoop, getWebGLSupport, serverWebGLSupport);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return organs.filter(
      (organ) =>
        layers[organ.layer] !== false &&
        (!q ||
          `${organ.name} ${organ.system} ${organ.description}`
            .toLowerCase()
            .includes(q))
    );
  }, [layers, query]);

  const grouped = systems
    .map((system) => ({
      system,
      items: filtered.filter((organ) => organ.system === system),
    }))
    .filter((group) => group.items.length > 0);

  const selected = organs.find((organ) => organ.id === selectedId) ?? null;

  const ledger = systems.map((system) => {
    const regions = organs.filter((organ) => organ.system === system);
    return {
      system,
      associated: regions.filter((organ) => organ.datasetAssociation === "associated").length,
      planned: regions.filter((organ) => organ.datasetAssociation === "planned").length,
    };
  });

  function toggleLayer(layer: string) {
    const next = { ...layers, [layer]: !layers[layer] };
    setLayers(next);
    if (!next[layer] && organs.find((organ) => organ.id === selectedId)?.layer === layer) {
      setSelectedId(null);
    }
  }

  function use3d() {
    setSceneError(false);
    setSceneKey((key) => key + 1);
    setMode("3d");
  }

  const show2d = mode === "2d" || webgl === false || sceneError;

  return (
    <TransitionIn className="qed-atlas space-y-4">
      <div
        role="note"
        className="flex items-start gap-2 rounded-md border border-stone-300 bg-white px-3 py-2 text-sm leading-relaxed text-stone-600"
      >
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-stone-500" aria-hidden />
        <span>
          Schematic anatomy for research navigation only. Dataset associations indicate a related study topic,
          not organ measurements, imaging, validation, diagnosis, or clinical guidance. Do not enter personal health data.
        </span>
      </div>

      <SectionCard
        title="Body atlas"
        subtitle="schematic body systems · dataset associations · no patient data"
        right={<ResearchBadge compact />}
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
          {/* -------------------------------------------------- stage */}
          <div className="min-w-0">
            <div className="relative h-[520px] min-h-[420px] overflow-hidden rounded-lg border border-stone-200 qed-atlas-stage qed-rule">
              <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-center justify-between border-b border-stone-200/60 bg-white/70 px-3 py-2 backdrop-blur-sm">
                <span className="qed-label">Anatomy · front view</span>
                <span className="qed-num text-sm text-stone-500">
                  {filtered.length} regions · {filtered.filter((organ) => organ.datasetAssociation === "associated").length} dataset associations
                </span>
              </div>
              {webgl === null ? (
                <StageLoading />
              ) : show2d ? (
                <div className="flex h-full flex-col">
                  {(webgl === false || sceneError) && (
                    <div
                      role="status"
                      className="flex items-center gap-2 border-b border-stone-200 bg-stone-100/70 px-3 py-2 text-sm text-stone-600"
                    >
                      <Info className="h-4 w-4 shrink-0" aria-hidden />
                      {sceneError
                        ? "The 3D scene could not start. Showing the accessible 2D atlas instead."
                        : "WebGL is unavailable in this browser. Showing the accessible 2D atlas instead."}
                    </div>
                  )}
                  <div className="min-h-0 flex-1 overflow-auto qed-scroll">
                    <OrganTable items={filtered} selectedId={selectedId} onSelect={setSelectedId} />
                    {filtered.length === 0 && (
                      <p className="px-3 py-4 text-sm text-stone-500">
                        No regions match the current layers or search.
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <SceneBoundary
                  key={sceneKey}
                  onError={() => {
                    setSceneError(true);
                    setMode("2d");
                  }}
                >
                  <AnatomyCanvas
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                    layers={layers}
                    clip={clip / 100}
                    reduceMotion={reduceMotion}
                  />
                </SceneBoundary>
              )}
            </div>

            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-sm text-stone-500">
              <span className="text-sm">Schematic 3D model · not medical imaging · associations are topic links only</span>
              <div className="inline-flex rounded-md border border-stone-200 p-0.5" role="group" aria-label="Atlas display mode">
                <button
                  type="button"
                  onClick={use3d}
                  disabled={webgl === false}
                  aria-pressed={!show2d}
                  className="qed-focus inline-flex items-center gap-1.5 rounded px-2 py-1 text-sm text-stone-600 disabled:cursor-not-allowed disabled:opacity-40 aria-pressed:bg-stone-900 aria-pressed:text-white"
                >
                  <Box className="h-4 w-4" aria-hidden /> 3D
                </button>
                <button
                  type="button"
                  onClick={() => setMode("2d")}
                  aria-pressed={show2d}
                  className="qed-focus inline-flex items-center gap-1.5 rounded px-2 py-1 text-sm text-stone-600 aria-pressed:bg-stone-900 aria-pressed:text-white"
                >
                  <List className="h-4 w-4" aria-hidden /> 2D list
                </button>
              </div>
            </div>

            {!show2d && (
              <OrganTable
                className="sr-only"
                items={filtered}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            )}
          </div>

          {/* -------------------------------------------------- rail */}
          <div className="space-y-4 border-t border-stone-200 pt-4 xl:border-l xl:border-t-0 xl:pl-4 xl:pt-0">
            <section aria-labelledby="atlas-layers-heading">
              <h3 id="atlas-layers-heading" className="qed-label mb-2 border-b border-stone-200 pb-1.5">
                Layers
              </h3>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                {LAYERS.map((layer) => (
                  <div key={layer} className="flex items-center justify-between gap-2">
                    <label
                      htmlFor={`atlas-layer-${layer}`}
                      className="flex min-w-0 items-center gap-1.5 text-sm capitalize text-stone-600"
                    >
                      <SystemDot system={layer === "skin" ? "integumentary" : layer} />
                      <span className="truncate">{layer}</span>
                    </label>
                    <Switch
                      id={`atlas-layer-${layer}`}
                      aria-label={`Show ${layer} layer`}
                      checked={layers[layer] !== false}
                      onCheckedChange={() => toggleLayer(layer)}
                    />
                  </div>
                ))}
              </div>
            </section>

            <section aria-labelledby="atlas-clip-heading" className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 id="atlas-clip-heading" className="qed-label">
                  Cross-section depth
                </h3>
                <span className="qed-num text-sm text-stone-500">{clip}%</span>
              </div>
              <Slider
                aria-label="Cross-section depth"
                min={0}
                max={100}
                step={1}
                value={[clip]}
                onValueChange={([value]) => setClip(value)}
              />
            </section>

            <section aria-labelledby="atlas-search-heading" className="space-y-2">
              <h3 id="atlas-search-heading" className="qed-label">
                Find a region
              </h3>
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-stone-400" aria-hidden />
                <Input
                  type="search"
                  aria-label="Search anatomical regions"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="search regions or systems"
                  className="qed-focus h-9 pl-8 pr-8 text-sm"
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    aria-label="Clear region search"
                    className="qed-focus absolute right-2 top-2 rounded p-0.5 text-stone-400 hover:text-stone-700"
                  >
                    <X className="h-4 w-4" aria-hidden />
                  </button>
                )}
              </div>

              <div className="max-h-56 space-y-3 overflow-y-auto pr-1 qed-scroll" aria-label="Regions grouped by system">
                {grouped.length === 0 && (
                  <p className="text-sm text-stone-500">No regions match the current filters.</p>
                )}
                {grouped.map((group) => (
                  <div key={group.system}>
                    <h4 className="qed-label mb-1 flex items-center gap-1.5">
                      <SystemDot system={group.system} />
                      {group.system} · {group.items.length}
                    </h4>
                    <ul className="space-y-0.5">
                      {group.items.map((organ) => (
                        <li key={organ.id}>
                          <button
                            type="button"
                            onClick={() => setSelectedId(organ.id)}
                            aria-pressed={organ.id === selectedId}
                            className={cn(
                              "qed-focus flex w-full items-center justify-between gap-2 rounded px-2 py-1 text-left text-sm transition-colors",
                              organ.id === selectedId
                                ? "bg-stone-900 text-white"
                                : "text-stone-600 hover:bg-stone-100"
                            )}
                          >
                            <span className="truncate">{organ.name}</span>
                            <span className={cn(
                              "shrink-0 text-sm",
                              organ.id === selectedId ? "text-stone-300" : "text-stone-400"
                            )}>
                              {organ.datasetAssociation === "associated" ? "dataset associated" : "planned"}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </section>

            <section aria-labelledby="atlas-detail-heading" className="rounded-md border border-stone-200 bg-white p-3">
              <div className="mb-2 flex items-start justify-between gap-2">
                <h3 id="atlas-detail-heading" className="qed-label">
                  Selected region
                </h3>
                {selected && (
                  <button
                    type="button"
                    onClick={() => setSelectedId(null)}
                    aria-label="Clear selected region"
                    className="qed-focus rounded p-0.5 text-stone-400 hover:text-stone-700"
                  >
                    <X className="h-4 w-4" aria-hidden />
                  </button>
                )}
              </div>
              <AnimatePresence mode="wait" initial={false}>
                {selected ? (
                  <motion.div
                    key={selected.id}
                    initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.18, ease: "easeOut" }}
                    className="space-y-2 border-l-2 pl-2"
                    style={{ borderLeftColor: SYSTEM_COLOR[selected.system] ?? "var(--qed-research)" }}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-medium text-stone-800">{selected.name}</span>
                      <span className="text-sm capitalize text-stone-500">{selected.system}</span>
                      <CoverageChip datasetAssociation={selected.datasetAssociation} system={selected.system} />
                    </div>
                    <p className="text-sm leading-relaxed text-stone-600">{selected.description}</p>
                    <div className="flex items-center justify-between gap-2 border-t border-stone-100 pt-2 text-sm">
                      <span className="text-stone-500">Related dataset topic</span>
                      <span className="qed-num text-right text-stone-700">{selected.linkedDataset ?? "none"}</span>
                    </div>
                    <p className="flex items-start gap-1.5 rounded border border-stone-200 bg-stone-50 px-2 py-1.5 text-sm leading-relaxed text-stone-500">
                      <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                      Dataset association only. The cohort does not establish measurements for this anatomical region.
                    </p>
                  </motion.div>
                ) : (
                  <motion.p
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-sm leading-relaxed text-stone-500"
                  >
                    Select a region to inspect its anatomy description and dataset association.
                  </motion.p>
                )}
              </AnimatePresence>
            </section>

            <section aria-labelledby="atlas-ledger-heading">
              <h3 id="atlas-ledger-heading" className="qed-label mb-2 border-b border-stone-200 pb-1.5">
                Dataset associations by system
              </h3>
              <div className="overflow-x-auto qed-scroll">
                <Table>
                  <caption className="sr-only">
                    Schematic regions with an associated dataset topic and regions without a current association.
                  </caption>
                  <TableHeader>
                    <TableRow className="border-stone-200">
                      <TableHead className="text-sm">System</TableHead>
                      <TableHead className="qed-num text-right text-sm">Associated</TableHead>
                      <TableHead className="qed-num text-right text-sm">Planned</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {ledger.map((row) => (
                      <TableRow key={row.system} className="border-stone-100">
                        <TableCell className="text-sm capitalize text-stone-600">{row.system}</TableCell>
                        <TableCell className="qed-num text-right text-sm text-stone-700">
                          <div>{row.associated}</div>
                          <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-stone-100">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${Math.round(
                                  (row.associated / Math.max(1, row.associated + row.planned)) * 100
                                )}%`,
                                backgroundColor: SYSTEM_COLOR[row.system] ?? "var(--qed-research)",
                              }}
                            />
                          </div>
                        </TableCell>
                        <TableCell className="qed-num text-right text-sm text-stone-500">{row.planned}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </section>
          </div>
        </div>

        <p className="sr-only" aria-live="polite">
          {selected
            ? `Selected ${selected.name}, ${selected.system} system, ${selected.datasetAssociation === "associated" ? "dataset associated" : "planned"}.`
            : "No anatomical region selected."}
        </p>
      </SectionCard>
    </TransitionIn>
  );
}
