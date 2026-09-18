/**
 * Shared QED-Net UI primitives: domain markers, stat cards, states.
 * Uses shadcn/ui foundations; custom pieces are kept minimal (Ponytail).
 */
"use client";

import { motion, useReducedMotion } from "framer-motion";
import { AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export type Domain = "quantum" | "classical" | "evaluation" | "explain" | "uncertainty" | "research";

export const DOMAIN_COLORS: Record<Domain, string> = {
  quantum: "var(--qed-quantum)",
  classical: "var(--qed-classical)",
  evaluation: "var(--qed-evaluation)",
  explain: "var(--qed-explain)",
  uncertainty: "var(--qed-uncertainty)",
  research: "var(--qed-research)",
};

/** Small colored dot marking a processing domain (subtle, non-hierarchical). */
export function DomainDot({ domain, className }: { domain: Domain; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", className)}
      style={{ backgroundColor: DOMAIN_COLORS[domain] }}
    />
  );
}

export function FamilyBadge({ family }: { family: "quantum" | "classical" | "unknown" }) {
  if (family === "quantum") {
    return (
      <Badge variant="outline" className="gap-1.5 border-violet-200 bg-violet-50 text-violet-800">
        <DomainDot domain="quantum" /> quantum
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="gap-1.5 border-teal-200 bg-teal-50 text-teal-800">
      <DomainDot domain="classical" /> classical
    </Badge>
  );
}

export function ResearchBadge({ compact }: { compact?: boolean }) {
  return (
    <Badge
      variant="outline"
      className="gap-1.5 border-stone-300 bg-stone-50 font-normal text-stone-600"
      title="Research and educational use only. Not a medical device. Not clinically validated. Not for diagnosis."
    >
      <DomainDot domain="research" />
      {compact ? "research only" : "research & educational use only"}
    </Badge>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    passed: "border-emerald-200 bg-emerald-50 text-emerald-800",
    failed: "border-rose-200 bg-rose-50 text-rose-800",
    pending: "border-stone-200 bg-stone-50 text-stone-600",
    running: "border-amber-200 bg-amber-50 text-amber-800",
    completed: "border-emerald-200 bg-emerald-50 text-emerald-800",
    queued: "border-stone-200 bg-stone-50 text-stone-600",
    cancelled: "border-stone-200 bg-stone-50 text-stone-500",
  };
  return (
    <Badge variant="outline" className={cn("font-normal", map[status] ?? map.pending)}>
      {status}
    </Badge>
  );
}

/** Big numeric stat with tabular figures. */
export function Stat({
  label,
  value,
  hint,
  domain,
}: {
  label: string;
  value: string;
  hint?: string;
  domain?: Domain;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wide text-stone-500">
        {domain && <DomainDot domain={domain} />}
        {label}
      </div>
      <div className="qed-num qed-display text-2xl font-semibold text-stone-900">{value}</div>
      {hint && <div className="text-xs text-stone-500">{hint}</div>}
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  right,
  children,
  className,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn("border-stone-200 bg-white/80 shadow-none", className)}>
      <CardHeader className="flex-row items-start justify-between space-y-0 pb-4">
        <div className="space-y-1">
          <CardTitle className="text-sm font-semibold tracking-tight text-stone-800">{title}</CardTitle>
          {subtitle && <p className="text-xs text-stone-500">{subtitle}</p>}
        </div>
        {right}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export function LoadingState({ label = "Loading experiment data…" }: { label?: string }) {
  return (
    <div className="flex min-h-[240px] items-center justify-center gap-3 rounded-lg border border-dashed border-stone-300 bg-white/50 text-sm text-stone-500">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="flex min-h-[160px] items-center justify-center gap-3 rounded-lg border border-rose-200 bg-rose-50/60 px-6 text-sm text-rose-800"
    >
      <AlertCircle className="h-4 w-4 shrink-0" aria-hidden />
      <span>{message}</span>
    </div>
  );
}

export function EmptyState({ message, hint }: { message: string; hint?: string }) {
  return (
    <div className="flex min-h-[240px] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-stone-300 bg-white/50 px-6 text-center">
      <p className="text-sm text-stone-600">{message}</p>
      {hint && <p className="max-w-md text-xs text-stone-400">{hint}</p>}
    </div>
  );
}

/** Spring presets per Apple Design skill: critically damped default,
 *  slight bounce only for momentum-driven interactions. */
export const SPRING = {
  smooth: { type: "spring" as const, bounce: 0, duration: 0.38 },
  soft: { type: "spring" as const, bounce: 0, duration: 0.5 },
};

/** View-transition wrapper honouring prefers-reduced-motion. */
export function TransitionIn({ children, className }: { children: React.ReactNode; className?: string }) {
  const reduce = useReducedMotion();
  if (reduce) {
    return (
      <motion.div
        className={className}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.18 }}
      >
        {children}
      </motion.div>
    );
  }
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={SPRING.smooth}
    >
      {children}
    </motion.div>
  );
}
