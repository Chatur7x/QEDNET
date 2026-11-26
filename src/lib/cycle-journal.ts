export type BleedingStatus = "bleeding" | "none" | "unknown";

export interface CycleJournalEntry {
  id: string;
  recordedAt: string;
  bleeding: BleedingStatus;
  eventDate: string;
  endDate: string;
  flow: string;
  symptoms: string[];
  pain: string;
  notes: string;
}

export type CycleJournalDraft = Omit<CycleJournalEntry, "id" | "recordedAt">;

function isValidDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00.000Z`);
  return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

export function validateCycleJournalDraft(draft: CycleJournalDraft): string | null {
  if (draft.bleeding !== "unknown" && !draft.eventDate) {
    return "Add a date for this bleeding or no-bleeding entry.";
  }
  if (draft.eventDate && !isValidDate(draft.eventDate)) {
    return "Enter a valid calendar date.";
  }
  if (draft.endDate && !isValidDate(draft.endDate)) {
    return "Enter a valid end date.";
  }
  if (draft.eventDate && draft.endDate && draft.endDate < draft.eventDate) {
    return "The end date must be on or after the event date.";
  }
  if (draft.pain && (!/^([0-9]|10)$/.test(draft.pain))) {
    return "Pain score must be a whole number from 0 to 10.";
  }
  return null;
}

function csvCell(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

export function cycleEntriesToCsv(entries: readonly CycleJournalEntry[]): string {
  const header = ["recorded_at", "bleeding_status", "event_date", "end_date", "flow", "symptoms", "pain_score", "notes"];
  const rows = entries.map((entry) => [
    entry.recordedAt,
    entry.bleeding,
    entry.eventDate,
    entry.endDate,
    entry.flow,
    entry.symptoms.join("; "),
    entry.pain,
    entry.notes,
  ]);
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\r\n");
}
