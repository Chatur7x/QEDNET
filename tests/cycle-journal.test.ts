import { describe, expect, test } from "bun:test";
import {
  cycleEntriesToCsv,
  validateCycleJournalDraft,
  type CycleJournalDraft,
  type CycleJournalEntry,
} from "../src/lib/cycle-journal";

const draft: CycleJournalDraft = {
  bleeding: "bleeding",
  eventDate: "2026-01-10",
  endDate: "2026-01-12",
  flow: "moderate",
  symptoms: ["Cramps"],
  pain: "4",
  notes: "fictional note",
};

describe("in-memory cycle journal helpers", () => {
  test("requires dates for bleeding and no-bleeding entries while allowing unknown dates", () => {
    expect(validateCycleJournalDraft({ ...draft, eventDate: "" })).toContain("date");
    expect(validateCycleJournalDraft({ ...draft, bleeding: "none", eventDate: "" })).toContain("date");
    expect(validateCycleJournalDraft({ ...draft, bleeding: "unknown", eventDate: "" })).toBeNull();
  });

  test("rejects reversed dates and invalid pain scores", () => {
    expect(validateCycleJournalDraft({ ...draft, endDate: "2026-01-09" })).toContain("on or after");
    expect(validateCycleJournalDraft({ ...draft, eventDate: "2026-02-29" })).toContain("valid calendar date");
    expect(validateCycleJournalDraft({ ...draft, endDate: "2026-02-30" })).toContain("valid end date");
    expect(validateCycleJournalDraft({ ...draft, pain: "11" })).toContain("0 to 10");
    expect(validateCycleJournalDraft({ ...draft, pain: "3.5" })).toContain("0 to 10");
  });

  test("exports exact journal fields with safe CSV escaping", () => {
    const entry: CycleJournalEntry = {
      id: "entry-id",
      recordedAt: "2026-01-13T00:00:00.000Z",
      ...draft,
      notes: 'fictional, "quoted" note',
    };
    expect(cycleEntriesToCsv([entry])).toBe([
      '"recorded_at","bleeding_status","event_date","end_date","flow","symptoms","pain_score","notes"',
      '"2026-01-13T00:00:00.000Z","bleeding","2026-01-10","2026-01-12","moderate","Cramps","4","fictional, ""quoted"" note"',
    ].join("\r\n"));
  });
});
