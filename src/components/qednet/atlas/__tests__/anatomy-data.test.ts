import { describe, expect, test } from "bun:test";

import {
  DATASET_LABELS,
  LAYER_KEYS,
  ORGANS,
  ORGAN_BY_ID,
  isOrganVisible,
  type DatasetId,
} from "../anatomy-data";

describe("anatomy atlas reference data", () => {
  test("ids are unique and lookup matches the list", () => {
    expect(new Set(ORGANS.map((organ) => organ.id)).size).toBe(ORGANS.length);
    for (const organ of ORGANS) {
      expect(ORGAN_BY_ID[organ.id]).toBe(organ);
    }
  });

  test("every layer key is valid and populated", () => {
    for (const organ of ORGANS) {
      expect(LAYER_KEYS).toContain(organ.layer);
    }
    for (const layer of LAYER_KEYS) {
      expect(ORGANS.some((organ) => organ.layer === layer)).toBe(true);
    }
  });

  test("linked datasets are only bundled ids", () => {
    for (const organ of ORGANS) {
      if (organ.linkedDataset) {
        expect(Object.keys(DATASET_LABELS)).toContain(organ.linkedDataset);
        expect(organ.linkedDataset satisfies DatasetId).toBe(organ.linkedDataset);
      }
    }
  });

  test("dataset associations match their mapped systems", () => {
    const expected: Record<string, DatasetId> = {
      heart: "heart_disease",
      brain: "parkinsons",
      spinal_cord: "parkinsons",
      pancreas: "pima_diabetes",
    };
    for (const [id, dataset] of Object.entries(expected)) {
      expect(ORGAN_BY_ID[id]?.datasetAssociation).toBe("associated");
      expect(ORGAN_BY_ID[id]?.linkedDataset).toBe(dataset);
    }
    const associated = ORGANS.filter((organ) => organ.datasetAssociation === "associated");
    expect(associated.every((organ) => organ.linkedDataset === expected[organ.id])).toBe(true);
  });

  test("layer filter hides disabled layers only", () => {
    const layers = Object.fromEntries(LAYER_KEYS.map((key) => [key, true]));
    expect(ORGANS.every((organ) => isOrganVisible(organ, layers))).toBe(true);
    expect(isOrganVisible(ORGAN_BY_ID.heart, { ...layers, cardiovascular: false })).toBe(false);
    expect(isOrganVisible(ORGAN_BY_ID.heart, { ...layers, respiratory: false })).toBe(true);
  });
});
