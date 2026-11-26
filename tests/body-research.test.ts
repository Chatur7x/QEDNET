import { describe, expect, test } from "bun:test";
import { registeredDatasetsForSystem } from "../src/lib/body-research";

describe("body research evidence map", () => {
  test("only shows datasets actually registered for a body system", () => {
    expect(registeredDatasetsForSystem("oncology", ["breast_cancer"])).toEqual(["breast_cancer"]);
    expect(registeredDatasetsForSystem("reproductive", ["breast_cancer"])).toEqual([]);
    expect(registeredDatasetsForSystem("cardiovascular", ["heart-disease"])).toEqual([]);
  });

  test("matches optional datasets by their registry IDs, not by guesswork", () => {
    expect(registeredDatasetsForSystem("metabolic", ["pima_diabetes", "diabetes.csv"])).toEqual(["pima_diabetes"]);
    expect(registeredDatasetsForSystem("neurological", ["parkinsons"])).toEqual(["parkinsons"]);
  });
});
