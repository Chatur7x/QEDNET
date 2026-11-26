export const BODY_SYSTEMS = [
  {
    id: "neurological",
    label: "Brain, nerves & sleep",
    description: "Neurologic conditions, cognition, sleep, and symptom timelines.",
    datasetNames: ["parkinsons"],
    point: { top: "17%", left: "50%" },
  },
  {
    id: "cardiovascular",
    label: "Heart & circulation",
    description: "Cardiovascular measurements, longitudinal signals, and outcomes.",
    datasetNames: ["heart_disease"],
    point: { top: "34%", left: "50%" },
  },
  {
    id: "respiratory",
    label: "Lungs & breathing",
    description: "Spirometry, symptoms, environmental context, and device quality.",
    datasetNames: [],
    point: { top: "37%", left: "44%" },
  },
  {
    id: "metabolic",
    label: "Metabolic & endocrine",
    description: "Longitudinal metabolic measures, laboratory units, and treatment context.",
    datasetNames: ["pima_diabetes"],
    point: { top: "51%", left: "50%" },
  },
  {
    id: "digestive",
    label: "Digestive & liver",
    description: "Time-aware laboratory and symptom studies with care-process context.",
    datasetNames: [],
    point: { top: "60%", left: "55%" },
  },
  {
    id: "renal",
    label: "Kidney & urinary",
    description: "Repeated measurements, units, and carefully defined outcomes.",
    datasetNames: [],
    point: { top: "65%", left: "45%" },
  },
  {
    id: "musculoskeletal",
    label: "Musculoskeletal & recovery",
    description: "Function, rehabilitation, mobility, and patient-reported outcomes.",
    datasetNames: [],
    point: { top: "78%", left: "50%" },
  },
  {
    id: "reproductive",
    label: "Reproductive & life-course health",
    description: "Menstrual patterns, reproductive health, and life-course research.",
    datasetNames: [],
    point: { top: "72%", left: "50%" },
  },
  {
    id: "oncology",
    label: "Cancer & hematology",
    description: "Distinct screening, diagnostic-workup, prognosis, and treatment studies.",
    datasetNames: ["breast_cancer"],
    point: { top: "43%", left: "56%" },
  },
  {
    id: "immune",
    label: "Immune & infectious",
    description: "Modality-specific studies with explicit reference standards.",
    datasetNames: [],
    point: { top: "47%", left: "42%" },
  },
] as const;

export type BodySystemId = (typeof BODY_SYSTEMS)[number]["id"];

export function registeredDatasetsForSystem(
  systemId: BodySystemId,
  registeredNames: readonly string[],
): string[] {
  const system = BODY_SYSTEMS.find((item) => item.id === systemId);
  if (!system) return [];
  const available = new Set(registeredNames);
  return system.datasetNames.filter((name) => available.has(name));
}
