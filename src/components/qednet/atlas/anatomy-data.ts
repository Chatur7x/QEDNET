/**
 * QED-Net anatomy atlas — static, de-identified reference data.
 *
 * Coordinates: y up, origin near the pelvis; +z is anterior (towards the
 * viewer); x runs left/right on screen. `size` holds half-extents (radii)
 * so meshes can be built by scaling unit primitives.
 *
 * A dataset association is a navigation hint only. Bundled tabular cohorts do
 * not measure these anatomical structures. No patient data or clinical claims live here.
 */

export type AnatomySystem =
  | "nervous"
  | "cardiovascular"
  | "respiratory"
  | "digestive"
  | "renal"
  | "endocrine"
  | "skeletal"
  | "integumentary";

export type AnatomyLayer =
  | "skin"
  | "skeleton"
  | "cardiovascular"
  | "respiratory"
  | "nervous"
  | "digestive"
  | "renal"
  | "endocrine";

export type DatasetId = "breast_cancer" | "heart_disease" | "parkinsons" | "pima_diabetes";

export type OrganShape =
  | "ellipsoid"
  | "bean"
  | "lobes"
  | "cluster"
  | "ring"
  | "tube";

export interface Organ {
  id: string;
  name: string;
  system: AnatomySystem;
  description: string;
  position: [number, number, number];
  /** Half-extents [x, y, z] in world units. */
  size: [number, number, number];
  color: string;
  datasetAssociation: "associated" | "planned";
  linkedDataset?: DatasetId;
  layer: AnatomyLayer;
  shape: OrganShape;
}

export const LAYER_KEYS: readonly AnatomyLayer[] = [
  "skin",
  "skeleton",
  "cardiovascular",
  "respiratory",
  "nervous",
  "digestive",
  "renal",
  "endocrine",
];

export const DATASET_LABELS: Record<DatasetId, string> = {
  breast_cancer: "Breast Cancer",
  heart_disease: "Heart Disease",
  parkinsons: "Parkinson's",
  pima_diabetes: "Pima Diabetes",
};

export const ORGANS: Organ[] = [
  // ---- nervous -----------------------------------------------------------
  {
    id: "brain",
    name: "Brain",
    system: "nervous",
    description: "Reference landmark only. The Parkinson’s cohort contains voice measurements, not brain imaging or brain-specific measurements.",
    position: [0, 1.49, 0],
    size: [0.2, 0.22, 0.18],
    color: "#e7a2b6",
    datasetAssociation: "associated",
    linkedDataset: "parkinsons",
    layer: "nervous",
    shape: "lobes",
  },
  {
    id: "spinal_cord",
    name: "Spinal cord",
    system: "nervous",
    description: "Reference landmark only. No spinal cord measurements are included in the Parkinson’s voice cohort.",
    position: [0, 0.62, -0.1],
    size: [0.045, 0.42, 0.045],
    color: "#d9a8bb",
    datasetAssociation: "associated",
    linkedDataset: "parkinsons",
    layer: "nervous",
    shape: "tube",
  },

  // ---- cardiovascular ----------------------------------------------------
  {
    id: "heart",
    name: "Heart",
    system: "cardiovascular",
    description: "Reference landmark associated with the heart-disease tabular cohort; the cohort does not contain heart imaging.",
    position: [0.05, 0.72, 0.09],
    size: [0.13, 0.16, 0.11],
    color: "#c84660",
    datasetAssociation: "associated",
    linkedDataset: "heart_disease",
    layer: "cardiovascular",
    shape: "bean",
  },
  {
    id: "aorta",
    name: "Aorta",
    system: "cardiovascular",
    description: "Main arterial trunk leaving the heart; simplified reference vessel.",
    position: [-0.02, 0.66, -0.02],
    size: [0.045, 0.35, 0.045],
    color: "#b8364d",
    datasetAssociation: "planned",
    layer: "cardiovascular",
    shape: "tube",
  },
  {
    id: "vena_cava",
    name: "Vena cava",
    system: "cardiovascular",
    description: "Large venous return trunk; simplified reference vessel.",
    position: [0.09, 0.55, -0.02],
    size: [0.04, 0.3, 0.04],
    color: "#5d6fb0",
    datasetAssociation: "planned",
    layer: "cardiovascular",
    shape: "tube",
  },

  // ---- respiratory -------------------------------------------------------
  {
    id: "lung_left",
    name: "Left lung",
    system: "respiratory",
    description: "Two-lobe lung; no respiratory dataset is bundled yet.",
    position: [-0.21, 0.74, 0.01],
    size: [0.15, 0.26, 0.13],
    color: "#d992a1",
    datasetAssociation: "planned",
    layer: "respiratory",
    shape: "lobes",
  },
  {
    id: "lung_right",
    name: "Right lung",
    system: "respiratory",
    description: "Three-lobe lung; no respiratory dataset is bundled yet.",
    position: [0.21, 0.74, 0.01],
    size: [0.15, 0.26, 0.13],
    color: "#d992a1",
    datasetAssociation: "planned",
    layer: "respiratory",
    shape: "lobes",
  },

  // ---- digestive ---------------------------------------------------------
  {
    id: "liver",
    name: "Liver",
    system: "digestive",
    description: "Large abdominal organ beneath the diaphragm.",
    position: [-0.17, 0.43, 0.08],
    size: [0.24, 0.13, 0.14],
    color: "#8e5b54",
    datasetAssociation: "planned",
    layer: "digestive",
    shape: "bean",
  },
  {
    id: "stomach",
    name: "Stomach",
    system: "digestive",
    description: "J-shaped digestive chamber to the left of the midline.",
    position: [0.16, 0.41, 0.08],
    size: [0.13, 0.16, 0.1],
    color: "#d6a870",
    datasetAssociation: "planned",
    layer: "digestive",
    shape: "bean",
  },
  {
    id: "spleen",
    name: "Spleen",
    system: "digestive",
    description: "Lymphoid organ near the stomach; grouped with abdominal structures.",
    position: [0.28, 0.46, -0.03],
    size: [0.1, 0.13, 0.08],
    color: "#805071",
    datasetAssociation: "planned",
    layer: "digestive",
    shape: "bean",
  },
  {
    id: "small_intestine",
    name: "Small intestine",
    system: "digestive",
    description: "Coiled absorptive tube in the lower abdomen.",
    position: [0, 0.16, 0.09],
    size: [0.17, 0.11, 0.08],
    color: "#ca9175",
    datasetAssociation: "planned",
    layer: "digestive",
    shape: "cluster",
  },
  {
    id: "large_intestine",
    name: "Large intestine",
    system: "digestive",
    description: "Outer colonic frame around the small intestine.",
    position: [0, 0.16, 0.03],
    size: [0.23, 0.15, 0.1],
    color: "#b87e63",
    datasetAssociation: "planned",
    layer: "digestive",
    shape: "ring",
  },

  // ---- endocrine / metabolic ---------------------------------------------
  {
    id: "pancreas",
    name: "Pancreas",
    system: "endocrine",
    description: "Reference landmark associated with a diabetes tabular cohort; this is not pancreas imaging or a pancreas-specific measurement.",
    position: [0, 0.3, 0],
    size: [0.2, 0.055, 0.06],
    color: "#d9b56d",
    datasetAssociation: "associated",
    linkedDataset: "pima_diabetes",
    layer: "endocrine",
    shape: "bean",
  },

  // ---- renal -------------------------------------------------------------
  {
    id: "kidney_left",
    name: "Left kidney",
    system: "renal",
    description: "Paired filtering organ; no renal dataset is bundled yet.",
    position: [-0.18, 0.24, -0.07],
    size: [0.09, 0.13, 0.07],
    color: "#a75d52",
    datasetAssociation: "planned",
    layer: "renal",
    shape: "bean",
  },
  {
    id: "kidney_right",
    name: "Right kidney",
    system: "renal",
    description: "Paired filtering organ; no renal dataset is bundled yet.",
    position: [0.18, 0.24, -0.07],
    size: [0.09, 0.13, 0.07],
    color: "#a75d52",
    datasetAssociation: "planned",
    layer: "renal",
    shape: "bean",
  },
  {
    id: "bladder",
    name: "Bladder",
    system: "renal",
    description: "Pelvic urinary reservoir.",
    position: [0, 0.0, 0.06],
    size: [0.1, 0.1, 0.08],
    color: "#d0c37c",
    datasetAssociation: "planned",
    layer: "renal",
    shape: "ellipsoid",
  },

  // ---- integumentary / whole-body planned link ---------------------------
  {
    id: "breast_tissue",
    name: "Breast tissue",
    system: "integumentary",
    description: "Whole-body reference landmark; dataset link is planned, not a clinical mapping.",
    position: [0, 0.82, 0.18],
    size: [0.32, 0.1, 0.09],
    color: "#e0b6ac",
    datasetAssociation: "planned",
    linkedDataset: "breast_cancer",
    layer: "skin",
    shape: "lobes",
  },

  // ---- skeleton (metadata; meshes are drawn procedurally) ----------------
  {
    id: "skull",
    name: "Skull",
    system: "skeletal",
    description: "Cranial bone shell.",
    position: [0, 1.5, 0],
    size: [0.25, 0.29, 0.24],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "ellipsoid",
  },
  {
    id: "spine_bones",
    name: "Vertebral column",
    system: "skeletal",
    description: "Stacked vertebrae from the skull base to the pelvis.",
    position: [0, 0.62, -0.11],
    size: [0.05, 0.42, 0.05],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "tube",
  },
  {
    id: "ribcage",
    name: "Ribcage",
    system: "skeletal",
    description: "Thoracic bone cage protecting heart and lungs.",
    position: [0, 0.75, 0],
    size: [0.33, 0.28, 0.2],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "ring",
  },
  {
    id: "pelvis_bones",
    name: "Pelvis",
    system: "skeletal",
    description: "Pelvic girdle connecting the spine and legs.",
    position: [0, 0.23, 0],
    size: [0.31, 0.18, 0.21],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "ring",
  },
  {
    id: "arm_bones_left",
    name: "Left arm bones",
    system: "skeletal",
    description: "Humerus, radius and ulna (simplified).",
    position: [-0.58, 0.48, 0],
    size: [0.035, 0.5, 0.035],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "tube",
  },
  {
    id: "arm_bones_right",
    name: "Right arm bones",
    system: "skeletal",
    description: "Humerus, radius and ulna (simplified).",
    position: [0.58, 0.48, 0],
    size: [0.035, 0.5, 0.035],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "tube",
  },
  {
    id: "leg_bones_left",
    name: "Left leg bones",
    system: "skeletal",
    description: "Femur, tibia and fibula (simplified).",
    position: [-0.22, -0.64, 0],
    size: [0.045, 0.82, 0.045],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "tube",
  },
  {
    id: "leg_bones_right",
    name: "Right leg bones",
    system: "skeletal",
    description: "Femur, tibia and fibula (simplified).",
    position: [0.22, -0.64, 0],
    size: [0.045, 0.82, 0.045],
    color: "#e8e2d4",
    datasetAssociation: "planned",
    layer: "skeleton",
    shape: "tube",
  },
];

export const ORGAN_BY_ID: Record<string, Organ> = Object.fromEntries(
  ORGANS.map((organ) => [organ.id, organ]),
);

/** Pure visibility rule shared by the renderer and the self-check. */
export function isOrganVisible(organ: Organ, layers: Record<string, boolean>): boolean {
  return layers[organ.layer] !== false;
}
