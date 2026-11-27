"use client";

/**
 * Client-only R3F stage for the anatomy atlas. The parent page is expected to
 * import this module with `next/dynamic` and `ssr: false` (WebGL is browser-only).
 */
import { useEffect, useRef, useState, type ComponentRef, type RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Grid, Html, OrbitControls } from "@react-three/drei";
import { useReducedMotion } from "framer-motion";
import * as THREE from "three";

import { DATASET_LABELS, ORGAN_BY_ID, type Organ } from "./anatomy-data";
import { HumanBody } from "./HumanBody";

export interface AnatomyCanvasProps {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  layers: Record<string, boolean>;
  clip: number;
  reduceMotion?: boolean;
}

type ControlsHandle = ComponentRef<typeof OrbitControls>;
type FocusGoal = { target: THREE.Vector3; distance: number; until: number };

/** Briefly eases the orbit target onto the selected organ; null resets. */
function CameraRig({
  controlsRef,
  focusId,
}: {
  controlsRef: RefObject<ControlsHandle | null>;
  focusId: string | null;
}) {
  const camera = useThree((state) => state.camera);
  const goal = useRef<FocusGoal | null>(null);

  useEffect(() => {
    const organ = focusId ? ORGAN_BY_ID[focusId] : undefined;
    goal.current = {
      target: organ ? new THREE.Vector3(...organ.position) : new THREE.Vector3(0, 0, 0),
      distance: organ ? 2.6 : 5.4,
      until: performance.now() + 750,
    };
  }, [focusId]);

  useFrame((_, delta) => {
    const active = goal.current;
    const controls = controlsRef.current;
    if (!active || !controls) return;
    if (performance.now() > active.until) {
      goal.current = null;
      return;
    }
    const t = 1 - Math.exp(-5 * delta);
    controls.target.lerp(active.target, t);
    const offset = camera.position.clone().sub(controls.target);
    if (offset.lengthSq() > 0) {
      const desired = controls.target.clone().addScaledVector(offset.normalize(), active.distance);
      camera.position.lerp(desired, t);
    }
    controls.update();
  });

  return null;
}

/** Thin screen-space callout anchored to an organ position. */
function OrganCallout({ organ, variant }: { organ: Organ; variant: "selected" | "hovered" }) {
  const strong = variant === "selected";
  return (
    <Html position={organ.position} zIndexRange={[50, 0]} style={{ pointerEvents: "none" }}>
      <div
        className={
          "relative -translate-y-1/2 translate-x-5 rounded-r border border-l-2 bg-white/95 px-2.5 py-1.5 shadow-sm backdrop-blur-sm " +
          (strong ? "border-l-rose-400" : "border-l-stone-300")
        }
      >
        <span
          aria-hidden
          className={
            "absolute -left-5 top-1/2 h-px w-5 " + (strong ? "bg-rose-300" : "bg-stone-300")
          }
        />
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-stone-800">{organ.name}</span>
          <span
            className={
              "rounded-sm px-1 py-px text-xs font-medium uppercase tracking-wide " +
              (organ.datasetAssociation === "associated"
              ? "bg-sky-50 text-sky-800"
                : "bg-stone-100 text-stone-500")
            }
          >
            {organ.datasetAssociation === "associated" ? "dataset associated" : "planned"}
          </span>
        </div>
        <div className="mt-0.5 text-xs uppercase tracking-wide text-stone-500">{organ.system}</div>
        {strong && (
          <p className="mt-1 max-w-[220px] whitespace-normal text-sm leading-snug text-stone-500">
            {organ.description}
          </p>
        )}
        {organ.linkedDataset && (
          <div className="mt-1 text-xs text-stone-500">
          associated cohort: {DATASET_LABELS[organ.linkedDataset]}
          </div>
        )}
      </div>
    </Html>
  );
}

function Scene({
  selectedId,
  onSelect,
  layers,
  clip,
  reduced,
}: AnatomyCanvasProps & { reduced: boolean }) {
  const controlsRef = useRef<ControlsHandle | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const selected = selectedId ? ORGAN_BY_ID[selectedId] : undefined;
  const hovered =
    hoveredId && hoveredId !== selectedId ? ORGAN_BY_ID[hoveredId] : undefined;

  return (
    <>
      <ambientLight intensity={0.72} />
      <hemisphereLight args={["#eef2ff", "#8a7f72", 0.45]} />
      <directionalLight position={[3, 5, 4]} intensity={1.2} />
      <directionalLight position={[-3, 1, -3]} intensity={0.35} color="#bcd7ff" />

      <HumanBody
        selectedId={selectedId}
        onSelect={onSelect}
        onHover={setHoveredId}
        layers={layers}
        clip={clip}
        reduceMotion={reduced}
      >
        {selected && <OrganCallout organ={selected} variant="selected" />}
        {hovered && <OrganCallout organ={hovered} variant="hovered" />}
      </HumanBody>

      <Grid
        position={[0, -1.76, 0]}
        args={[8, 8]}
        cellSize={0.35}
        cellThickness={0.5}
        cellColor="#d6d3d1"
        sectionSize={1.4}
        sectionThickness={0.9}
        sectionColor="#a8a29e"
        fadeDistance={11}
        fadeStrength={1.3}
        infiniteGrid
      />
      <ContactShadows
        position={[0, -1.75, 0]}
        scale={4.5}
        opacity={0.28}
        blur={2.6}
        far={3.2}
        resolution={512}
        color="#1c1917"
      />

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        enablePan={false}
        minDistance={1.6}
        maxDistance={9}
        maxPolarAngle={Math.PI * 0.92}
        target={[0, 0, 0]}
      />
      <CameraRig controlsRef={controlsRef} focusId={selectedId} />
    </>
  );
}

export function AnatomyCanvas({
  selectedId,
  onSelect,
  layers,
  clip,
  reduceMotion,
}: AnatomyCanvasProps) {
  const preferredReduced = useReducedMotion();
  const reduced = reduceMotion ?? preferredReduced ?? false;

  return (
    <Canvas
      camera={{ position: [0, 0.1, 5.4], fov: 38, near: 0.1, far: 100 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true;
      }}
      onPointerMissed={() => onSelect(null)}
      style={{ width: "100%", height: "100%", touchAction: "none" }}
    >
      <Scene
        selectedId={selectedId}
        onSelect={onSelect}
        layers={layers}
        clip={clip}
        reduced={reduced}
      />
    </Canvas>
  );
}

export default AnatomyCanvas;
