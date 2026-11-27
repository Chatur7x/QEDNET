"use client";

/**
 * Procedural human body: translucent skin shell, skeleton and data-driven
 * organs. Everything is generated in code — no external assets or patient data.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useCursor } from "@react-three/drei";
import { useReducedMotion } from "framer-motion";
import * as THREE from "three";

import {
  ORGAN_BY_ID,
  ORGANS,
  isOrganVisible,
  type AnatomyLayer,
  type Organ,
} from "./anatomy-data";

export interface HumanBodyProps {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onHover?: (id: string | null) => void;
  layers: Record<string, boolean>;
  /** 0 = intact, 1 = maximum anterior cross-section. */
  clip: number;
  skinOpacity?: number;
  /** Explicit override; otherwise prefers-reduced-motion is used. */
  reduceMotion?: boolean;
  /** Rendered inside the (optionally idling) body group, e.g. 3D callouts. */
  children?: ReactNode;
}

type OrganHandlers = {
  onClick: (event: ThreeEvent<MouseEvent>) => void;
  onPointerOver: (event: ThreeEvent<PointerEvent>) => void;
  onPointerOut: (event: ThreeEvent<PointerEvent>) => void;
};

const UP = new THREE.Vector3(0, 1, 0);

const ARM_L: [[number, number, number], [number, number, number], [number, number, number]] = [
  [-0.46, 0.96, 0],
  [-0.62, 0.5, 0.01],
  [-0.7, 0.08, 0.02],
];
const ARM_R: typeof ARM_L = [
  [0.46, 0.96, 0],
  [0.62, 0.5, 0.01],
  [0.7, 0.08, 0.02],
];
const LEG_L: typeof ARM_L = [
  [-0.18, 0.23, 0],
  [-0.22, -0.62, 0],
  [-0.22, -1.5, 0],
];
const LEG_R: typeof ARM_L = [
  [0.18, 0.23, 0],
  [0.22, -0.62, 0],
  [0.22, -1.5, 0],
];

/** Capsule spanning two points, used for both skin limbs and long bones. */
function Bone({
  from,
  to,
  radius,
  color,
  clippingPlanes,
  handlers,
  opacity = 1,
}: {
  from: [number, number, number];
  to: [number, number, number];
  radius: number;
  color: string;
  clippingPlanes: THREE.Plane[];
  handlers?: OrganHandlers;
  opacity?: number;
}) {
  const { position, quaternion, length } = useMemo(() => {
    const a = new THREE.Vector3(...from);
    const b = new THREE.Vector3(...to);
    const delta = b.clone().sub(a);
    const total = delta.length();
    return {
      position: a.clone().add(b).multiplyScalar(0.5),
      quaternion: new THREE.Quaternion().setFromUnitVectors(UP, delta.normalize()),
      length: total,
    };
  }, [from, to]);

  return (
    <mesh position={position} quaternion={quaternion} {...handlers}>
      <capsuleGeometry args={[radius, Math.max(length - radius * 2, 0.01), 4, 10]} />
      <meshStandardMaterial
        color={color}
        roughness={0.72}
        metalness={0.02}
        transparent={opacity < 1}
        opacity={opacity}
        clippingPlanes={clippingPlanes}
      />
    </mesh>
  );
}

/** Shared tissue look with a subtle emissive lift for the selected organ. */
function TissueMaterial({
  organ,
  selected,
  clippingPlanes,
  opacity = 0.94,
}: {
  organ: Organ;
  selected: boolean;
  clippingPlanes: THREE.Plane[];
  opacity?: number;
}) {
  return (
    <meshStandardMaterial
      color={organ.color}
      emissive={selected ? organ.color : "#000000"}
      emissiveIntensity={selected ? 0.7 : 0}
      roughness={0.6}
      metalness={0.04}
      transparent
      opacity={selected ? Math.min(1, opacity + 0.06) : opacity}
      clippingPlanes={clippingPlanes}
      side={THREE.DoubleSide}
    />
  );
}

/** Data-driven organ mesh; multi-part shapes stay as a small group of primitives. */
function OrganMesh({
  organ,
  selected,
  clippingPlanes,
  handlers,
}: {
  organ: Organ;
  selected: boolean;
  clippingPlanes: THREE.Plane[];
  handlers: OrganHandlers;
}) {
  const [sx, sy, sz] = organ.size;
  const material = (
    <TissueMaterial organ={organ} selected={selected} clippingPlanes={clippingPlanes} />
  );

  if (organ.shape === "lobes") {
    return (
      <group position={organ.position} {...handlers}>
        {[-1, 1].map((side) => (
          <mesh key={side} position={[side * sx * 0.44, 0, 0]} scale={[sx * 0.62, sy, sz]}>
            <sphereGeometry args={[1, 20, 14]} />
            {material}
          </mesh>
        ))}
      </group>
    );
  }

  if (organ.shape === "cluster") {
    const offsets: [number, number, number][] = [
      [-0.5, 0.2, 0],
      [0.4, 0.35, 0.25],
      [-0.2, -0.4, -0.2],
      [0.5, -0.25, 0.1],
    ];
    return (
      <group position={organ.position} {...handlers}>
        {offsets.map((offset, index) => (
          <mesh
            key={index}
            position={[offset[0] * sx, offset[1] * sy, offset[2] * sz]}
            scale={[sx * 0.5, sy * 0.62, sz * 0.62]}
          >
            <sphereGeometry args={[1, 16, 12]} />
            {material}
          </mesh>
        ))}
      </group>
    );
  }

  if (organ.shape === "ring") {
    return (
      <group position={organ.position} {...handlers}>
        <mesh scale={[sx / 0.6, sy / 0.6, sz / 0.6]}>
          <torusGeometry args={[0.5, 0.1, 8, 30]} />
          {material}
        </mesh>
      </group>
    );
  }

  if (organ.shape === "tube") {
    return (
      <mesh position={organ.position} scale={[1, 1, sz / sx]} {...handlers}>
        <cylinderGeometry args={[sx, sx, sy * 2, 14]} />
        {material}
      </mesh>
    );
  }

  // ellipsoid / bean: a single scaled sphere, bean tilted slightly.
  return (
    <mesh
      position={organ.position}
      scale={[sx, sy, sz]}
      rotation={organ.shape === "bean" ? [0, 0, 0.22] : [0, 0, 0]}
      {...handlers}
    >
      <sphereGeometry args={[1, 22, 16]} />
      {material}
    </mesh>
  );
}

function SkinShell({
  opacity,
  clippingPlanes,
}: {
  opacity: number;
  clippingPlanes: THREE.Plane[];
}) {
  const material = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color("#d8b5a6"),
        roughness: 0.82,
        metalness: 0,
        transparent: true,
        opacity,
        depthWrite: false,
        clippingPlanes,
        side: THREE.FrontSide,
      }),
    [opacity, clippingPlanes],
  );

  useEffect(() => () => material.dispose(), [material]);

  const torsoGeometry = useMemo(() => {
    const profile = [
      new THREE.Vector2(0.3, 0.22),
      new THREE.Vector2(0.37, 0.42),
      new THREE.Vector2(0.39, 0.62),
      new THREE.Vector2(0.36, 0.84),
      new THREE.Vector2(0.42, 1.0),
      new THREE.Vector2(0.2, 1.08),
    ];
    const geometry = new THREE.LatheGeometry(profile, 32);
    geometry.computeVertexNormals();
    return geometry;
  }, []);

  return (
    <group>
      <mesh position={[0, 1.5, 0]} scale={[0.25, 0.3, 0.24]} material={material}>
        <sphereGeometry args={[1, 24, 18]} />
      </mesh>
      {/* subtle facial landmarks: enough for a human silhouette, not medical anatomy */}
      <mesh position={[0, 1.35, 0.12]} scale={[0.16, 0.07, 0.15]} material={material}>
        <sphereGeometry args={[1, 20, 12]} />
      </mesh>
      <mesh position={[0, 1.46, 0.235]} rotation={[Math.PI / 2, 0, 0]} scale={[0.042, 0.07, 0.042]} material={material}>
        <coneGeometry args={[1, 1.6, 12]} />
      </mesh>
      {[-0.245, 0.245].map((x) => (
        <mesh key={`ear-${x}`} position={[x, 1.5, 0]} scale={[0.035, 0.075, 0.05]} material={material}>
          <sphereGeometry args={[1, 12, 10]} />
        </mesh>
      ))}
      <mesh position={[0, 1.16, 0]} material={material}>
        <cylinderGeometry args={[0.11, 0.13, 0.2, 16]} />
      </mesh>
      <mesh geometry={torsoGeometry} scale={[1, 1, 0.62]} material={material} />
      <mesh position={[0, 0.22, 0]} scale={[0.34, 0.22, 0.24]} material={material}>
        <sphereGeometry args={[1, 24, 16]} />
      </mesh>

      {[ARM_L, ARM_R].map((points, index) => (
        <group key={`skin-arm-${index}`}>
          <Bone from={points[0]} to={points[1]} radius={0.085} color="#d8b5a6" clippingPlanes={clippingPlanes} opacity={opacity} />
          <Bone from={points[1]} to={points[2]} radius={0.068} color="#d8b5a6" clippingPlanes={clippingPlanes} opacity={opacity} />
          <mesh position={[points[2][0], points[2][1] - 0.07, points[2][2]]} scale={[0.07, 0.1, 0.05]} material={material}>
            <sphereGeometry args={[1, 14, 10]} />
          </mesh>
        </group>
      ))}
      {[LEG_L, LEG_R].map((points, index) => (
        <group key={`skin-leg-${index}`}>
          <Bone from={points[0]} to={points[1]} radius={0.14} color="#d8b5a6" clippingPlanes={clippingPlanes} opacity={opacity} />
          <Bone from={points[1]} to={points[2]} radius={0.1} color="#d8b5a6" clippingPlanes={clippingPlanes} opacity={opacity} />
          <mesh position={[points[2][0], points[2][1] - 0.09, points[2][2] + 0.1]} scale={[0.1, 0.07, 0.18]} material={material}>
            <sphereGeometry args={[1, 14, 10]} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

/** Procedural skeleton. Each part carries the id of its atlas entry. */
function Skeleton({
  selectedId,
  clippingPlanes,
  handlersFor,
}: {
  selectedId: string | null;
  clippingPlanes: THREE.Plane[];
  handlersFor: (id: string) => OrganHandlers;
}) {
  const skull = ORGAN_BY_ID.skull;
  const spine = ORGAN_BY_ID.spine_bones;
  const ribs = ORGAN_BY_ID.ribcage;
  const pelvis = ORGAN_BY_ID.pelvis_bones;

  const tissue = (organ: Organ) => (
    <TissueMaterial organ={organ} selected={selectedId === organ.id} clippingPlanes={clippingPlanes} />
  );

  return (
    <group>
      <mesh position={skull.position} scale={[skull.size[0], skull.size[1], skull.size[2]]} {...handlersFor(skull.id)}>
        <sphereGeometry args={[1, 22, 16]} />
        {tissue(skull)}
      </mesh>

      <group {...handlersFor(spine.id)}>
        {Array.from({ length: 12 }, (_, index) => (
          <mesh key={index} position={[0, 0.24 + index * 0.07, -0.11]} scale={[0.05, 0.029, 0.045]}>
            <sphereGeometry args={[1, 12, 10]} />
            {tissue(spine)}
          </mesh>
        ))}
      </group>

      <group {...handlersFor(ribs.id)}>
        {Array.from({ length: 7 }, (_, index) => {
          const t = index / 6;
          const y = 0.52 + t * 0.45;
          const radius = 0.33 * (1 - 0.32 * Math.abs(t - 0.42));
          return (
            <mesh key={index} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[1, 0.62, 1]}>
              <torusGeometry args={[radius, 0.012, 6, 32]} />
              {tissue(ribs)}
            </mesh>
          );
        })}
      </group>

      <mesh
        position={pelvis.position}
        rotation={[Math.PI / 2, 0, 0]}
        scale={[1, 0.78, 1]}
        {...handlersFor(pelvis.id)}
      >
        <torusGeometry args={[0.27, 0.028, 8, 32]} />
        {tissue(pelvis)}
      </mesh>

      {[
        { points: ARM_L, id: "arm_bones_left" },
        { points: ARM_R, id: "arm_bones_right" },
      ].map(({ points, id }) => (
        <group key={id}>
          <Bone from={points[0]} to={points[1]} radius={0.022} color={ORGAN_BY_ID[id].color} clippingPlanes={clippingPlanes} handlers={handlersFor(id)} />
          <Bone from={points[1]} to={points[2]} radius={0.017} color={ORGAN_BY_ID[id].color} clippingPlanes={clippingPlanes} handlers={handlersFor(id)} />
        </group>
      ))}
      {[
        { points: LEG_L, id: "leg_bones_left" },
        { points: LEG_R, id: "leg_bones_right" },
      ].map(({ points, id }) => (
        <group key={id}>
          <Bone from={points[0]} to={points[1]} radius={0.035} color={ORGAN_BY_ID[id].color} clippingPlanes={clippingPlanes} handlers={handlersFor(id)} />
          <Bone from={points[1]} to={points[2]} radius={0.028} color={ORGAN_BY_ID[id].color} clippingPlanes={clippingPlanes} handlers={handlersFor(id)} />
        </group>
      ))}
    </group>
  );
}

export function HumanBody({
  selectedId,
  onSelect,
  onHover,
  layers,
  clip,
  skinOpacity = 0.16,
  reduceMotion,
  children,
}: HumanBodyProps) {
  const groupRef = useRef<THREE.Group>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const preferredReduced = useReducedMotion();
  const allowIdle = !(reduceMotion ?? preferredReduced ?? false);

  useCursor(hoverId !== null);

  // Construct the plane from `clip`; do not mutate a memoised value.
  const clippingPlanes = useMemo(() => {
    if (clip <= 0) return [];
    const constant = 0.55 - THREE.MathUtils.clamp(clip, 0, 1) * 0.8;
    return [new THREE.Plane(new THREE.Vector3(0, 0, -1), constant)];
  }, [clip]);

  useFrame((state, delta) => {
    const group = groupRef.current;
    if (!group) return;
    if (!allowIdle || selectedId !== null || hoverId !== null) {
      group.rotation.y = THREE.MathUtils.damp(group.rotation.y, 0, 6, delta);
      return;
    }
    group.rotation.y = Math.sin(state.clock.elapsedTime * 0.25) * 0.12;
  });

  const handlersFor = (id: string): OrganHandlers => ({
    onClick: (event) => {
      event.stopPropagation();
      onSelect(selectedId === id ? null : id);
    },
    onPointerOver: (event) => {
      event.stopPropagation();
      setHoverId(id);
      onHover?.(id);
    },
    onPointerOut: () => {
      setHoverId((current) => (current === id ? null : current));
      onHover?.(null);
    },
  });

  const visible = (layer: AnatomyLayer) => layers[layer] !== false;

  return (
    <group ref={groupRef}>
      {visible("skin") && <SkinShell opacity={skinOpacity} clippingPlanes={clippingPlanes} />}
      {visible("skeleton") && (
        <Skeleton selectedId={selectedId} clippingPlanes={clippingPlanes} handlersFor={handlersFor} />
      )}
      {ORGANS.filter(
        (organ) => isOrganVisible(organ, layers) && organ.layer !== "skeleton",
      ).map((organ) => (
        <OrganMesh
          key={organ.id}
          organ={organ}
          selected={selectedId === organ.id}
          clippingPlanes={clippingPlanes}
          handlers={handlersFor(organ.id)}
        />
      ))}
      {children}
    </group>
  );
}

export default HumanBody;
