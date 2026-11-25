"use client";

import { useEffect, useMemo, useRef, type RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { Baseball } from "./baseball";
import { BALL_RADIUS_M, RUBBER_TO_PLATE_M } from "@/lib/physics/constants";
import { sampleAtTime, tunnelPlaneY, type Sample } from "@/lib/physics/trajectory";
import type { Vec3 } from "@/lib/physics/vec";

export type ScenePitch = {
  id: string;
  name: string;
  color: string;
  samples: Sample[];
  spinless: Sample[];
  omega: Vec3;
  active: boolean;
};

export type CameraName = "diamond" | "catcher" | "pitcher" | "side" | "above";

const CAMERAS: Record<CameraName, { pos: [number, number, number]; target: [number, number, number] }> = {
  diamond: { pos: [6.6, 3.5, 5.4], target: [0, 1.15, 11] },
  catcher: { pos: [0.05, 1.45, 21.5], target: [0, 1.0, 14] },
  pitcher: { pos: [-0.7, 1.85, -1.15], target: [0, 0.85, 16] },
  side: { pos: [12.5, 2.1, 10], target: [0, 1.05, 10] },
  above: { pos: [0.1, 15.5, 10.2], target: [0, 0, 10] },
};

function toThree(p: { x: number; y: number; z: number }): [number, number, number] {
  return [p.x, p.z, p.y];
}

function Trail({ samples, color, radius }: { samples: Sample[]; color: string; radius: number }) {
  const geometry = useMemo(() => {
    if (samples.length < 2) return null;
    const points = samples.map((sample) => new THREE.Vector3(...toThree(sample.p)));
    const curve = new THREE.CatmullRomCurve3(points);
    return new THREE.TubeGeometry(curve, Math.min(140, points.length * 2), radius, 5, false);
  }, [samples, radius]);
  if (!geometry) return null;
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} roughness={0.4} metalness={0.08} />
    </mesh>
  );
}

function Ghost({ samples }: { samples: Sample[] }) {
  const line = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const arr = new Float32Array(samples.length * 3);
    samples.forEach((sample, index) => {
      const [x, y, z] = toThree(sample.p);
      arr[index * 3] = x;
      arr[index * 3 + 1] = y;
      arr[index * 3 + 2] = z;
    });
    geom.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return new THREE.Line(geom, new THREE.LineBasicMaterial({ color: "#9a937f", transparent: true, opacity: 0.55 }));
  }, [samples]);
  return <primitive object={line} />;
}

function FlyingBall({
  pitch,
  timeRef,
  radius,
}: {
  pitch: ScenePitch;
  timeRef: RefObject<number>;
  radius: number;
}) {
  const ref = useRef<THREE.Group>(null);
  const axis = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    const group = ref.current;
    if (!group) return;
    const sample = sampleAtTime(pitch.samples, timeRef.current ?? 0);
    if (!sample) return;
    group.position.set(sample.p.x, sample.p.z, sample.p.y);
    axis.set(pitch.omega.x, pitch.omega.z, pitch.omega.y);
    const spin = axis.length();
    if (spin > 1e-6) {
      axis.multiplyScalar(1 / spin);
      group.quaternion.setFromAxisAngle(axis, spin * (timeRef.current ?? 0));
    }
  });
  return (
    <group ref={ref}>
      <Baseball radius={radius} />
    </group>
  );
}

function SpinAxis({ pitch, timeRef }: { pitch: ScenePitch; timeRef: RefObject<number> }) {
  const ref = useRef<THREE.Group>(null);
  const dir = useMemo(() => new THREE.Vector3(), []);
  const up = useMemo(() => new THREE.Vector3(0, 1, 0), []);
  useFrame(() => {
    const group = ref.current;
    if (!group) return;
    const sample = sampleAtTime(pitch.samples, timeRef.current ?? 0);
    if (!sample) return;
    group.position.set(sample.p.x, sample.p.z, sample.p.y);
    dir.set(pitch.omega.x, pitch.omega.z, pitch.omega.y);
    if (dir.lengthSq() < 1e-8) return;
    dir.normalize();
    group.quaternion.setFromUnitVectors(up, dir);
  });
  return (
    <group ref={ref}>
      <mesh position={[0, 0.22, 0]}>
        <cylinderGeometry args={[0.008, 0.008, 0.36, 8]} />
        <meshBasicMaterial color="#e0b15a" />
      </mesh>
      <mesh position={[0, 0.42, 0]}>
        <coneGeometry args={[0.028, 0.07, 8]} />
        <meshBasicMaterial color="#e0b15a" />
      </mesh>
    </group>
  );
}

function Clock({
  timeRef,
  playingRef,
  rateRef,
  duration,
}: {
  timeRef: RefObject<number>;
  playingRef: RefObject<boolean>;
  rateRef: RefObject<number>;
  duration: number;
}) {
  const hold = useRef(0);
  useFrame((_, dt) => {
    if (!playingRef.current) return;
    if (hold.current > 0) {
      hold.current -= dt;
      if (hold.current <= 0) timeRef.current = 0;
      return;
    }
    timeRef.current = (timeRef.current ?? 0) + dt * (rateRef.current ?? 0.45);
    if ((timeRef.current ?? 0) >= duration) {
      timeRef.current = duration;
      hold.current = 0.6;
    }
  });
  return null;
}

function CameraRig({ camera }: { camera: CameraName }) {
  const { camera: cam, controls } = useThree();
  useEffect(() => {
    const shot = CAMERAS[camera];
    cam.position.set(...shot.pos);
    cam.lookAt(...shot.target);
    const orbit = controls as { target?: THREE.Vector3; update?: () => void } | null;
    orbit?.target?.set(...shot.target);
    orbit?.update?.();
  }, [camera, cam, controls]);
  return null;
}

function HomePlate() {
  const geometry = useMemo(() => {
    const shape = new THREE.Shape();
    const w = 0.2159;
    shape.moveTo(-w, 0.12);
    shape.lineTo(w, 0.12);
    shape.lineTo(w, -0.08);
    shape.lineTo(0, -0.3);
    shape.lineTo(-w, -0.08);
    shape.closePath();
    return new THREE.ShapeGeometry(shape);
  }, []);
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.012, RUBBER_TO_PLATE_M]}>
      <meshStandardMaterial color="#f7f3ea" side={THREE.DoubleSide} />
    </mesh>
  );
}

function StrikeZone({ zone }: { zone: { bottomFt: number; topFt: number } }) {
  const geometry = useMemo(() => {
    const w = 0.4318 / 2;
    const z0 = zone.bottomFt * 0.3048;
    const z1 = zone.topFt * 0.3048;
    const y = RUBBER_TO_PLATE_M;
    const corners = [
      [-w, z0],
      [w, z0],
      [w, z1],
      [-w, z1],
    ];
    const arr = new Float32Array(24);
    corners.forEach(([x, z], index) => {
      const next = corners[(index + 1) % corners.length];
      arr[index * 6] = x;
      arr[index * 6 + 1] = z;
      arr[index * 6 + 2] = y;
      arr[index * 6 + 3] = next[0];
      arr[index * 6 + 4] = next[1];
      arr[index * 6 + 5] = y;
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return new THREE.Line(geom, new THREE.LineBasicMaterial({ color: "#f4f0e4" }));
  }, [zone.bottomFt, zone.topFt]);
  return <primitive object={geometry} />;
}

function Field({ zone }: { zone: { bottomFt: number; topFt: number } }) {
  const tunnel = tunnelPlaneY();
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 12]}>
        <planeGeometry args={[48, 36]} />
        <meshStandardMaterial color="#173226" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0.2]}>
        <circleGeometry args={[2.7, 48]} />
        <meshStandardMaterial color="#6d4c34" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, RUBBER_TO_PLATE_M + 0.4]}>
        <circleGeometry args={[1.7, 40]} />
        <meshStandardMaterial color="#5c4632" />
      </mesh>
      <mesh position={[0, 0.11, 0]}>
        <boxGeometry args={[0.61, 0.025, 0.15]} />
        <meshStandardMaterial color="#f4f0e6" />
      </mesh>
      <HomePlate />
      <StrikeZone zone={zone} />
      <mesh position={[0, 0.02, tunnel]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[8, 0.035]} />
        <meshBasicMaterial color="#e0b15a" transparent opacity={0.55} />
      </mesh>
      <mesh position={[0, 1.05, tunnel]}>
        <boxGeometry args={[1.6, 1.5, 0.01]} />
        <meshBasicMaterial color="#e0b15a" transparent opacity={0.08} />
      </mesh>
    </group>
  );
}

export function FieldScene({
  pitches,
  showGhost,
  enlargeBall,
  zone,
  camera,
  timeRef,
  playingRef,
  rateRef,
  duration,
}: {
  pitches: ScenePitch[];
  showGhost: boolean;
  enlargeBall: boolean;
  zone: { bottomFt: number; topFt: number };
  camera: CameraName;
  timeRef: RefObject<number>;
  playingRef: RefObject<boolean>;
  rateRef: RefObject<number>;
  duration: number;
}) {
  const active = pitches.find((pitch) => pitch.active);
  return (
    <Canvas camera={{ position: CAMERAS.diamond.pos, fov: 38 }} dpr={[1, 1.75]} gl={{ antialias: true }}>
      <color attach="background" args={["#10140f"]} />
      <fog attach="fog" args={["#10140f", 16, 38]} />
      <hemisphereLight args={["#d5e0c9", "#3a2a1c", 0.7]} />
      <directionalLight position={[7, 14, 4]} intensity={2.4} color="#fff1dc" />
      <directionalLight position={[-5, 5, 20]} intensity={0.55} color="#9eb4c4" />
      <Field zone={zone} />
      <CameraRig camera={camera} />
      <Clock timeRef={timeRef} playingRef={playingRef} rateRef={rateRef} duration={duration} />
      {pitches.map((pitch) => (
        <group key={pitch.id}>
          <Trail samples={pitch.samples} color={pitch.color} radius={pitch.active ? 0.042 : 0.026} />
          {showGhost && pitch.active ? <Ghost samples={pitch.spinless} /> : null}
          <FlyingBall
            pitch={pitch}
            timeRef={timeRef}
            radius={BALL_RADIUS_M * (enlargeBall ? (pitch.active ? 3.35 : 2.5) : 1)}
          />
          {pitch.samples.length > 0 ? (
            <mesh position={toThree(pitch.samples[pitch.samples.length - 1].p)}>
              <sphereGeometry args={[0.055, 16, 12]} />
              <meshBasicMaterial color={pitch.color} />
            </mesh>
          ) : null}
        </group>
      ))}
      {active ? <SpinAxis pitch={active} timeRef={timeRef} /> : null}
      <OrbitControls makeDefault enableDamping target={CAMERAS.diamond.target} maxPolarAngle={Math.PI / 2.05} />
    </Canvas>
  );
}
