"use client";

import { useMemo } from "react";
import * as THREE from "three";
import { baseballSeamSamples } from "@/lib/seam";

export function Baseball({ radius }: { radius: number }) {
  const seam = useMemo(() => {
    const points = baseballSeamSamples(140).map(
      ([x, y, z]) => new THREE.Vector3(x * radius, z * radius, y * radius),
    );
    const curve = new THREE.CatmullRomCurve3(points, true, "catmullrom", 0.15);
    return new THREE.TubeGeometry(curve, 180, radius * 0.038, 5, true);
  }, [radius]);

  const stitches = useMemo(() => {
    const points = baseballSeamSamples(108).map(
      ([x, y, z]) => new THREE.Vector3(x * radius, z * radius, y * radius),
    );
    const geom = new THREE.BoxGeometry(radius * 0.09, radius * 0.028, radius * 0.02);
    const mesh = new THREE.InstancedMesh(geom, new THREE.MeshStandardMaterial({ color: "#9d2418", roughness: 0.55 }), points.length);
    const dummy = new THREE.Object3D();
    const xAxis = new THREE.Vector3(1, 0, 0);
    points.forEach((point, index) => {
      const next = points[(index + 1) % points.length];
      const tangent = next.clone().sub(point);
      if (tangent.lengthSq() < 1e-8) tangent.set(1, 0, 0);
      tangent.normalize();
      dummy.position.copy(point).addScaledVector(point.clone().normalize(), radius * 0.012);
      dummy.quaternion.setFromUnitVectors(xAxis, tangent);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    return mesh;
  }, [radius]);

  return (
    <group>
      <mesh>
        <sphereGeometry args={[radius * 0.98, 40, 28]} />
        <meshStandardMaterial color="#f4f0e6" roughness={0.58} metalness={0.04} />
      </mesh>
      <mesh geometry={seam}>
        <meshStandardMaterial color="#c4372a" roughness={0.42} />
      </mesh>
      <primitive object={stitches} />
    </group>
  );
}
