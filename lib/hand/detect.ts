import { handFromLandmarks, type Point3 } from "./profile";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL = "/mediapipe/hand_landmarker.task";

export type DetectedHand = ReturnType<typeof handFromLandmarks> & {
  landmarks: Point3[];
  imageWidth: number;
  imageHeight: number;
};

let landmarkerPromise: Promise<{
  detect: (image: HTMLImageElement) => {
    landmarks?: { x: number; y: number; z: number }[][];
    worldLandmarks?: { x: number; y: number; z: number }[][];
  };
  close: () => void;
}> | null = null;

async function landmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
      const files = await FilesetResolver.forVisionTasks(WASM);
      return HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
        runningMode: "IMAGE",
        numHands: 1,
      });
    })();
  }
  return landmarkerPromise;
}

export async function detectHand(image: HTMLImageElement, referenceLengthCm: number): Promise<DetectedHand> {
  const marker = await landmarker();
  const result = marker.detect(image);
  const world = result.worldLandmarks?.[0];
  const imagePoints = result.landmarks?.[0];
  if (!world || world.length < 21 || !imagePoints) {
    throw new Error("手を検出できませんでした。手のひらをカメラに向け、指が全部写る写真にしてください。");
  }
  const measured = handFromLandmarks(world, referenceLengthCm);
  return {
    ...measured,
    landmarks: imagePoints.map((point) => ({ x: point.x, y: point.y, z: point.z })),
    imageWidth: image.naturalWidth,
    imageHeight: image.naturalHeight,
  };
}
