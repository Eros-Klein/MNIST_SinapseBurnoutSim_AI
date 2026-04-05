import * as THREE from "three";
import { NoToneMapping, SRGBColorSpace } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import type { ParsedVolume } from "./npyParse";

export type VolumeViewHandle = {
  dispose: () => void;
  setThreshold: (t: number) => void;
  /** Call after the host becomes visible (e.g. modal opened) so size/camera match layout. */
  relayout: () => void;
};

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/** Grayscale from normalized intensity [0,1] — matches the real voxel values after min–max scaling. */
function grayscaleRgb(t: number): [number, number, number] {
  const g = clamp(t, 0, 1);
  return [g, g, g];
}

function drawSliceCanvas(
  canvas: HTMLCanvasElement,
  getV: (a: number, b: number) => number,
  dimA: number,
  dimB: number,
  scale: number,
): void {
  const bw = dimA * scale;
  const bh = dimB * scale;
  canvas.width = bw;
  canvas.height = bh;
  canvas.style.width = `${bw}px`;
  canvas.style.height = `${bh}px`;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;

  const img = ctx.createImageData(dimA, dimB);
  let k = 0;
  for (let b = 0; b < dimB; b++) {
    for (let a = 0; a < dimA; a++) {
      const v = Math.round(clamp(getV(a, b), 0, 1) * 255);
      img.data[k++] = v;
      img.data[k++] = v;
      img.data[k++] = v;
      img.data[k++] = 255;
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = dimA;
  tmp.height = dimB;
  const tctx = tmp.getContext("2d");
  if (!tctx) return;
  tctx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, bw, bh);
  ctx.drawImage(tmp, 0, 0, bw, bh);
}

export function createVolumeView(
  host: HTMLElement,
  volume: ParsedVolume,
  sliceEls: {
    xy: HTMLCanvasElement;
    xz: HTMLCanvasElement;
    yz: HTMLCanvasElement;
    rangeZ: HTMLInputElement;
    rangeY: HTMLInputElement;
    rangeX: HTMLInputElement;
    valZ?: HTMLSpanElement;
    valY?: HTMLSpanElement;
    valX?: HTMLSpanElement;
  },
): VolumeViewHandle {
  const { data, d, h, w } = volume;
  let threshold = 0.07;

  sliceEls.rangeZ.max = String(d - 1);
  sliceEls.rangeY.max = String(h - 1);
  sliceEls.rangeX.max = String(w - 1);
  sliceEls.rangeZ.value = String(Math.floor(d / 2));
  sliceEls.rangeY.value = String(Math.floor(h / 2));
  sliceEls.rangeX.value = String(Math.floor(w / 2));

  const scene = new THREE.Scene();
  /* Dark neutral background; slightly above black so dim voxels stay visible. */
  scene.background = new THREE.Color(0x16161c);

  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 500);
  camera.position.set(42, 34, 46);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(typeof window !== "undefined" ? window.devicePixelRatio : 1, 2));
  /* Default ACES tone mapping crushes mid-gray vertex colors to black; disable for raw intensity view. */
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = NoToneMapping;
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  let points: THREE.Points | null = null;
  let geom: THREE.BufferGeometry | null = null;
  let mat: THREE.PointsMaterial | null = null;

  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const cz = (d - 1) / 2;

  function buildPoints() {
    if (points) {
      scene.remove(points);
      geom?.dispose();
      mat?.dispose();
      points = null;
      geom = null;
      mat = null;
    }

    const positions: number[] = [];
    const colors: number[] = [];

    for (let z = 0; z < d; z++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const t = data[z * h * w + y * w + x];
          if (t < threshold) continue;
          positions.push(x - cx, -(y - cy), -(z - cz));
          const [r, g, b] = grayscaleRgb(t);
          colors.push(r, g, b);
        }
      }
    }

    geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(positions), 3));
    geom.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(colors), 3));

    mat = new THREE.PointsMaterial({
      size: 1.35,
      vertexColors: true,
      sizeAttenuation: true,
      opacity: 1,
      depthWrite: true,
    });

    points = new THREE.Points(geom, mat);
    scene.add(points);
  }

  function drawSlices() {
    const zi = clamp(Number.parseInt(sliceEls.rangeZ.value, 10), 0, d - 1);
    const yi = clamp(Number.parseInt(sliceEls.rangeY.value, 10), 0, h - 1);
    const xi = clamp(Number.parseInt(sliceEls.rangeX.value, 10), 0, w - 1);

    if (sliceEls.valZ) sliceEls.valZ.textContent = String(zi);
    if (sliceEls.valY) sliceEls.valY.textContent = String(yi);
    if (sliceEls.valX) sliceEls.valX.textContent = String(xi);

    const scale = 5;
    drawSliceCanvas(sliceEls.xy, (x, y) => data[zi * h * w + y * w + x], w, h, scale);
    drawSliceCanvas(sliceEls.xz, (x, z) => data[z * h * w + yi * w + x], w, d, scale);
    drawSliceCanvas(sliceEls.yz, (y, z) => data[z * h * w + y * w + xi], h, d, scale);
  }

  buildPoints();
  drawSlices();

  let raf = 0;
  function tick() {
    raf = requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  function resize() {
    const rect = host.getBoundingClientRect();
    const width = Math.max(240, Math.floor(rect.width));
    const height = Math.max(260, Math.floor(rect.height));
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    drawSlices();
  }

  const ro = new ResizeObserver(() => resize());
  ro.observe(host);
  resize();

  const onSlice = () => drawSlices();
  sliceEls.rangeZ.addEventListener("input", onSlice);
  sliceEls.rangeY.addEventListener("input", onSlice);
  sliceEls.rangeX.addEventListener("input", onSlice);

  function dispose() {
    cancelAnimationFrame(raf);
    ro.disconnect();
    sliceEls.rangeZ.removeEventListener("input", onSlice);
    sliceEls.rangeY.removeEventListener("input", onSlice);
    sliceEls.rangeX.removeEventListener("input", onSlice);
    controls.dispose();
    if (points) {
      scene.remove(points);
      geom?.dispose();
      mat?.dispose();
    }
    renderer.dispose();
    host.removeChild(renderer.domElement);
  }

  return {
    dispose,
    setThreshold(t: number) {
      threshold = clamp(t, 0, 1);
      buildPoints();
    },
    relayout: resize,
  };
}
