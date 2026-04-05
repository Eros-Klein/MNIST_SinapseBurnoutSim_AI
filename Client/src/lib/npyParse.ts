/** Parse a `.npy` file (v1.0 / v2.0) into a normalized 3D volume for the viewer. */

export type ParsedVolume = {
  /** Min–max normalized values in C order: index z * H * W + y * W + x */
  data: Float32Array;
  d: number;
  h: number;
  w: number;
};

const MAGIC = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);

function readHeaderLength(view: DataView, major: number, offset: number): { len: number; dataOffset: number } {
  if (major === 1) {
    const len = view.getUint16(offset, true);
    return { len, dataOffset: offset + 2 };
  }
  if (major === 2) {
    const len = view.getUint32(offset, true);
    return { len, dataOffset: offset + 4 };
  }
  throw new Error(`Unsupported NPY version major=${major}`);
}

function parseHeaderDict(headerText: string): { descr: string; fortran: boolean; shape: number[] } {
  const descrM = /'descr':\s*'([^']+)'/.exec(headerText);
  const fortM = /'fortran_order':\s*(True|False)/.exec(headerText);
  const shapeM = /'shape':\s*\(([^)]*)\)/.exec(headerText);
  if (!descrM || !shapeM) {
    throw new Error("Invalid NPY header (missing descr or shape).");
  }
  const descr = descrM[1];
  const fortran = fortM ? fortM[1] === "True" : false;
  const raw = shapeM[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number.parseInt(s, 10));
  if (raw.some((n) => !Number.isFinite(n))) {
    throw new Error("Invalid NPY shape.");
  }
  return { descr, fortran, shape: raw };
}

function descrBase(descr: string): string {
  return descr.replace(/^[=<>|]+/, "");
}

function fileBytesPerElement(descr: string): number {
  const base = descrBase(descr);
  if (base === "u1" || base === "i1" || base === "b1") return 1;
  if (base === "u2" || base === "i2") return 2;
  if (base === "i4" || base === "u4" || base === "f4") return 4;
  if (base === "f8" || base === "i8" || base === "u8") return 8;
  throw new Error(`Unsupported NPY dtype descr: ${descr}`);
}

function readValuesAsFloat32(buffer: ArrayBuffer, byteOffset: number, count: number, descr: string): Float32Array {
  const base = descrBase(descr);
  const dv = new DataView(buffer);

  const out = new Float32Array(count);

  if (base === "u1" || base === "i1" || base === "b1") {
    const u8 = new Uint8Array(buffer, byteOffset, count);
    for (let i = 0; i < count; i++) out[i] = u8[i];
    return out;
  }
  if (base === "u2") {
    for (let i = 0; i < count; i++) out[i] = dv.getUint16(byteOffset + i * 2, true);
    return out;
  }
  if (base === "i2") {
    for (let i = 0; i < count; i++) out[i] = dv.getInt16(byteOffset + i * 2, true);
    return out;
  }
  if (base === "i4") {
    for (let i = 0; i < count; i++) out[i] = dv.getInt32(byteOffset + i * 4, true);
    return out;
  }
  if (base === "u4") {
    for (let i = 0; i < count; i++) out[i] = dv.getUint32(byteOffset + i * 4, true);
    return out;
  }
  if (base === "f4") {
    if (byteOffset % 4 !== 0) {
      for (let i = 0; i < count; i++) out[i] = dv.getFloat32(byteOffset + i * 4, true);
      return out;
    }
    return new Float32Array(buffer, byteOffset, count);
  }
  if (base === "f8") {
    for (let i = 0; i < count; i++) out[i] = dv.getFloat64(byteOffset + i * 8, true);
    return out;
  }

  throw new Error(`Unsupported NPY dtype descr: ${descr}`);
}

function minMaxNormalize(src: Float32Array): Float32Array {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const out = new Float32Array(src.length);
  if (!(max > min)) return out;
  const inv = 1 / (max - min);
  for (let i = 0; i < src.length; i++) out[i] = (src[i] - min) * inv;
  return out;
}

function toVolume3d(shape: number[], raw: Float32Array, fortran: boolean): ParsedVolume {
  if (fortran) {
    throw new Error("Fortran-ordered NPY volumes are not supported in the viewer.");
  }

  let d: number;
  let h: number;
  let w: number;
  let values: Float32Array;

  if (shape.length === 3) {
    [d, h, w] = shape;
    values = raw;
  } else if (shape.length === 4) {
    const [n, sd, sh, sw] = shape;
    d = sd;
    h = sh;
    w = sw;
    const volSize = d * h * w;
    if (raw.length < n * volSize) {
      throw new Error("NPY data size does not match shape.");
    }
    values = Float32Array.from(raw.subarray(0, volSize));
  } else {
    throw new Error(`Expected 3D or 4D array, got shape (${shape.join(", ")}).`);
  }

  const expected = d * h * w;
  if (values.length !== expected) {
    throw new Error(`Volume size mismatch: ${values.length} vs ${expected}.`);
  }

  return {
    data: minMaxNormalize(values),
    d,
    h,
    w,
  };
}

export function parseNpyBuffer(buffer: ArrayBuffer): ParsedVolume {
  if (buffer.byteLength < 10) {
    throw new Error("File too small to be a valid .npy file.");
  }

  const head = new Uint8Array(buffer, 0, 6);
  for (let i = 0; i < 6; i++) {
    if (head[i] !== MAGIC[i]) {
      throw new Error("Not a NumPy .npy file (bad magic).");
    }
  }

  const view = new DataView(buffer);
  const major = view.getUint8(6);
  const minor = view.getUint8(7);
  if (major !== 1 && major !== 2) {
    throw new Error(`Unsupported NPY format version ${major}.${minor}.`);
  }

  const { len: headerLen, dataOffset: headerStart } = readHeaderLength(view, major, 8);
  const headerEnd = headerStart + headerLen;
  if (headerEnd > buffer.byteLength) {
    throw new Error("Corrupt NPY header.");
  }

  const headerBytes = new Uint8Array(buffer, headerStart, headerLen);
  const headerText = new TextDecoder("latin1").decode(headerBytes);
  const { descr, fortran, shape } = parseHeaderDict(headerText);

  let dataByteOffset = headerEnd;
  const pad = (16 - (dataByteOffset % 16)) % 16;
  dataByteOffset += pad;

  const { d, h, w } = (() => {
    if (shape.length === 3) {
      return { d: shape[0], h: shape[1], w: shape[2] };
    }
    if (shape.length === 4) {
      return { d: shape[1], h: shape[2], w: shape[3] };
    }
    throw new Error(`Expected 3D or 4D array, got shape (${shape.join(", ")}).`);
  })();

  const count = shape.length === 4 ? shape[0] * d * h * w : d * h * w;
  const payloadBytes = count * fileBytesPerElement(descr);
  if (dataByteOffset + payloadBytes > buffer.byteLength) {
    throw new Error("NPY file truncated.");
  }
  const raw = readValuesAsFloat32(buffer, dataByteOffset, count, descr);

  return toVolume3d(shape, raw, fortran);
}
