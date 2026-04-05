/** Backend base URL (no trailing slash). Set `PUBLIC_API_URL` in `.env` for production. */
export function getApiBase(): string {
  const raw = import.meta.env.PUBLIC_API_URL;
  if (typeof raw === "string" && raw.trim()) {
    return raw.replace(/\/$/, "");
  }
  /* In the browser, default to same hostname as the site (port 8000) so dev matches
   * e.g. http://localhost:4321 → http://localhost:8000 (avoids localhost vs 127.0.0.1 mix-ups). */
  if (typeof window !== "undefined" && window.location?.hostname) {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
}

export interface EvaluationResult {
  label: string;
  score: number;
  details?: Record<string, unknown> | null;
}

export interface ImageEntry {
  id: string;
  created_at: string;
  shape: [number, number, number];
  evaluation: EvaluationResult;
  npy_url: string;
}

export interface ImageListResponse {
  offset: number;
  limit: number;
  count: number;
  items: ImageEntry[];
}

export async function fetchImageList(offset = 0, limit = 50): Promise<ImageListResponse> {
  const base = getApiBase();
  const res = await fetch(`${base}/images?offset=${offset}&limit=${limit}`);
  if (!res.ok) {
    const err = await safeDetail(res);
    throw new Error(err ?? `Failed to load scans (${res.status})`);
  }
  return res.json() as Promise<ImageListResponse>;
}

export function volumeAbsoluteUrl(imageId: string): string {
  const base = getApiBase();
  return `${base}/images/${encodeURIComponent(imageId)}/npy`;
}

export async function fetchVolumeNpy(imageId: string): Promise<ArrayBuffer> {
  const res = await fetch(volumeAbsoluteUrl(imageId));
  if (!res.ok) {
    const err = await safeDetail(res);
    throw new Error(err ?? `Could not load volume (${res.status})`);
  }
  return res.arrayBuffer();
}

export async function uploadVolume(file: File): Promise<ImageEntry> {
  const base = getApiBase();
  const body = new FormData();
  body.append("image", file);
  const res = await fetch(`${base}/images`, {
    method: "POST",
    body,
  });
  if (!res.ok) {
    const err = await safeDetail(res);
    throw new Error(err ?? `Upload failed (${res.status})`);
  }
  return res.json() as Promise<ImageEntry>;
}

async function safeDetail(res: Response): Promise<string | null> {
  try {
    const data = (await res.json()) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((x) => (typeof x === "object" && x && "msg" in x ? String((x as { msg: string }).msg) : String(x))).join("; ");
    }
  } catch {
    /* ignore */
  }
  return null;
}
