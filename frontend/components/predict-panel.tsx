"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type PredictResponse = {
  filename: string | null;
  anomaly_scores: number[];
  anomaly_score: number | null;
  anomaly_score_std?: number | null;
  monte_carlo_runs?: number;
  per_run_scores?: number[];
  anomaly_heatmap_png_base64?: string;
  heatmap_patch_grid_hw?: number[];
  heatmap_patch_size?: number;
  heatmap_note?: string;
  input_size_hw: number[];
  mask_ratio: number;
  device: string;
};

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((x) => (typeof x === "object" && x && "msg" in x ? String((x as { msg: string }).msg) : JSON.stringify(x)))
      .join("; ");
  }
  return "Eroare necunoscută";
}

/** Afișare lizibilă (nu notație științifică forțată). */
function formatScore(n: number): string {
  return new Intl.NumberFormat("ro-RO", {
    maximumSignificantDigits: 8,
    minimumFractionDigits: 0,
    maximumFractionDigits: 8,
  }).format(n);
}

function isImageFile(f: File): boolean {
  if (f.type.startsWith("image/")) return true;
  const ext = f.name.toLowerCase().split(".").pop() ?? "";
  return ["tif", "tiff", "png", "jpg", "jpeg", "webp", "bmp", "gif"].includes(ext);
}

export function PredictPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      setPreviewFailed(false);
      return;
    }
    setPreviewFailed(false);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFile = useCallback((f: File | null) => {
    setResult(null);
    setError(null);
    setFile(f);
  }, []);

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_BASE}/predict_mean`, {
        method: "POST",
        body,
      });
      const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
      if (!res.ok) {
        const detail = data.detail;
        throw new Error(formatDetail(detail));
      }
      setResult(data as unknown as PredictResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Nu s-a putut contacta API-ul.");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && isImageFile(f)) onFile(f);
    else setError("Trage aici un fișier imagine (PNG, JPEG, TIFF, …).");
  }, [onFile]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8">
      <header className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-widest text-sky-400/90">
          GenVAD Live
        </p>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50 sm:text-3xl">
          Detectare anomalii din imagine
        </h1>
        <p className="text-sm leading-relaxed text-zinc-400">
          Scor mediu + <strong className="text-zinc-300">hartă spațială</strong> a erorii de reconstrucție pe patch
          (media peste mai multe rulări MAE). Backend:{" "}
          <code className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-200">
            {API_BASE}
          </code>
        </p>
      </header>

      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className="group relative cursor-pointer rounded-2xl border border-dashed border-zinc-600 bg-zinc-900/50 px-6 py-14 text-center transition hover:border-sky-500/60 hover:bg-zinc-900"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f && isImageFile(f)) onFile(f);
            else if (f) setError("Selectează un fișier imagine.");
            e.target.value = "";
          }}
        />
        <div className="pointer-events-none space-y-2">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/30">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="text-sm font-medium text-zinc-200">
            Click sau trage imaginea aici
          </p>
          <p className="text-xs text-zinc-500">PNG, JPEG, WebP, TIFF … (previzualizarea TIFF poate lipsi în browser)</p>
        </div>
      </div>

      {previewUrl && file && (
        <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/40">
          <div className="border-b border-zinc-800 px-4 py-3">
            <p className="truncate text-sm text-zinc-300">
              <span className="text-zinc-500">Fișier:</span> {file.name}
            </p>
          </div>
          <div className="flex min-h-[12rem] justify-center bg-zinc-950/50 p-4">
            {previewFailed ? (
              <div className="flex max-w-md flex-col items-center justify-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900/80 px-6 py-10 text-center">
                <p className="text-sm text-zinc-300">
                  Browserul nu afișează previzualizare pentru acest format (frecvent la{" "}
                  <span className="font-mono text-sky-300">.tif</span> / <span className="font-mono text-sky-300">.tiff</span>
                  ).
                </p>
                <p className="text-xs text-zinc-500">
                  Fișierul e încărcat corect — poți rula <strong className="text-zinc-400">Predict</strong> la fel ca pentru PNG/JPEG.
                </p>
              </div>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={previewUrl}
                alt="Previzualizare"
                className="max-h-72 max-w-full rounded-lg object-contain shadow-lg"
                onError={() => setPreviewFailed(true)}
              />
            )}
          </div>
          <div className="flex flex-wrap gap-3 border-t border-zinc-800 p-4">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                void predict();
              }}
              disabled={loading}
              className="inline-flex items-center justify-center rounded-xl bg-sky-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-sky-900/30 transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Predict + hartă…" : "Predict"}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onFile(null);
              }}
              className="rounded-xl border border-zinc-700 px-4 py-2.5 text-sm text-zinc-300 transition hover:bg-zinc-800"
            >
              Șterge
            </button>
          </div>
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-red-900/60 bg-red-950/40 px-4 py-3 text-sm text-red-200"
        >
          {error}
        </div>
      )}

      {result && (
        <section className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
            Rezultate
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-xl bg-zinc-950/60 p-4 ring-1 ring-zinc-800">
              <p className="text-xs text-zinc-500">
                Scor mediu (eroare reconstrucție)
                {result.monte_carlo_runs != null && (
                  <span className="text-zinc-600"> — {result.monte_carlo_runs} rulări</span>
                )}
              </p>
              <p className="mt-1 font-mono text-3xl font-semibold tabular-nums text-sky-400">
                {result.anomaly_score != null ? formatScore(result.anomaly_score) : "—"}
              </p>
              {result.anomaly_score_std != null && result.anomaly_score_std > 0 && (
                <p className="mt-1 text-xs text-zinc-500">
                  Abatere standard peste rulări:{" "}
                  <span className="font-mono text-zinc-300">{formatScore(result.anomaly_score_std)}</span>
                </p>
              )}
              {result.anomaly_score != null && (
                <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                  Compară între frame-uri sau setează un prag pe date normale; valoarea absolută depinde de model
                  și normalizare.
                </p>
              )}
            </div>
            <dl className="space-y-2 rounded-xl bg-zinc-950/60 p-4 text-sm ring-1 ring-zinc-800">
              <div className="flex justify-between gap-2">
                <dt className="text-zinc-500">Dimensiune intrare (H×W)</dt>
                <dd className="font-mono text-zinc-200">
                  {result.input_size_hw?.join(" × ") ?? "—"}
                </dd>
              </div>
              <div className="flex flex-col gap-1 sm:flex-row sm:justify-between sm:gap-2">
                <dt className="shrink-0 text-zinc-500">
                  mask_ratio{" "}
                  <span className="block font-normal normal-case text-zinc-600 sm:inline sm:pl-1">
                    (fracțiune mascată în MAE, ex. 0,5 ≈ 50% ascuns)
                  </span>
                </dt>
                <dd className="font-mono text-zinc-200">{formatScore(result.mask_ratio)}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-zinc-500">Device</dt>
                <dd className="truncate font-mono text-xs text-zinc-300">{result.device}</dd>
              </div>
            </dl>
          </div>

          {result.anomaly_heatmap_png_base64 && (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                Hartă eroare reconstrucție (tonuri deschise = eroare mai mare pe patch)
              </h3>
              <div
                className={`grid gap-4 ${previewUrl && !previewFailed ? "sm:grid-cols-2" : ""}`}
              >
                {previewUrl && !previewFailed && (
                  <div className="space-y-2">
                    <p className="text-xs text-zinc-500">Imagine intrare</p>
                    <div className="flex justify-center rounded-xl border border-zinc-800 bg-zinc-950/50 p-2">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={previewUrl}
                        alt="Intrare"
                        className="max-h-64 max-w-full rounded-lg object-contain"
                      />
                    </div>
                  </div>
                )}
                <div className="space-y-2">
                  <p className="text-xs text-zinc-500">
                    Patch-uri {result.heatmap_patch_grid_hw?.join("×") ?? "—"}
                    {result.heatmap_patch_size != null && (
                      <span className="text-zinc-600"> (patch {result.heatmap_patch_size}px)</span>
                    )}
                  </p>
                  <div className="flex justify-center rounded-xl border border-zinc-800 bg-zinc-950/50 p-2">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.anomaly_heatmap_png_base64}`}
                      alt="Hartă eroare reconstrucție"
                      className="max-h-64 max-w-full rounded-lg object-contain"
                    />
                  </div>
                </div>
              </div>
              {result.heatmap_note && (
                <p className="text-xs leading-relaxed text-zinc-500">{result.heatmap_note}</p>
              )}
            </div>
          )}

          {result.per_run_scores && result.per_run_scores.length > 0 && (
            <details className="group rounded-xl border border-zinc-800 bg-zinc-950/40">
              <summary className="cursor-pointer list-none px-4 py-3 text-sm text-zinc-400 marker:content-none [&::-webkit-details-marker]:hidden">
                <span className="inline-flex items-center gap-2">
                  <span className="text-zinc-300 group-open:text-sky-400">
                    Scoruri pe fiecare rulare ({result.per_run_scores.length})
                  </span>
                  <svg className="h-4 w-4 transition group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </span>
              </summary>
              <div className="max-h-48 overflow-auto border-t border-zinc-800 p-3 font-mono text-xs text-zinc-400">
                <pre className="whitespace-pre-wrap break-all">
                  {result.per_run_scores.map((s, i) => `${i + 1}: ${formatScore(s)}`).join("\n")}
                </pre>
              </div>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
