export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const fmtTime = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : "–");
export const fmtDur = (s) => (s == null ? "–" : s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const body = isJson ? await res.json() : null;
  if (!res.ok) {
    const err = new Error(body?.title || `${res.status} ${res.statusText}`);
    err.code = body?.code; err.status = res.status; err.requestId = body?.request_id;
    throw err;
  }
  return body;
}
export const errText = (e) => `${e.message}${e.code ? ` (${e.code})` : ""}${e.requestId ? ` · request ${e.requestId}` : ""}`;

export function setBar(bar, frac, text) {
  bar.classList.remove("hidden");
  $("div", bar).style.width = `${Math.round(frac * 100)}%`;
  $("span", bar).textContent = text;
}

export function showErr(sel, text) { const el = $(sel); el.textContent = text; el.classList.remove("hidden"); }
