import { $, $$, api } from "./dom.js";
import { loadCameras } from "./cameras.js";
import { loadJobs } from "./history.js";
import "./video.js";

$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${t.dataset.tab}`));
  if (t.dataset.tab === "cameras") loadCameras();
  if (t.dataset.tab === "history") loadJobs();
}));

async function pollReady() {
  const el = $("#ready");
  try {
    const r = await fetch("/readyz").then((x) => x.json());
    const bad = Object.entries(r.checks).filter(([, v]) => !v.ok).map(([k]) => k);
    el.className = `status ${r.ready ? "ok" : "bad"}`;
    $(".label", el).textContent = r.ready ? "ready" : `not ready: ${bad.join(", ")}`;
  } catch {
    el.className = "status bad"; $(".label", el).textContent = "service unreachable";
  }
}
pollReady(); setInterval(pollReady, 10000);
api("/version").then((v) => {
  $("#foot").textContent = `model ${v.model.name}:${v.model.version} · service ${v.version}`;
  $("#limits").textContent = `MP4 / MOV / MKV · up to ${Math.round(v.limits.max_upload_bytes / 2 ** 30)} GB, ` +
    `${Math.round(v.limits.max_video_seconds / 60)} min`;
}).catch(() => {});
