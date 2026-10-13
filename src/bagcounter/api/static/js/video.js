import { $, $$, api, errText, esc, fmtDur, setBar, showErr } from "./dom.js";
import { LineEditor, bindDirectionControls } from "./line-editor.js";

const state = { upload: null, job: null, poll: null };
const editor = new LineEditor($("#preview"), $("#line-canvas"), () => { $("#start").disabled = !editor.valid(); });
bindDirectionControls(editor, $("#flip"), $("#both"));
$("#line-clear").addEventListener("click", () => editor.clear());

const drop = $("#drop");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));

function upload(file) {
  resetResult();
  $("#up-err").classList.add("hidden");
  $("#step-line").classList.add("disabled");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/v1/uploads?filename=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = (e) => e.lengthComputable && setBar($("#up-bar"), e.loaded / e.total,
    `${file.name} · ${Math.round((100 * e.loaded) / e.total)}%`);
  xhr.onload = () => {
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch { body = null; }
    if (xhr.status !== 201) {
      showErr("#up-err", body?.title ? `${body.title} (${body.code})` : `Upload failed (${xhr.status})`);
      return;
    }
    state.upload = body;
    const v = body.video;
    setBar($("#up-bar"), 1, `${file.name} · ${v.width}×${v.height} · ${v.fps} fps · ${fmtDur(v.duration_s)}`);
    $("#preview").src = body.preview_url;
    editor.clear();
    $("#step-line").classList.remove("disabled");
    $("#step-line").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  xhr.onerror = () => showErr("#up-err", "Upload failed: connection lost");
  xhr.send(file);
}

function resetResult() {
  clearInterval(state.poll);
  $("#step-result").classList.add("disabled");
  ["#job-running", "#job-result", "#job-err"].forEach((s) => $(s).classList.add("hidden"));
}

$("#start").addEventListener("click", async () => {
  if (!state.upload || !editor.valid()) return;
  $("#start").disabled = true;
  try {
    const job = await api("/v1/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: state.upload.upload_id, line: editor.value(), annotate: $("#annotate").checked }),
    });
    state.job = job;
    $("#step-result").classList.remove("disabled");
    $("#job-running").classList.remove("hidden");
    $("#step-result").scrollIntoView({ behavior: "smooth", block: "start" });
    state.poll = setInterval(pollJob, 1000); pollJob();
  } catch (e) {
    $("#start").disabled = false;
    $("#step-result").classList.remove("disabled");
    showErr("#job-err", errText(e));
  }
});

$("#cancel").addEventListener("click", async () => { if (state.job) await api(`/v1/jobs/${state.job.job_id}/cancel`, { method: "POST" }); });

async function pollJob() {
  let j;
  try { j = await api(`/v1/jobs/${state.job.job_id}`); } catch { return; }
  setBar($("#job-bar"), j.progress, j.status === "queued" ? "waiting in queue…" : `${Math.round(j.progress * 100)}%`);
  $("#job-live").textContent = j.count ?? 0;
  if (["succeeded", "failed", "cancelled"].includes(j.status)) {
    clearInterval(state.poll);
    $("#job-running").classList.add("hidden");
    if (j.status === "succeeded") renderResult(j);
    else showErr("#job-err", j.status === "cancelled" ? "Cancelled." : `Failed: ${j.error?.message} (${j.error?.code})`);
    state.upload = null;
  }
}

async function renderResult(j) {
  const s = j.summary || {};
  const box = $("#job-result");
  box.classList.remove("hidden");
  box.innerHTML = `
    <div class="result-head">
      <div><div class="big">${j.count}</div><div class="muted">bags counted</div></div>
      <div class="kv">
        <div>Video<b>${esc(j.filename)}</b></div>
        <div>Length<b>${fmtDur(s.video?.duration_s)}</b></div>
        <div>Processing<b>${fmtDur(s.processing_s)} · ${s.processing_fps} fps</b></div>
        <div>Counted after a dropout<b>${s.late_events ?? 0}</b></div>
        <div>Model<b>${esc(j.model_version)}</b></div>
      </div>
    </div>
    <div class="downloads">
      ${j.files.annotated_video ? `<a class="ghost" href="${j.files.annotated_video}" download>⬇ Annotated video</a>` : ""}
      <a class="ghost" href="${j.files.events_csv}" download>⬇ Events (CSV)</a>
      <a class="ghost" href="${j.files.report_json}" download>⬇ Report (JSON)</a>
    </div>
    ${j.files.annotated_video ? `<video controls preload="metadata" src="${j.files.annotated_video}"></video>` : ""}
    <div class="chart" id="chart"></div>
    <h3>Every counted bag</h3>
    <p class="hint">Click a picture to jump to that moment in the video. Check these to verify the count.</p>
    <div class="thumbs" id="thumbs"></div>`;
  const rep = await api(j.files.report_json);
  chart($("#chart"), rep.events.map((e) => e.time_s), s.video?.duration_s || 1);
  $("#thumbs").innerHTML = rep.events.map((e) => `
    <figure data-t="${e.time_s}"><img loading="lazy" src="/v1/jobs/${j.job_id}/files/${e.snapshot}" alt="bag ${e.seq}">
    <figcaption>#${e.seq} · ${e.time_s.toFixed(1)} s${e.late ? ' · <span class="late">late</span>' : ""}</figcaption></figure>`).join("");
  $$("#thumbs figure").forEach((f) => f.addEventListener("click", () => {
    const v = $("#job-result video");
    if (v) { v.currentTime = Math.max(0, +f.dataset.t - 1); v.play(); v.scrollIntoView({ behavior: "smooth" }); }
  }));
}

function chart(el, times, duration) {
  const bucket = [1, 5, 10, 30, 60, 300, 900, 3600].find((b) => duration / b <= 120) || 3600;
  const n = Math.max(1, Math.ceil(duration / bucket));
  const counts = new Array(n).fill(0);
  times.forEach((t) => counts[Math.min(n - 1, Math.floor(t / bucket))]++);
  const max = Math.max(1, ...counts), W = 1000, H = 120, bw = W / n;
  const unit = bucket < 60 ? `${bucket} s` : `${bucket / 60} min`;
  el.innerHTML = `<div class="hint">Bags per ${unit}</div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${counts.map((c, i) => {
      const h = (c / max) * (H - 16);
      return `<rect x="${i * bw + 1}" y="${H - 14 - h}" width="${Math.max(1, bw - 2)}" height="${h}"><title>${c} bags</title></rect>`;
    }).join("")}<text x="0" y="${H - 2}">0</text><text x="${W - 40}" y="${H - 2}">${fmtDur(duration)}</text></svg>`;
}
