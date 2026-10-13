import { $, $$, api, errText, showErr } from "./dom.js";
import { LineEditor, bindDirectionControls } from "./line-editor.js";

let camTimer = null;
$("#cam-add-open").addEventListener("click", () => $("#cam-add").classList.remove("hidden"));
$("#cam-add-cancel").addEventListener("click", () => $("#cam-add").classList.add("hidden"));
$("#cam-add").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    await api("/v1/cameras", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: f.get("name"), url: f.get("url") }) });
    e.target.reset(); e.target.classList.add("hidden"); $("#cam-err").classList.add("hidden");
    loadCameras();
  } catch (err) { showErr("#cam-err", errText(err)); }
});

const STATE_TEXT = { live: "live", connecting: "connecting…", stalled: "no signal", frozen: "picture frozen",
  needs_line: "draw a line", disabled: "off", error: "error", unknown: "worker not reporting", pending: "starting…" };

export async function loadCameras() {
  clearTimeout(camTimer);
  if (!$("#tab-cameras").classList.contains("active")) return;
  let cams = [];
  try { cams = await api("/v1/cameras"); } catch { camTimer = setTimeout(loadCameras, 5000); return; }
  $("#cams-empty").classList.toggle("hidden", cams.length > 0);
  const root = $("#cams");
  const ids = new Set(cams.map((c) => c.camera_id));
  $$(".cam", root).forEach((el) => { if (!ids.has(el.dataset.id)) el.remove(); });
  for (const c of cams) {
    let el = $(`.cam[data-id="${c.camera_id}"]`, root);
    if (!el) {
      el = document.createElement("div");
      el.className = "card cam"; el.dataset.id = c.camera_id;
      el.innerHTML = `<div class="cam-top"><div><div class="cam-name"></div><div class="cam-url"></div></div><span class="badge"></span></div>
        <div class="pic"><img alt="live picture"></div>
        <div class="cam-count"><span class="big"></span><span class="muted">bags in total</span></div>
        <div class="cam-stats"></div>
        <div class="controls"><button class="ghost" data-act="line">Set counting line</button>
          <button class="ghost" data-act="toggle"></button><button class="ghost danger" data-act="delete">Delete</button></div>`;
      $("img", el).src = c.live_url;
      el.addEventListener("click", (e) => camAction(e, c.camera_id));
      root.append(el);
    }
    el.cam = c;
    $(".cam-name", el).textContent = c.name;
    $(".cam-url", el).textContent = c.url;
    const b = $(".badge", el); b.className = `badge ${c.state}`; b.textContent = STATE_TEXT[c.state] || c.state;
    $(".big", el).textContent = c.count_total;
    const s = c.status || {};
    $(".cam-stats", el).textContent = c.enabled && s.width ? `${s.width}×${s.height} · ${s.processed_fps} fps processed · ` +
      `latency p95 ${s.latency_ms_p95 ?? "–"} ms · ${Math.round((s.dropped_share || 0) * 100)}% frames skipped · ` +
      `${s.reconnects} reconnects` : "";
    $('[data-act="toggle"]', el).textContent = c.enabled ? "Pause" : "Resume";
  }
  camTimer = setTimeout(loadCameras, 3000);
}

async function camAction(e, id) {
  const act = e.target.dataset?.act; if (!act) return;
  const el = e.currentTarget, c = el.cam;
  if (act === "delete" && confirm(`Delete camera "${c.name}" and its count history?`)) {
    await api(`/v1/cameras/${id}`, { method: "DELETE" }); loadCameras();
  } else if (act === "toggle") {
    await api(`/v1/cameras/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !c.enabled }) }); loadCameras();
  } else if (act === "line") openLineDialog(c);
}

const dlg = $("#line-dialog");
const ldEditor = new LineEditor($("#ld-img"), $("#ld-canvas"), () => { $("#ld-save").disabled = !ldEditor.valid(); });
bindDirectionControls(ldEditor, $("#ld-flip"), $("#ld-both"));
$("#ld-cancel").addEventListener("click", () => dlg.close());
function openLineDialog(c) {
  dlg.camId = c.camera_id;
  $("#ld-name").textContent = c.name;
  $("#ld-img").src = `${c.preview_url}?t=${Date.now()}`;
  $("#ld-img").onerror = () => { $("#ld-img").alt = "no picture from this camera yet – check the URL"; };
  ldEditor.set(c.line); $("#ld-both").checked = ldEditor.both; $("#ld-save").disabled = !ldEditor.valid();
  dlg.showModal(); setTimeout(() => ldEditor.draw(), 50);
}
$("#ld-save").addEventListener("click", async () => {
  await api(`/v1/cameras/${dlg.camId}`, { method: "PATCH", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ line: ldEditor.value() }) });
  dlg.close(); loadCameras();
});
