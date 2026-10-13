import { $, api, esc, fmtDur, fmtTime } from "./dom.js";

export async function loadJobs() {
  const jobs = await api("/v1/jobs?limit=50");
  $("#jobs tbody").innerHTML = jobs.map((j) => `<tr>
    <td>${esc(j.filename)}</td><td><span class="pill ${j.status}">${j.status}</span>${j.error ? ` <span class="muted">${esc(j.error.code)}</span>` : ""}</td>
    <td class="num">${j.count ?? "–"}</td><td>${fmtTime(j.started_at)}</td>
    <td>${j.finished_at && j.started_at ? fmtDur(j.finished_at - j.started_at) : "–"}</td>
    <td>${j.files ? `<a href="${j.files.report_json}" download>report</a> · <a href="${j.files.events_csv}" download>csv</a>` : ""}</td></tr>`).join("")
    || `<tr><td colspan="6" class="muted">No jobs yet.</td></tr>`;
}
