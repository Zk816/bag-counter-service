// direction convention must match bagcounter/geometry.py: positive = side the normal (-dy, dx) points to
export class LineEditor {
  constructor(img, canvas, onChange) {
    this.img = img; this.canvas = canvas; this.onChange = onChange;
    this.pts = []; this.flip = false; this.both = false; this.hover = null;
    canvas.addEventListener("click", (e) => this.click(e));
    canvas.addEventListener("mousemove", (e) => { this.hover = this.pos(e); this.draw(); });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(img);
    img.addEventListener("load", () => this.draw());
  }
  pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  }
  click(e) {
    if (this.pts.length >= 2) this.pts = [];
    this.pts.push(this.pos(e)); this.draw(); this.onChange();
  }
  set(line) {
    this.pts = line ? [[line.ax, line.ay], [line.bx, line.by]] : [];
    this.both = line?.direction === "both"; this.flip = line?.direction === "negative"; this.draw();
  }
  clear() { this.pts = []; this.draw(); this.onChange(); }
  valid() {
    if (this.pts.length !== 2) return false;
    const [[ax, ay], [bx, by]] = this.pts;
    return Math.hypot(bx - ax, by - ay) >= 0.02;
  }
  value() {
    const [[ax, ay], [bx, by]] = this.pts;
    const r = (v) => Math.round(v * 10000) / 10000;
    return { ax: r(ax), ay: r(ay), bx: r(bx), by: r(by), direction: this.both ? "both" : this.flip ? "negative" : "positive" };
  }
  draw() {
    const c = this.canvas, w = c.clientWidth, h = c.clientHeight, dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const P = (p) => [p[0] * w, p[1] * h];
    if (this.pts.length === 1 && this.hover) {
      const [x1, y1] = P(this.pts[0]), [x2, y2] = P(this.hover);
      g.setLineDash([6, 6]); g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.setLineDash([]);
    }
    for (const p of this.pts) {
      const [x, y] = P(p);
      g.fillStyle = "#e11d48"; g.beginPath(); g.arc(x, y, 6, 0, Math.PI * 2); g.fill();
    }
    if (this.pts.length !== 2) return;
    const [ax, ay] = P(this.pts[0]), [bx, by] = P(this.pts[1]);
    g.strokeStyle = "#e11d48"; g.lineWidth = 4; g.lineCap = "round";
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
    const mx = (ax + bx) / 2, my = (ay + by) / 2, dx = bx - ax, dy = by - ay, n = Math.hypot(dx, dy) || 1;
    const nx = -dy / n, ny = dx / n, len = Math.max(36, Math.min(80, n * 0.25));
    const arrow = (sx, sy) => {
      const ex = mx + sx * len, ey = my + sy * len, a = Math.atan2(sy, sx);
      g.strokeStyle = "#fde047"; g.fillStyle = "#fde047"; g.lineWidth = 4;
      g.beginPath(); g.moveTo(mx, my); g.lineTo(ex, ey); g.stroke();
      g.beginPath(); g.moveTo(ex, ey);
      g.lineTo(ex - 14 * Math.cos(a - 0.45), ey - 14 * Math.sin(a - 0.45));
      g.lineTo(ex - 14 * Math.cos(a + 0.45), ey - 14 * Math.sin(a + 0.45)); g.closePath(); g.fill();
    };
    if (this.both || !this.flip) arrow(nx, ny);
    if (this.both || this.flip) arrow(-nx, -ny);
    const label = this.both ? "counted both ways" : "counted direction";
    g.font = "600 13px system-ui";
    const tw = g.measureText(label).width;
    const tx = Math.min(Math.max(4, mx + 10), w - tw - 6), ty = Math.max(16, my - 12);
    g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(tx - 4, ty - 13, tw + 8, 18);
    g.fillStyle = "#fde047"; g.fillText(label, tx, ty);
  }
}

export function bindDirectionControls(editor, flip, both) {
  flip.addEventListener("click", () => { editor.flip = !editor.flip; editor.both = false; both.checked = false; editor.draw(); });
  both.addEventListener("change", (e) => { editor.both = e.target.checked; editor.draw(); });
}
