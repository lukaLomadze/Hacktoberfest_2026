// The federation canvas, ported from web/player.js (DESIGN.md §7.1) for any number of
// participants: companies along the top, the coordinator below, an edge from each, packets
// animated along the edges as `packet` events arrive. Colors come from the CSS tokens and from
// the server-assigned participant colors (PLATFORM-API §4.5), never from literals here.

import { AGENT_LETTERS } from "./state.js";

const HOP_MS = 600; // DESIGN §8: ~600 ms per hop
const HEIGHT = 320;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const T = (n) => css.getPropertyValue(`--${n}`).trim();
  return {
    bgRaise: T("bg-raise"), panel: T("panel"), border: T("border"), text: T("text"), dim: T("text-dim"),
    faint: T("text-faint"), coord: T("coord"), ok: T("ok"), danger: T("danger"),
    fallback: [T("hosp-a"), T("hosp-b"), T("hosp-c")],
    fontData: T("font-data"), fontUi: T("font-ui"),
  };
}

// participant.state → how the node is drawn
const DIM_STATES = new Set(["offline"]);
const BAD_STATES = new Set(["offline", "excluded"]);
const BUSY_STATES = new Set(["probing", "planning", "training", "masking", "sent"]);

/** `secure` (the TaskSpec's secure_aggregation) only changes the coordinator's caption. */
export function createCanvas({ secure = true } = {}) {
  const canvas = document.createElement("canvas");
  canvas.className = "live-canvas";
  canvas.setAttribute("role", "img");
  const C = tokens();
  let ctx = null;
  let width = 0;
  let state = null;
  let packets = []; // {from, to, kind, masked, rejected, t0}
  let raf = 0;

  function resize() {
    const w = Math.max(280, Math.round(canvas.parentElement ? canvas.parentElement.clientWidth : 640));
    const dpr = window.devicePixelRatio || 1;
    width = w;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(HEIGHT * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${HEIGHT}px`;
    ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  const ids = () => (state ? [...state.participants.keys()] : []);

  function pos(id) {
    if (id === "coord") return { x: width / 2, y: 236 };
    const list = ids();
    const i = list.indexOf(id);
    const n = Math.max(1, list.length);
    const margin = Math.min(110, width / (n * 2));
    const x = n === 1 ? width / 2 : margin + (i * (width - 2 * margin)) / (n - 1);
    return { x, y: 70 };
  }

  function colorOf(id) {
    if (id === "coord") return C.coord;
    const p = state && state.participants.get(id);
    if (p && p.color) return p.color;
    const i = ids().indexOf(id);
    return C.fallback[(i < 0 ? 0 : i) % C.fallback.length];
  }

  function text(str, x, y, { size = 13, color = C.dim, font = C.fontUi, weight = "", align = "center", max } = {}) {
    ctx.font = `${weight} ${size}px ${font}`.trim();
    ctx.fillStyle = color;
    ctx.textAlign = align;
    let s = String(str);
    if (max) while (s.length > 3 && ctx.measureText(s).width > max) s = `${s.slice(0, -2)}…`;
    ctx.fillText(s, x, y);
    ctx.textAlign = "left";
  }

  function roundedRect(x, y, w, hh, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + hh, r);
    ctx.arcTo(x + w, y + hh, x, y + hh, r);
    ctx.arcTo(x, y + hh, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawLock(x, y) {
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y - 3, 3, Math.PI, 0);
    ctx.stroke();
    ctx.fillStyle = C.text;
    ctx.fillRect(x - 4.5, y - 3, 9, 7);
  }

  function drawParticipant(p, slot) {
    const { x, y } = pos(p.node_id);
    const color = colorOf(p.node_id);
    ctx.globalAlpha = DIM_STATES.has(p.state) ? 0.3 : 1;
    ctx.beginPath();
    ctx.arc(x, y, 22, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    if (BUSY_STATES.has(p.state)) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = C.text;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    text(p.company_name, x, y - 32, { size: 13, color: C.text, weight: 600, max: slot });
    text(p.state.replaceAll("_", " "), x, y + 38, {
      size: 12, font: C.fontData, color: BAD_STATES.has(p.state) ? C.danger : C.dim, max: slot,
    });
    // S P T C: lit when that agent spoke this round (round 0 = setup)
    const acted = p.acted.get(state.round) || new Set();
    const w = 16;
    const gap = 3;
    const x0 = x - (AGENT_LETTERS.length * w + (AGENT_LETTERS.length - 1) * gap) / 2;
    AGENT_LETTERS.forEach(([agent, letter], i) => {
      const cx = x0 + i * (w + gap);
      roundedRect(cx, y + 48, w, 14, 3);
      if (acted.has(agent)) {
        ctx.fillStyle = color;
        ctx.fill();
      } else {
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      text(letter, cx + w / 2, y + 59, { size: 10, weight: 700, font: C.fontData, color: acted.has(agent) ? C.panel : C.faint });
    });
  }

  function drawPacket(pk, t) {
    const a = pos(pk.from);
    const b = pos(pk.to);
    const stop = pk.rejected ? 0.78 : 1; // a rejected update never reaches aggregation
    const k = ease(Math.min(t, 1)) * stop;
    const x = a.x + (b.x - a.x) * k;
    const y = a.y + (b.y - a.y) * k;
    const halted = pk.rejected && t >= 1;
    ctx.fillStyle = halted ? C.danger : pk.kind === "broadcast" || pk.kind === "probe" ? C.coord : colorOf(pk.from);
    if (pk.kind === "key" || pk.kind === "probe") {
      // small messages: a single dot, not a weight vector
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      [10, 14, 8].forEach((hh, i) => ctx.fillRect(x - 7 + i * 5, y + 7 - hh, 4, hh));
    }
    if (pk.masked) drawLock(x + 14, y + 2);
    if (halted) {
      ctx.strokeStyle = C.danger;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 10, y - 14); ctx.lineTo(x + 10, y + 6);
      ctx.moveTo(x + 10, y - 14); ctx.lineTo(x - 10, y + 6);
      ctx.stroke();
    }
  }

  function draw(now = performance.now()) {
    if (!ctx) return;
    ctx.clearRect(0, 0, width, HEIGHT);
    ctx.fillStyle = C.bgRaise;
    for (let x = 16; x < width; x += 32) for (let y = 16; y < HEIGHT; y += 32) ctx.fillRect(x - 1, y - 1, 2, 2);
    if (!state) {
      text("loading…", width / 2, HEIGHT / 2, { color: C.faint, font: C.fontData });
      return;
    }
    const list = ids();
    const c = pos("coord");
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    for (const id of list) {
      const p = pos(id);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(c.x, c.y);
      ctx.stroke();
    }
    const slot = list.length > 1 ? (width - 40) / list.length : width - 40;
    for (const p of state.participants.values()) drawParticipant(p, slot);

    roundedRect(c.x - 24, c.y - 24, 48, 48, 8);
    ctx.fillStyle = C.coord;
    ctx.fill();
    text("Coordinator", c.x, c.y + 42, { size: 13, color: C.text, weight: 600 });
    text(secure ? "sees only the masked sum Σ" : "aggregates the updates", c.x, c.y + 60, { size: 12 });
    if (!list.length) text("no participants yet", width / 2, 80, { color: C.faint, font: C.fontData });

    packets = packets.filter((pk) => now - pk.t0 < HOP_MS * 1.4); // hold briefly at the end
    for (const pk of packets) drawPacket(pk, (now - pk.t0) / HOP_MS);
    canvas.setAttribute("aria-label", `${list.length} participants and the coordinator; round ${state.round}`);
  }

  function loop() {
    draw();
    raf = packets.length ? requestAnimationFrame(loop) : 0;
  }

  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => resize()) : null;

  return {
    el: canvas,
    mount() {
      resize();
      if (observer && canvas.parentElement) observer.observe(canvas.parentElement);
    },
    render(s) {
      state = s;
      if (!raf) draw();
    },
    /** Animate one §6 packet event (skipped under prefers-reduced-motion, and when `animate` is off). */
    packet(e, animate = true) {
      if (REDUCED || !animate || !e.from || !e.to) return;
      packets.push({ from: e.from, to: e.to, kind: e.kind, masked: !!e.masked, rejected: !!e.rejected,
                     t0: performance.now() });
      if (!raf) raf = requestAnimationFrame(loop);
    },
    destroy() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      packets = [];
      if (observer) observer.disconnect();
    },
  };
}
