// AUC and loss charts for the live view, ported from web/player.js drawSeries (DESIGN.md §7.2):
// the AUC axis is fitted to the data in 0.05 steps (never a fixed 0.6–1.0), with dashed
// reference lines once the published model reports them (centralized in --ok, the company
// that is worst off alone in --danger).

import { h } from "../app.js";

const fmt = (v, d = 3) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const T = (n) => css.getPropertyValue(`--${n}`).trim();
  return { border: T("border"), faint: T("text-faint"), dim: T("text-dim"), coord: T("coord"), ok: T("ok"),
           danger: T("danger"), fontData: T("font-data") };
}

function setup(canvas, height) {
  const w = Math.max(200, Math.round(canvas.parentElement ? canvas.parentElement.clientWidth : 320));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: height };
}

function drawSeries(C, S, points, { refs = [], color, pad = 30, decimals = 1, step } = {}) {
  const { ctx, w, h: hh } = S;
  ctx.clearRect(0, 0, w, hh);
  const series = points.map((p) => p.value);
  const vals = series.concat(refs.map((r) => r.v)).filter((v) => typeof v === "number" && Number.isFinite(v));
  if (!vals.length) {
    ctx.font = `12px ${C.fontData}`;
    ctx.fillStyle = C.faint;
    ctx.textAlign = "center";
    ctx.fillText("waiting for the first round", w / 2, hh / 2);
    ctx.textAlign = "left";
    return;
  }
  let min;
  let max;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  if (step) {
    min = Math.floor((lo - 0.01) / step) * step;
    max = Math.ceil((hi + 0.01) / step) * step;
    if (max - min < step * 3) max = min + step * 3;
  } else {
    const span = Math.max(hi - lo, 1e-3);
    min = lo - span * 0.15;
    max = hi + span * 0.15;
  }
  const rounds = points.map((p) => p.round);
  const r0 = Math.min(0, ...rounds);
  const r1 = Math.max(r0 + 1, ...rounds);
  const top = 6;
  const bottom = step ? 18 : 6;
  const ticks = [];
  if (step) for (let v = min; v <= max + 1e-9; v += step) ticks.push(v);
  ctx.font = `12px ${C.fontData}`;
  const widest = Math.max(0, ...ticks.map((v) => ctx.measureText(v.toFixed(decimals)).width));
  pad = Math.max(pad, Math.ceil(widest) + 8);
  const X = (r) => pad + ((r - r0) / (r1 - r0)) * (w - pad - 8);
  const Y = (v) => hh - bottom - ((v - min) / (max - min)) * (hh - top - bottom);

  if (step) {
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    ctx.fillStyle = C.faint;
    ctx.textAlign = "right";
    ticks.forEach((v) => {
      ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(w - 8, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(decimals), pad - 4, Y(v) + 4);
    });
    ctx.textAlign = "center";
    for (let r = r0; r <= r1; r++) ctx.fillText(`r${r}`, X(r), hh - 4);
    ctx.textAlign = "left";
  }
  refs.forEach((r) => {
    if (typeof r.v !== "number") return;
    ctx.strokeStyle = r.color;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, Y(r.v)); ctx.lineTo(w - 8, Y(r.v)); ctx.stroke();
    ctx.setLineDash([]);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p.round), Y(p.value)) : ctx.lineTo(X(p.round), Y(p.value))));
  ctx.stroke();
  ctx.fillStyle = color;
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(X(p.round), Y(p.value), i === points.length - 1 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

export function createChart() {
  const C = tokens();
  const auc = document.createElement("canvas");
  const loss = document.createElement("canvas");
  const aucNow = h("b.mono", "—");
  const lossNow = h("b.mono", "—");
  const legend = h("div.chart-legend");
  const el = h("section.panel.live-chart", { "aria-label": "Accuracy" },
    h("div.live-head", h("h2", "Global model AUC"), h("span.faint", "on the coordinator's evaluation set"), aucNow),
    h("div.chart-box", auc), legend,
    h("div.live-head", h("h2", "Training loss"), lossNow), h("div.chart-box", loss));
  let last = null;

  function render(state) {
    last = state;
    const m = (state.published && state.published.metrics) || {};
    const refs = [];
    const items = [h("span", h("i.sw", { class: "sw-fed" }), `federated ${fmt(state.auc.at(-1)?.value)}`)];
    if (typeof m.auc_centralized === "number") {
      refs.push({ v: m.auc_centralized, color: C.ok });
      items.push(h("span", h("i.sw.dash", { class: "sw-ok" }), `centralized ${fmt(m.auc_centralized)}`));
    }
    const solo = m.auc_solo_by_company && Object.entries(m.auc_solo_by_company);
    if (solo && solo.length) {
      const [, worst] = solo.sort((a, b) => a[1] - b[1])[0];
      refs.push({ v: worst, color: C.danger });
      items.push(h("span", h("i.sw.dash", { class: "sw-bad" }), `worst company alone ${fmt(worst)}`));
    }
    legend.replaceChildren(...items);
    drawSeries(C, setup(auc, 160), state.auc, { refs, color: C.coord, step: 0.05, decimals: 2 });
    drawSeries(C, setup(loss, 56), state.loss, { color: C.dim, pad: 8 });
    aucNow.textContent = fmt(state.auc.at(-1)?.value);
    lossNow.textContent = fmt(state.loss.at(-1)?.value);
  }

  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => last && render(last)) : null;
  return {
    el,
    render,
    mount() { if (observer) observer.observe(el); },
    destroy() { if (observer) observer.disconnect(); },
  };
}
