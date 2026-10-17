# PrivateLearn — Demo Player Design

The player is the thing a judge sees for 5 minutes. It must look **deliberate**, not
generated. This document is the spec; `web/index.html`, `web/player.css` and `web/player.js`
implement it and may not deviate from it. If the code and this doc disagree, the code is wrong.

## 1. Intent

The product is a **privacy laboratory for hospital data** — precise, calm, trustworthy. The
visual language is a clinical instrument, not a consumer app: dark, quiet, data-first. The
emotion comes from the numbers and the motion, never from decoration.

Three feelings to land, in order:
1. **Trust** — the "0 bytes" counter never leaves the screen and never changes.
2. **Clarity** — a judge can read the three headline numbers (solo / federated / centralized)
   without anyone explaining them.
3. **Tension** — the attack panel shows a sample being *recovered*, then the same attack
   *failing*. That before/after is the payoff.

## 2. Principles

1. **Data is the hero.** Every pixel of chrome must serve the numbers. No gradients for
   decoration, no drop shadows that don't mean something.
2. **One color per meaning.** A color used for two different meanings is a bug.
3. **Motion only where it carries information.** Packets animate because they show the flow
   of updates; nothing else should move unless it is telling the judge something.
4. **Tabular numerals everywhere.** Every number renders with `font-variant-numeric: tabular-nums`
   so digits don't jitter as values change.
5. **A judge must be able to screenshot any frame and read it cold.**

## 3. Color tokens

Semantic tokens only. No raw hex values in components — always a variable.

```css
--bg            #0a0e14   page background (near-black, blue-tinted)
--bg-raise      #0e131c   subtle raised surface
--panel         #111826   card background
--panel-2       #171f30   nested surface (chips, wells)
--border        #222c3d   hairlines
--border-strong #2e3a50   focus/hover hairlines

--text          #e7ecf4   primary text
--text-dim      #9aa5b8   secondary text
--text-faint    #5c6a80   tertiary / labels

--hosp-a        #4d8bff   Hospital A
--hosp-b        #3fcf8e   Hospital B
--hosp-c        #ff6b6b   Hospital C
--coord         #f5b94d   Coordinator

--ok            #46d48a   privacy intact / attack FAILED (the good outcome)
--danger        #ff5c5c   data exposed / attack RECOVERED (the bad outcome)
```

Semantic rules:
- **Green (`--ok`) means "private".** The "0 bytes" counter is green; the attack *failing* is green.
- **Red (`--danger`) means "exposed".** The attack *recovering* a row is red.
- Hospital and coordinator colors are used **only** for those entities — never as generic accents.

## 4. Typography

Two families, never more:

| role | font | usage |
|---|---|---|
| Data | `ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Consolas, monospace` | numbers, weights, hashes, byte counter, errors |
| UI | `"Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` | headings, labels, prose |

Scale (no other sizes):

```
12px   micro labels, log timestamps
13px   body, legend
14px   panel headings, buttons
16px   stage caption
20px   page title
32px   the "0 bytes" hero number
```

All numerals use `tabular-nums`.

## 5. Spacing, radii, borders

- Spacing on a 4px grid: `4 8 12 16 24 32`. No other values.
- Panel radius `8px`; pill/chip radius `999px`.
- Hairlines are `1px solid var(--border)`. Hover/focus is `var(--border-strong)`.

## 6. Layout (three zones)

```
┌────────────────────────────────────────────────────────────────────────┐
│ topbar: brand · HERO COUNTER · controls(play reset scrub round)        │
├──────────────────────────────────────────┬─────────────────────────────┤
│  stage (panel)                            │  rail                       │
│  ┌ panel-head: "Federation" · legend ┐    │  ┌ accuracy (AUC)         ┐ │
│  ├ canvas: node graph                ┤    │  ├ decision log           ┤ │
│  ├ stage caption                     ┤    │  └ gradient-inversion     ┘ │
│  └────────────────────────────────────┘    │                             │
└──────────────────────────────────────────┴─────────────────────────────┘
```

- **Topbar** (56px tall): brand left, hero counter center, controls right. The counter is the
  largest text on the page and always green.
- **Stage** (flex 1): the node graph. Hospitals along the top, coordinator centered below,
  edges fanning in. A legend sits in the panel head.
- **Rail** (fixed 360px): three stacked panels — accuracy, decision log, attack.

The whole page fits a 16:9 recording frame at 1280×720 without scrolling.

## 7. Components

### 7.1 Node graph (canvas)
- Background: `--panel`, with a faint 32px dot grid in `--bg-raise`.
- **Hospitals:** filled disc in the hospital color, a short label, and a small "4 agents"
  tag beneath in `--text-faint` at 10px (the full role list lives in the stage caption).
- **Coordinator:** a rounded square (not a circle) in `--coord`, label "Coordinator".
- **Edges:** 1px `--border` lines from each hospital to the coordinator.
- **Packets:** 5px disc in the source hospital's color, moving along the edge with an
  ease-in-out curve. Broadcast packets move coordinator → hospital in `--coord` and only
  to hospitals not in `round.excluded`.
- A packet is 7 weights; draw it as a small stack (2–3 short bars) rather than a plain dot,
  so it reads as "numbers", not "a thing".

### 7.2 Accuracy panel
- Line chart of `metrics.auc_series` in `--coord`, 2px, with a dot per round.
- Two dashed reference lines, labelled at the left: **solo** in `--danger`, **centralized** in `--ok`.
- Y axis labelled AUC (0.6 → 1.0), X axis = rounds. Gridlines every 0.1 in `--border`.
- Legend line under the chart: solo / federated / centralized in their colors.

### 7.3 Decision log
- One row per completed round: a strategy **badge** (pill) + the round number + the `decision`
  string in `--text-dim` at 13px.
- Badge colors: `fedavg` = `--coord`, `fedprox` = `--hosp-a`, `trimmed_mean` = `--hosp-c`.
- Rows where `round.excluded` is non-empty get a left border in `--danger` and a small
  "excluded h2" tag.

### 7.4 Gradient-inversion panel
- Two cards stacked, each: a label, a big `RECOVERED` / `FAILED` stamp, and the `feature_error`
  number in monospace.
- Top card "Without protection": if `recovered` → red stamp **RECOVERED** + the error.
- Bottom card "With masking + DP": if `not recovered` → green stamp **FAILED** + the error.
- Caption: "same attack, same sample — only the privacy layer changed."

### 7.5 Buttons & controls
- Primary (`Play`/`Pause`): filled `--coord` with dark text.
- Ghost (`Reset`): transparent, `--border` hairline.
- Scrubber: thin track `--border`, thumb `--coord`.
- Round label: `Round 2 / 4` in `--text-dim`, tabular.

## 8. Motion

- Packet travel: `ease-in-out`, ~600ms per hop, looped per round.
- Round advance: on packet completion, advance one round; the decision log grows and the chart
  dot for that round fills.
- Nothing else animates. No parallax, no pulsing, no blinking. `prefers-reduced-motion` turns
  all packet animation off and jumps straight to the final state.

## 9. States

- **Loading:** the canvas area shows `loading trace…` in `--text-faint`.
- **Error:** a full-page monospace block in `--danger` with the error and the "serve with
  `python -m http.server`" hint.
- **Empty (no rounds yet):** graph drawn, log empty, chart shows only the reference lines.

## 10. Verification

The author cannot see a browser. After any change, **Person C must**:
1. Open `http://localhost:8000/web/index.html` and confirm the three-zone layout at 1280×720.
2. Confirm the hero counter stays green `0` for the whole replay.
3. Confirm the attack panel shows red RECOVERED then green FAILED.
4. Confirm three headline numbers are legible in a still screenshot.
