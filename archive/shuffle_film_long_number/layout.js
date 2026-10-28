/* The Long Number: layout(frame), a pure function of the frame number.

   Loaded by page.html (where render() draws what layout() returns) and usable from Node for tests.
   Everything here reads build/story.json (passed to init) and a measuring function; it never touches
   a clock, a random source or the DOM. The same layout() feeds the renderer and the checker, so the
   checks cannot drift from the picture.

   Coordinates: the page layer lives in page units (1080 x 1920 at scale 1); the page transform maps it
   to the screen: x' = x s + tx, y' = y s + ty with tx = 800 (1 - s), so the spine x = 800 never moves,
   and ty = pivot_y (1 - s) (story.json camera.pivot_y ~ 234.8), so the page contracts about one fixed
   screen point beside the deck and every page point moves monotonically while s shrinks. s is a PCHIP
   through story.json's camera targets (one per landing); still to row 12's landing, settled from row 49.
   The screen layer (deck, hook, label area, block, beats) is never scaled. Cards are returned in screen
   coordinates with their own scale.

   Glow: a number in the mock's units. 0.5 is the rest halo baked into every glyph, 1.2 the stamp's
   bloom (the first two frames), smoothstepping back to 0.5 over 15 frames; an item that has stamped
   out keeps a decaying halo (a "ghost", ink false) for 9 frames.
*/
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LongNumber = factory();
})(typeof self !== 'undefined' ? self : this, function () {
'use strict';

const LN = {};
let S = null;            // story.json
let measureRaw = null;   // (text, cssFont) -> {width, left, right, ascent, descent}
const mcache = new Map();
let camS = null, camTy = null, camStillAfter = Infinity, camPivot = null;

// ------------------------------------------------------------------------------------------ maths
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = v => clamp(v, 0, 1);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => { t = clamp01(t); return t * t * (3 - 2 * t); };
const easeOutExp = (t, k) => { t = clamp01(t); k = k || 5; return (1 - Math.exp(-k * t)) / (1 - Math.exp(-k)); };
const round3 = v => Math.round(v * 1000) / 1000;

function hexRgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
function rgbHex(r, g, b) {
  return '#' + [r, g, b].map(v => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0')).join('').toUpperCase();
}
function mixHex(a, b, t) {
  if (t <= 0) return a; if (t >= 1) return b;
  const A = hexRgb(a), B = hexRgb(b);
  return rgbHex(lerp(A[0], B[0], t), lerp(A[1], B[1], t), lerp(A[2], B[2], t));
}
const colour = role => S.meta.colours[role];

/* Monotone cubic (PCHIP, Fritsch-Carlson slopes with the three-point end rule) through (xs, ys);
   flat before the first point and after the last. */
function pchip(xs, ys) {
  const n = xs.length;
  if (n === 1) return () => ys[0];
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h.push(xs[i + 1] - xs[i]); d.push((ys[i + 1] - ys[i]) / h[i]); }
  const m = new Array(n).fill(0);
  if (n === 2) { m[0] = m[1] = d[0]; }
  else {
    for (let i = 1; i < n - 1; i++) {
      if (d[i - 1] * d[i] <= 0) m[i] = 0;
      else { const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
    }
    const end = (h0, h1, d0, d1) => {
      let v = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
      if (Math.sign(v) !== Math.sign(d0)) v = 0;
      else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(v) > 3 * Math.abs(d0)) v = 3 * d0;
      return v;
    };
    m[0] = end(h[0], h[1], d[0], d[1]);
    m[n - 1] = end(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
  }
  return function (x) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    const t = (x - xs[lo]) / h[lo], t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[lo] + (t3 - 2 * t2 + t) * h[lo] * m[lo]
         + (-2 * t3 + 3 * t2) * ys[lo + 1] + (t3 - t2) * h[lo] * m[lo + 1];
  };
}

// --------------------------------------------------------------------------------------- measuring
function fontOf(faceKey, size) { const d = S.meta.faces[faceKey]; return `${d.weight} ${size}px "${d.family}"`; }
function measure(text, faceKey, size) {
  const key = faceKey + '|' + size + '|' + text;
  let m = mcache.get(key);
  if (m === undefined) { m = measureRaw(text, fontOf(faceKey, size)); mcache.set(key, m); }
  return m;
}
const glowRadius = size => Math.max(1, 0.07 * size);   // the rest halo's visible edge (its stroke width), for the ink boxes

// ------------------------------------------------------------------------------------------- init
LN.init = function (story, measureFn) {
  S = story; measureRaw = measureFn; mcache.clear();
  const T = S.camera.targets, xs = T.map(t => t.frame);
  camS = pchip(xs, T.map(t => t.s));
  camTy = pchip(xs, T.map(t => t.ty));
  camPivot = typeof S.camera.pivot_y === 'number' ? S.camera.pivot_y : null;   // ty = pivot (1 - s) when the story gives a pivot
  // after the fold has faded the old page out, nothing is left on the page layer and the page is still again
  const fold = S.events.find(e => e.kind === 'fold');
  camStillAfter = fold ? fold.fade_out.end : Infinity;
  return LN;
};
LN.story = () => S;
LN.fontOf = fontOf;
LN.measure = measure;
LN.mixHex = mixHex;

function cameraAt(f) {
  if (f >= camStillAfter) return { s: 1, tx: 0, ty: 0 };
  const s = camS(f), ty = camPivot != null ? camPivot * (1 - s) : camTy(f);
  return { s, tx: S.geometry.spine_x * (1 - s), ty };
}
LN.cameraAt = cameraAt;
const toScreen = (cam, x, y) => [x * cam.s + cam.tx, y * cam.s + cam.ty];
function boxToScreen(cam, b) { const [x0, y0] = toScreen(cam, b.x0, b.y0), [x1, y1] = toScreen(cam, b.x1, b.y1); return { x0, y0, x1, y1 }; }

const rowBase = k => S.geometry.row_baseline.first + S.geometry.row_baseline.pitch * (k - 1);

// ------------------------------------------------------------------------------------------- glow
function bloomGlow(d, peak) {
  const B = S.meta.stamp_bloom;
  if (peak == null) peak = B.peak_glow;
  if (d < 0) return B.rest_glow;
  if (d < B.frames_at_peak) return peak;
  const t = (d - B.frames_at_peak) / B.settle_frames;
  if (t >= 1) return B.rest_glow;
  return B.rest_glow + (peak - B.rest_glow) * (1 - smooth(t));
}

// ------------------------------------------------------------------------------- the event state
function stateAt(f) {
  const st = {
    text: Object.create(null), rows: Object.create(null), cells: Object.create(null),
    flights: [], rules: { seconds: [], people: [] }, bands: [], passes: [], nebula: [],
    deck: { box: 'hook', reads: null, numOn: null, bloom: null, travel: null, leave: null, ret: null },
    fold: null,
  };
  const ts = id => (st.text[id] || (st.text[id] = { on: null, off: null, type: null, pieces: null, fold: null, fades: [], peak: null }));
  for (const e of S.events) {
    if (e.frame > f) break;
    switch (e.kind) {
      case 'nebula': st.nebula.push(e); break;
      case 'stamp':
        if (e.id === 'deck52') { st.deck.reads = 52; st.deck.numOn = e.frame; }
        else ts(e.id).on = e.frame;
        break;
      case 'unstamp': for (const id of e.ids) ts(id).off = e.frame; break;
      case 'type': { const t = ts(e.id); t.on = e.frame; t.type = e; } break;
      case 'fade': for (const id of e.ids) ts(id).fades.push(e); break;
      case 'lead': { const t = ts(e.id); if (t.on == null) t.on = e.frame; (t.pieces || (t.pieces = [])).push(e); } break;
      case 'fold': { st.fold = e; const t = ts(e.to.id); t.on = e.end; t.fold = e; } break;
      case 'flight': st.flights.push(e); st.deck.reads = e.deck_after || null; break;
      case 'row': st.rows[e.k] = e; break;
      case 'cell': st.cells[e.k] = e; break;
      case 'rule': st.rules[e.ref].push(e); break;
      case 'band': st.bands.push(e); break;
      case 'pass': st.passes.push(e); break;
      case 'bloom': st.deck.bloom = e; break;
      case 'deck_travel': st.deck.travel = e; st.deck.box = 'head'; break;
      case 'deck_leave': st.deck.box = null; st.deck.leave = e; st.deck.reads = null; break;
      case 'deck_return': st.deck.box = 'hook'; st.deck.ret = e; st.deck.reads = null; break;
      default: break;
    }
  }
  return st;
}

/* Visibility of a stamped / typed text at frame f: null (absent), or {ink, alpha, glow, ghost, since}. */
function visOf(t, f) {
  if (!t || t.on == null || f < t.on) return null;
  const decay = S.meta.unstamp_glow_decay_frames;
  if (t.off != null && f >= t.off) {
    const d = f - t.off;
    if (d >= decay) return null;
    return { ink: false, alpha: 1, glow: S.meta.stamp_bloom.rest_glow * (1 - d / decay), ghost: true, since: t.on };
  }
  let a = 1;
  for (const fd of t.fades) {
    if (f < fd.frame) break;
    a = lerp(a, fd.to_alpha, clamp01((f - fd.frame) / (fd.end - fd.frame)));
  }
  if (a <= 0) return null;
  return { ink: true, alpha: a, glow: bloomGlow(f - t.on), ghost: false, since: t.on };
}

// ------------------------------------------------------------------------------- text items
function placeItem(it) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, a0 = Infinity, a1 = -Infinity;
  for (const r of it.runs) {
    const m = measure(r.text, it.face, it.size);
    x0 = Math.min(x0, r.x - m.left); x1 = Math.max(x1, r.x + m.right);
    y0 = Math.min(y0, it.baseline - m.ascent); y1 = Math.max(y1, it.baseline + m.descent);
    a0 = Math.min(a0, r.x); a1 = Math.max(a1, r.x + m.width);
  }
  const g = glowRadius(it.size);
  it.box = { x0: x0 - g, y0: y0 - g, x1: x1 + g, y1: y1 + g };
  it.width = a1 - a0;
  return it;
}

/* A worded text (one run, or spans in their colours), fitted down in 2 px steps if over 808 px. */
function wordItem(id, t, vis, extra) {
  const maxW = S.geometry.safe.max_text_width;
  let size = t.size, fitted = false;
  for (;;) {
    const parts = [];
    let w = 0;
    if (t.spans) {
      for (const [s, c] of t.spans) { parts.push({ text: s, dx: w, colour: colour(c) }); w += measure(s, t.face, size).width; }
    } else {
      parts.push({ text: t.text, dx: 0, colour: colour(t.colour) }); w = measure(t.text, t.face, size).width;
    }
    if (w <= maxW || size <= 12) {
      const x0 = t.align === 'left' ? t.x : t.align === 'right' ? t.x - w : t.x - w / 2;
      const runs = parts.map(p => ({ text: p.text, x: x0 + p.dx, colour: p.colour, alpha: 1, glow: vis.glow }));
      return placeItem(Object.assign({
        kind: 'text', layer: t.layer, id, line: 0, text: t.text, face: t.face, size, bakeSize: size, mono: false,
        x: t.x, baseline: t.baseline, align: t.align, colour: colour(t.colour),
        alpha: vis.alpha, glow: vis.glow, ink: vis.ink, ghost: vis.ghost, fitted, runs, role: t.group,
      }, extra || {}));
    }
    size -= 2; fitted = true;
  }
}

/* A grouped number flush right at xRight, one item per line.
   groups: the digit groups (strings); lineGroups: groups per line; baselines per line; sig: bright digits
   (null = all bright); prefix on the first line; visibleGroup(g) -> {alpha, glow}|null per group. */
function numberLines(spec) {
  const { id, layer, groups, lineGroups, baselines, size, colourHex, sig, prefix, xRight, face, alpha, ink, ghost, role } = spec;
  const adv = S.geometry.mono.advance_em * size, gap = S.geometry.mono.group_gap_em * size;
  const placeholder = S.meta.row_style.placeholder_alpha;
  const items = [];
  let gi = 0, di = 0;
  for (let li = 0; li < lineGroups.length; li++) {
    const n = lineGroups[li], gs = groups.slice(gi, gi + n);
    const ndig = gs.reduce((a, g) => a + g.length, 0);
    const w = ndig * adv + (n - 1) * gap;
    const left = xRight - w;
    const runs = [];
    let x = left, shown = '';
    for (let j = 0; j < n; j++) {
      const g = gs[j], gIndex = gi + j;
      const v = spec.groupState ? spec.groupState(gIndex) : { alpha: 1, glow: spec.glow };
      if (v) {
        let run = null;
        for (let c = 0; c < g.length; c++) {
          const bright = sig == null || di + c < sig;
          const st = spec.digitStyle ? spec.digitStyle(di + c, bright) : null;
          const col = (st && st.colour) || colourHex;
          const gl = (st && st.glow != null) ? st.glow : v.glow;
          const al = v.alpha * (bright ? 1 : placeholder);
          if (run && run.colour === col && run.alpha === al && run.glow === gl) run.text += g[c];
          else { run = { text: g[c], x: x + c * adv, colour: col, alpha: al, glow: gl, group: gIndex }; runs.push(run); }
        }
        shown += g;
      }
      di += g.length;
      x += g.length * adv + gap;
    }
    if (li === 0 && prefix && runs.length) {
      const v0 = spec.groupState ? spec.groupState(0) : { alpha: 1, glow: spec.glow };
      runs.unshift({ text: prefix, x: left - 2 * adv, colour: colourHex, alpha: v0.alpha, glow: v0.glow, group: 0, prefix: true });
    }
    if (runs.length) {
      items.push(placeItem({
        kind: 'text', layer, id, line: li, text: gs.join(' '), digitsShown: shown, face, size, bakeSize: size, mono: true,
        x: xRight, baseline: baselines[li], align: 'right', colour: colourHex, alpha, glow: spec.glow, ink, ghost,
        runs, role, k: spec.k, ref: spec.ref, complete: spec.complete !== false, left, lineGroups: [gi, gi + n - 1],
      }));
    }
    gi += n;
  }
  return items;
}

// ------------------------------------------------------------------------------------- layout
LN.layout = function (f) {
  const st = stateAt(f);
  const cam = cameraAt(f);
  const G = S.geometry, C = S.meta.colours;
  const items = [];
  const decay = S.meta.unstamp_glow_decay_frames;

  // ---- nebula -------------------------------------------------------------------------------
  let neb = colour('lavender');
  if (st.nebula.length) {
    const e = st.nebula[st.nebula.length - 1];
    const prev = st.nebula.length > 1 ? colour(st.nebula[st.nebula.length - 2].colour) : colour(e.colour);
    const t = e.fade_frames > 0 ? clamp01((f - e.frame) / e.fade_frames) : 1;
    neb = mixHex(prev, colour(e.colour), t);
  }
  items.push({ kind: 'nebula', layer: 'nebula', id: 'nebula', colour: neb, centre: G.nebula.centre, radius: G.nebula.radius, peak: G.nebula.peak });

  // ---- the passes (lead digit accents, dims) --------------------------------------------------
  const dimAlpha = Object.create(null);     // k -> alpha factor
  const leadStyle = Object.create(null);    // k -> {colour, glow}
  for (const p of st.passes) {
    const d = p.dim_rows;
    let a = 1;
    if (f >= d.start && f < d.end) a = d.alpha;
    else if (f >= d.end && f < d.end + 12) a = lerp(d.alpha, 1, smooth((f - d.end) / 12));
    if (a < 1) for (const k of d.rows) dimAlpha[k] = Math.min(dimAlpha[k] == null ? 1 : dimAlpha[k], a);
    const fade = p.band.fade;
    let t = 0;
    if (f >= fade[0]) t = clamp01((f - fade[0]) / (fade[1] - fade[0]));
    const col = mixHex(colour(p.colour), colour('starlight'), t);
    const glow = t >= 1 ? null : bloomGlow(f - p.lead_digit_bloom.start, p.lead_digit_bloom.glow);
    leadStyle[p.k] = { colour: col, glow, digits: p.digits_left_of_rule };
  }

  // ---- the fold's fade of the old page ---------------------------------------------------------
  let pageFade = 1;
  if (st.fold) {
    const fo = st.fold.fade_out;
    pageFade = f >= fo.end ? 0 : 1 - clamp01((f - fo.start) / (fo.end - fo.start));
  }

  // ---- newest row (for the rules' bottom) -------------------------------------------------------
  let newest = 0;
  for (const k in st.rows) newest = Math.max(newest, +k);
  const ruleBottom = newest ? rowBase(newest) + 12 : G.rules.seconds.bottom_before_rows;

  // ---- bands (page layer) ------------------------------------------------------------------------
  for (const p of st.passes) {
    const b = p.band;
    let a = 0;
    if (f >= b.fade[1]) continue;
    if (f >= b.fade[0]) a = b.alpha * (1 - clamp01((f - b.fade[0]) / (b.fade[1] - b.fade[0])));
    else a = b.alpha * clamp01((f - b.fill[0]) / (b.fill[1] - b.fill[0]));
    if (a <= 0) continue;
    const box = { x0: b.x[0], y0: b.y[0], x1: b.x[1], y1: b.y[1] };
    items.push({ kind: 'band', layer: 'band', id: 'band_' + p.ref, ref: p.ref, k: p.k, colour: colour(p.colour), alpha: round3(a), box, screenBox: boxToScreen(cam, box) });
  }

  // ---- rules (page layer) -------------------------------------------------------------------------
  for (const ref of ['seconds', 'people']) {
    const evs = st.rules[ref];
    if (!evs.length) continue;
    const g = G.rules[ref];
    let alpha = g.alpha_rest, y1 = ruleBottom, gone = false;
    for (const e of evs) {
      if (e.action === 'drop') { const t = clamp01((f - e.frame) / (e.end - e.frame)); y1 = lerp(g.y_top, ruleBottom, smooth(t)); }
      else if (e.action === 'full') { if (f < e.end) alpha = g.alpha_pass; }
      else if (e.action === 'fade') { const t = clamp01((f - e.frame) / (e.end - e.frame)); if (t >= 1) gone = true; else alpha = alpha * (1 - t); }
      else if (e.action === 'pulse_dim') { const t = clamp01((f - e.frame) / (e.end - e.frame)); alpha = Math.max(alpha, g.alpha_rest + (e.alpha_peak - g.alpha_rest) * Math.sin(Math.PI * t)); }
    }
    if (gone || y1 <= g.y_top) continue;
    // the pulse of light down the rule
    let pulse = null;
    for (const p of st.passes) {
      if (p.ref !== ref) continue;
      const pu = p.pulse;
      if (f >= pu.start && f < pu.end + 6) {
        const t = clamp01((f - pu.start) / (pu.end - pu.start));
        const y = lerp(pu.from_y, pu.to_y, smooth(t));
        const a = f < pu.end ? 1 : 1 - (f - pu.end) / 6;
        pulse = { y: round3(y), alpha: round3(a), len: 56 };
      }
    }
    const box = { x0: g.x - 1.5, y0: g.y_top, x1: g.x + 1.5, y1 };
    items.push({ kind: 'rule', layer: 'rule', id: 'rule_' + ref, ref, x: g.x, y0: g.y_top, y1: round3(y1), colour: colour(g.colour),
      alpha: round3(alpha), pulse, box, screenBox: boxToScreen(cam, box) });
  }

  // ---- the references (page layer, typed) ---------------------------------------------------------
  const refStyle = { ref_seconds_label: 'ref_seconds_number', ref_people_label: 'ref_people_number' };
  const typedNumber = (id, t, vis, layer) => {
    const ev = st.text[id].type;
    const ref = S.numbers.references[t.ref];
    const gfs = ev ? ev.group_frames : null;
    const groupState = g => {
      if (!gfs) return { alpha: 1, glow: vis.glow };
      if (f < gfs[g]) return null;
      return { alpha: 1, glow: vis.ghost ? vis.glow : bloomGlow(f - gfs[g]) };
    };
    const complete = !gfs || f >= ev.end;
    return numberLines({
      id, layer: layer || t.layer, groups: ref.groups, lineGroups: t.line_groups, baselines: t.baselines || [t.baseline],
      size: t.size, colourHex: colour(t.colour), sig: t.sig, prefix: t.prefix, xRight: t.x, face: t.face,
      alpha: vis.alpha, glow: vis.glow, ink: vis.ink, ghost: vis.ghost, role: t.group, ref: t.ref, groupState, complete,
    });
  };
  for (const id of ['ref_seconds_label', 'ref_people_label', 'ref_seconds_number', 'ref_people_number']) {
    const t = S.texts[id], vis = visOf(st.text[id], f);
    if (!vis) continue;
    if (pageFade <= 0) continue;
    vis.alpha *= pageFade;
    if (t.ref) items.push(...typedNumber(id, t, vis));
    else items.push(wordItem(id, t, vis));
  }

  // ---- rows and cells (page layer) -------------------------------------------------------------------
  const foldActive = st.fold && f >= st.fold.frame;
  for (const kk in st.rows) {
    const k = +kk, row = S.numbers.rows[k - 1], landed = st.rows[k].frame;
    if (foldActive && (k === 52 || pageFade <= 0)) continue;
    const dim = dimAlpha[k] == null ? 1 : dimAlpha[k];
    const peak = st.rows[k].muted ? 0.8 : null;
    const glow = bloomGlow(f - landed, peak);
    const ls = leadStyle[k];
    const digitStyle = ls ? (i => (i < ls.digits ? { colour: ls.colour, glow: ls.glow } : null)) : null;
    items.push(...numberLines({
      id: 'row' + k, layer: 'page', groups: row.groups, lineGroups: [row.groups.length], baselines: [rowBase(k)],
      size: S.meta.row_style.size, colourHex: colour('starlight'), sig: null, prefix: null, xRight: G.spine_x, face: 'mono',
      alpha: round3(dim * pageFade), glow, ink: true, ghost: false, role: 'row', k, digitStyle,
    }));
    const ce = st.cells[k];
    if (ce) {
      const cs = S.meta.cell_style;
      items.push(placeItem({
        kind: 'text', layer: 'page', id: 'cell' + k, line: 0, text: row.cell_text, face: cs.face, size: cs.size, bakeSize: cs.size, mono: true,
        x: cs.x, baseline: rowBase(k), align: 'left', colour: colour(cs.colour), alpha: round3(dim * pageFade), glow: bloomGlow(f - ce.frame),
        ink: true, ghost: false, role: 'cell', k, runs: [{ text: row.cell_text, x: cs.x, colour: colour(cs.colour), alpha: 1, glow: bloomGlow(f - ce.frame) }],
      }));
    }
  }

  // ---- cards: the deck, flights, resting cards ------------------------------------------------------
  const cards = [];
  const hookBox = G.deck.hook, headBox = G.deck.head;
  const cardItem = (id, x, y, w, h, bake, number, numSize, numGlow, extra) => {
    const sc = w / G.deck[bake].w;
    const cx = x + w / 2, cy = y + h / 2;
    const glowPad = 10 * sc;
    const it = Object.assign({
      kind: 'card', layer: 'card', id, x: round3(x), y: round3(y), w: round3(w), h: round3(h), bake, scale: round3(sc),
      number: number == null ? null : String(number), numSize: round3(numSize), numBake: G.deck[bake].num_size,
      numX: round3(cx), numBaseline: round3(cy + 0.365 * numSize), numGlow: round3(numGlow), alpha: 1, face: true, ghost: false,
      edgeGlow: S.meta.stamp_bloom.rest_glow,
    }, extra || {});
    it.box = { x0: it.x - glowPad, y0: it.y - glowPad, x1: it.x + it.w + glowPad, y1: it.y + it.h + glowPad };
    it.screenBox = it.box;
    return it;
  };
  const D = st.deck;
  if (D.box === 'hook') {
    let numGlow = S.meta.stamp_bloom.rest_glow;
    if (D.numOn != null && D.reads != null) numGlow = bloomGlow(f - D.numOn);
    let edgeGlow = S.meta.stamp_bloom.rest_glow;
    if (D.ret && f >= D.ret.frame) edgeGlow = bloomGlow(f - D.ret.frame);
    cards.push(cardItem('deck', hookBox.x, hookBox.y, hookBox.w, hookBox.h, 'hook', D.reads, hookBox.num_size, numGlow, { edgeGlow: round3(edgeGlow) }));
  } else if (D.box === 'head') {
    const tr = D.travel;
    if (f < tr.end) {
      const p = easeOutExp((f - tr.frame) / (tr.end - tr.frame), 4);
      const x = lerp(hookBox.x, headBox.x, p), y = lerp(hookBox.y, headBox.y, p);
      const w = lerp(hookBox.w, headBox.w, p), h = lerp(hookBox.h, headBox.h, p);
      cards.push(cardItem('deck', x, y, w, h, 'hook', D.reads, lerp(hookBox.num_size, headBox.num_size, p), S.meta.stamp_bloom.rest_glow, { travelling: true }));
    } else {
      let numGlow = S.meta.stamp_bloom.rest_glow;
      if (D.bloom && f >= D.bloom.frame) {
        const d = f - D.bloom.frame, span = D.bloom.end - D.bloom.frame;
        numGlow = d < 2 ? S.meta.stamp_bloom.peak_glow : S.meta.stamp_bloom.rest_glow + (S.meta.stamp_bloom.peak_glow - S.meta.stamp_bloom.rest_glow) * (1 - smooth((d - 2) / (span - 2)));
      }
      cards.push(cardItem('deck', headBox.x, headBox.y, headBox.w, headBox.h, 'head', D.reads, headBox.num_size, numGlow));
    }
  } else if (D.leave && f < D.leave.frame + decay) {
    const d = f - D.leave.frame;
    cards.push(cardItem('deck', headBox.x, headBox.y, headBox.w, headBox.h, 'head', null, headBox.num_size, 0, { face: false, ghost: true, edgeGlow: round3(S.meta.stamp_bloom.rest_glow * (1 - d / decay)) }));
  }
  const cellCentre = k => (k === 1 ? [G.spine_x - S.numbers.rows[0].width48 / 2, rowBase(1) - 0.365 * 48] : [headBox.cx, rowBase(k) - 0.365 * 48]);
  for (const fl of st.flights) {
    const k = fl.k;
    if (f < fl.land) {
      const p = easeOutExp((f - fl.frame) / fl.duration, 5);
      const [dx, dy] = cellCentre(k);
      const [sx, sy] = toScreen(cam, dx, dy);
      const cx = lerp(headBox.cx, sx, p), cy = lerp(headBox.cy, sy, p);
      const sc = lerp(1, cam.s, p);
      const w = headBox.w * sc, h = headBox.h * sc;
      cards.push(cardItem('card' + k, cx - w / 2, cy - h / 2, w, h, 'head', fl.reads, headBox.num_size * sc, S.meta.stamp_bloom.rest_glow, { k, flying: true }));
    } else if (fl.becomes === 'card' && f < fl.cell) {
      const [dx, dy] = cellCentre(k);
      const [sx, sy] = toScreen(cam, dx, dy);
      const w = headBox.w * cam.s, h = headBox.h * cam.s;
      cards.push(cardItem('card' + k, sx - w / 2, sy - h / 2, w, h, 'head', fl.reads, headBox.num_size * cam.s, S.meta.stamp_bloom.rest_glow, { k, resting: true }));
    } else if (fl.becomes === 'card' && f < fl.cell + decay) {
      const d = f - fl.cell;
      const [dx, dy] = cellCentre(k);
      const [sx, sy] = toScreen(cam, dx, dy);
      const w = headBox.w * cam.s, h = headBox.h * cam.s;
      cards.push(cardItem('card' + k, sx - w / 2, sy - h / 2, w, h, 'head', null, headBox.num_size * cam.s, 0, { k, face: false, ghost: true, edgeGlow: round3(S.meta.stamp_bloom.rest_glow * (1 - d / decay)) }));
    }
  }
  items.push(...cards);

  // ---- screen-layer texts ---------------------------------------------------------------------------
  for (const id in S.texts) {
    const t = S.texts[id];
    if (t.layer !== 'screen' || id === 'deck52' || t.group === 'reference') continue;
    const tst = st.text[id];
    if (!tst) continue;
    if (id === 'lead') {
      const vis = visOf(tst, f);
      if (!vis) continue;
      const sep = t.separator, pieces = t.pieces;
      const runs = [];
      let prefix = '';
      for (let i = 0; i < tst.pieces.length; i++) {
        const pe = tst.pieces[i];
        const x = t.x + measure(prefix, t.face, t.size).width;
        runs.push({ text: pieces[i], x, colour: colour(t.colour), alpha: 1, glow: vis.ghost ? vis.glow : bloomGlow(f - pe.frame) });
        prefix += pieces[i] + sep;
      }
      items.push(placeItem({
        kind: 'text', layer: 'screen', id, line: 0, text: tst.pieces[tst.pieces.length - 1].text_so_far, face: t.face, size: t.size, bakeSize: t.size,
        mono: true, x: t.x, baseline: t.baseline, align: 'left', colour: colour(t.colour), alpha: vis.alpha, glow: vis.glow, ink: vis.ink, ghost: vis.ghost,
        runs, role: t.group,
      }));
      continue;
    }
    if (id === 'block') {
      const vis = visOf(tst, f);
      if (!vis) continue;
      const ref = S.numbers.references[t.ref];
      items.push(...numberLines({
        id, layer: 'screen', groups: ref.groups, lineGroups: t.line_groups, baselines: t.baselines, size: t.size, colourHex: colour(t.colour),
        sig: null, prefix: null, xRight: t.x, face: t.face, alpha: vis.alpha, glow: vis.glow, ink: vis.ink, ghost: vis.ghost, role: t.group, ref: t.ref,
      }));
      continue;
    }
    const vis = visOf(tst, f);
    if (!vis) continue;
    if (t.ref) items.push(...typedNumber(id, t, vis, 'screen'));
    else items.push(wordItem(id, t, vis));
  }

  // ---- the fold: row 52's groups travelling into the block ------------------------------------------------
  if (st.fold && f < st.fold.end) {
    const fo = st.fold, from = fo.from, to = fo.to, blk = S.texts[to.id];
    const camF = { s: from.page_scale, tx: from.page_tx, ty: from.page_ty };
    const row = S.numbers.rows[51];
    const adv0 = S.geometry.mono.advance_em * from.size, gap0 = S.geometry.mono.group_gap_em * from.size;
    const adv1 = S.geometry.mono.advance_em * to.size, gap1 = S.geometry.mono.group_gap_em * to.size;
    const w0 = row.ndigits * adv0 + (row.groups.length - 1) * gap0;
    let x0 = from.x_right - w0;
    const starts = [];
    for (const g of row.groups) { starts.push(x0); x0 += g.length * adv0 + gap0; }
    let gi = 0;
    for (let li = 0; li < to.line_groups.length; li++) {
      const n = to.line_groups[li];
      const gs = row.groups.slice(gi, gi + n);
      const w1 = gs.reduce((a, g) => a + g.length, 0) * adv1 + (n - 1) * gap1;
      let x1 = to.x_right - w1;
      // each group rises on its own (staggered) clock; a line fans out and grows on its leading group's
      // clock, so the groups of one line stay one non-overlapping interval throughout
      const tLine = clamp01((f - fo.group_starts[gi]) / (fo.end - fo.group_starts[gi]));
      const ex = smooth((tLine - 0.35) / 0.65);
      for (let j = 0; j < n; j++) {
        const g = gi + j, text = row.groups[g];
        const t = clamp01((f - fo.group_starts[g]) / (fo.end - fo.group_starts[g]));
        const ey = smooth(t);
        const [sx, sy] = toScreen(camF, starts[g], from.baseline);
        const size = lerp(from.size * from.page_scale, to.size, ex);
        const x = lerp(sx, x1, ex), y = lerp(sy, to.baselines[li], ey);
        const col = mixHex(colour('starlight'), colour(blk.colour), t);
        items.push(placeItem({
          kind: 'text', layer: 'screen', id: 'fold', line: g, text, face: blk.face, size: round3(size), bakeSize: to.size, mono: true,
          x: round3(x), baseline: round3(y), align: 'left', colour: col, alpha: 1, glow: S.meta.stamp_bloom.rest_glow, ink: true, ghost: false,
          runs: [{ text, x: round3(x), colour: col, alpha: 1, glow: S.meta.stamp_bloom.rest_glow, group: g }], role: 'fold', ref: 'factorial', foldT: round3(t),
          glowMix: { from: colour('starlight'), to: colour(blk.colour), t: round3(t) },
        }));
        x1 += text.length * adv1 + gap1;
      }
      gi += n;
    }
  }

  // screen boxes for page items
  for (const it of items) {
    if (it.kind !== 'text') continue;
    it.screenBox = it.layer === 'page' ? boxToScreen(cam, it.box) : it.box;
    it.screenBaseline = it.layer === 'page' ? round3(it.baseline * cam.s + cam.ty) : it.baseline;
    it.screenWidth = round3(it.layer === 'page' ? it.width * cam.s : it.width);
  }
  return { frame: f, page: { s: cam.s, tx: cam.tx, ty: cam.ty }, nebula: neb, items };
};

// ------------------------------------------------------------------------------------- stills
LN.STILLS = [
  ['frame0', 0], ['frame1', 1], ['hook_line4', 66], ['cover', 144], ['reset_travel', 192], ['target_seconds', 222],
  ['card1_flight', 237], ['card1_landed', 246], ['card2_resting', 306], ['teach', 330], ['card3_cell', 354], ['target_people', 402],
  ['card5', 432], ['card6_arrival', 462], ['card6_hold', 600], ['card10_arrival', 696], ['near_miss', 750], ['card11_arrival', 768],
  ['eleven_lead', 786], ['card11_hold', 900], ['reframe', 924], ['run_early', 1032], ['run_mid', 1110], ['run_late', 1160],
  ['lead_x4', 1200], ['lead_complete', 1278], ['wedge', 1330], ['fold_mid', 1416], ['block', 1440], ['note', 1500],
  ['beat1', 1720], ['beat2_typing', 1860], ['beat2', 1920], ['ending', 2040], ['deck_return', 2184], ['last', 2207],
];

// ------------------------------------------------------------------------------------- checks
const boxesMeet = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

LN.checkFrames = function () {
  const N = S.meta.frames, set = new Set();
  const add = v => { if (Number.isInteger(v)) for (const d of [-1, 0, 1]) { const x = v + d; if (x >= 0 && x < N) set.add(x); } };
  const addAll = v => { if (Array.isArray(v)) v.forEach(addAll); else if (v && typeof v === 'object') Object.values(v).forEach(addAll); else add(v); };
  for (const e of S.events) {
    add(e.frame); add(e.frame + 2); add(e.frame + S.meta.stamp_bloom.frames_at_peak + S.meta.stamp_bloom.settle_frames);
    if (e.kind === 'unstamp') add(e.frame + S.meta.unstamp_glow_decay_frames);
    for (const key of ['end', 'land', 'cell', 'hold_until', 'silence_until', 'knock_at', 'group_frames', 'group_starts', 'band', 'dim_rows', 'pulse', 'lead_digit_bloom', 'rule_full', 'fade_out']) {
      if (e[key] != null) addAll(e[key]);
    }
  }
  for (let f = 0; f < N; f += 6) set.add(f);
  return [...set].sort((a, b) => a - b);
};

LN.check = function (opts) {
  opts = opts || {};
  const G = S.geometry, safe = G.safe;
  const frames = opts.frames || LN.checkFrames();
  const viol = [];
  const v = (frame, check, id, detail) => viol.push({ frame, check, id, detail });
  const widths = {};
  const fitted = new Set();
  let itemCount = 0, textCount = 0;
  const digitsOf = s => s.replace(/[^0-9]/g, '');
  const deckReadsExpected = f => {
    let reads = null;
    for (const e of S.events) { if (e.frame > f) break; if (e.kind === 'stamp' && e.id === 'deck52') reads = 52; if (e.kind === 'flight') reads = e.deck_after || null; if (e.kind === 'deck_leave' || e.kind === 'deck_return') reads = null; }
    return reads;
  };
  for (const f of frames) {
    const L = LN.layout(f);
    itemCount += L.items.length;
    const texts = L.items.filter(i => i.kind === 'text' && !i.ghost);
    textCount += texts.length;
    const cards = L.items.filter(i => i.kind === 'card' && !i.ghost);
    for (const it of texts) {
      const sb = it.screenBox;
      const key = it.id + (it.line ? '#' + it.line : '');
      if (it.fitted) fitted.add(it.id);
      if (!(it.id === 'fold')) widths[key] = Math.max(widths[key] || 0, Math.round(it.screenWidth * 10) / 10);
      if (it.screenBaseline > safe.baseline_max_y) v(f, 'baseline', key, `baseline ${it.screenBaseline}`);
      if (sb.y1 > safe.ink_max_y) v(f, 'ink_floor', key, `ink bottom ${round3(sb.y1)}`);
      if (sb.x1 > safe.button_column.x_min && sb.y1 > safe.button_column.y[0] && sb.y0 < safe.button_column.y[1]) v(f, 'button_column', key, `x1 ${round3(sb.x1)} y ${round3(sb.y0)}-${round3(sb.y1)}`);
      if (sb.x0 < safe.x_min || sb.x1 > safe.x_max) v(f, 'x_bounds', key, `x ${round3(sb.x0)}-${round3(sb.x1)}`);
      if (it.screenWidth > safe.max_text_width + 1e-6) v(f, 'width', key, `width ${round3(it.screenWidth)} on screen`);
      // number strings
      if (it.role === 'row' && it.complete) { const d = digitsOf(it.digitsShown); if (d !== S.numbers.rows[it.k - 1].digits) v(f, 'digits', key, 'row digits differ'); }
      if (it.role === 'cell') { if (it.text !== S.numbers.rows[it.k - 1].cell_text) v(f, 'digits', key, 'cell text differs'); }
    }
    // multi-line / typed numbers complete: concatenation equals the reference
    const byId = {};
    for (const it of texts) if (it.ref && it.role !== 'row' && it.role !== 'fold') (byId[it.id] || (byId[it.id] = [])).push(it);
    for (const id in byId) {
      const parts = byId[id].sort((a, b) => a.line - b.line);
      if (!parts.every(p => p.complete)) continue;
      const t = S.texts[id];
      const want = S.numbers.references[t.ref].digits;
      const got = digitsOf(parts.map(p => p.digitsShown).join(''));
      const linesWant = t.line_groups.length;
      if (parts.length === linesWant && got !== want) v(f, 'digits', id, `typed ${got.length} digits, story ${want.length}`);
      const r = S.numbers.references[t.ref];
      if (r.exponent != null && r.ndigits !== r.exponent + 1) v(f, 'digits', id, 'approximate figure has the wrong digit count');
    }
    const foldParts = texts.filter(i => i.role === 'fold').sort((a, b) => a.line - b.line);
    if (foldParts.length && foldParts.map(p => p.text).join('') !== S.numbers.references.factorial.digits) v(f, 'digits', 'fold', 'fold groups differ from 52 factorial');
    // the deck's number
    const deck = cards.find(c => c.id === 'deck');
    const want = deckReadsExpected(f);
    const got = deck ? (deck.number == null ? null : +deck.number) : null;
    if (got !== want) v(f, 'digits', 'deck', `deck reads ${got}, expected ${want}`);
    // overlaps on the same layer (text vs text)
    for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i], b = texts[j];
      if (a.layer !== b.layer) continue;
      if (boxesMeet(a.box, b.box)) v(f, 'overlap', a.id + (a.line ? '#' + a.line : '') + ' & ' + b.id + (b.line ? '#' + b.line : ''), `${a.layer}: ${JSON.stringify(a.box)} ${JSON.stringify(b.box)}`);
    }
    // rules and bands touch only the rows whitelisted to cross them, and no label
    for (const r of L.items.filter(i => i.kind === 'rule' || i.kind === 'band')) {
      const minK = r.ref === 'people' ? 6 : 11;
      for (const t of texts) {
        if (t.layer !== 'page') continue;
        if (!boxesMeet(r.box, t.box)) continue;
        if (t.role === 'row' && t.k >= minK) continue;
        if (t.role === 'cell') { v(f, 'rule_touch', r.id, `touches cell ${t.k}`); continue; }
        v(f, 'rule_touch', r.id, `touches ${t.id}`);
      }
    }
    // a card never intersects a screen-layer text box
    for (const c of cards) for (const t of texts) {
      if (t.layer !== 'screen') continue;
      if (boxesMeet(c.box, t.box)) v(f, 'card_vs_text', c.id, `meets ${t.id}`);
    }
    // nothing outside x 132-948 (cards, rules, bands)
    for (const it of L.items) {
      if (it.kind === 'text' || it.ghost || !it.screenBox) continue;
      const sb = it.screenBox;
      if (sb.x0 < safe.x_min || sb.x1 > safe.x_max) v(f, 'x_bounds', it.id, `x ${round3(sb.x0)}-${round3(sb.x1)}`);
      if (sb.x1 > safe.button_column.x_min && sb.y1 > safe.button_column.y[0] && sb.y0 < safe.button_column.y[1]) v(f, 'button_column', it.id, `x1 ${round3(sb.x1)}`);
    }
    // the cover frame's key text
    if (f === S.meta.loop.cover_frame) {
      for (const t of texts) if (t.screenBox.y0 < safe.cover_key_text_y[0] || t.screenBox.y1 > safe.cover_key_text_y[1]) v(f, 'cover', t.id, `y ${round3(t.screenBox.y0)}-${round3(t.screenBox.y1)}`);
      if (texts.length < 6) v(f, 'cover', 'cover', `only ${texts.length} text items on the cover`);
    }
  }
  // layout(0) equals layout(2207) item for item
  const a = LN.layout(0), b = LN.layout(S.meta.frames - 1);
  const strip = L => JSON.stringify({ page: L.page, nebula: L.nebula, items: L.items });
  if (strip(a) !== strip(b)) v(S.meta.frames - 1, 'loop', 'layout', 'layout(0) differs from layout(last)');
  const camera = LN.checkCamera(v);
  return { frames_checked: frames.length, items_total: itemCount, text_items_total: textCount, violations: viol, widths, fitted: [...fitted], loop_identical: strip(a) === strip(b), camera };
};

/* The run's camera, frame by frame: still (s = 1, ty = 0) through camera.still_until; from there to the fold
   s never increases and ty never decreases (one direction only, no reversal), with no step (|ds| per frame
   bounded) and no kink (the per-frame change of ds bounded: C1); settled (constant) from settled_at to the
   fold; at every landing from row 13 the row fits the 660 px span and lands between y 1100 and 1240 on
   screen; ty = pivot (1 - s) at every frame. Reports the measured extremes. */
LN.checkCamera = function (v) {
  const C = S.camera, G = S.geometry;
  const still = C.still_until, end = C.still_after, settled = C.settled_at != null ? C.settled_at : end;
  const eps = 1e-9, maxStepS = 0.02, maxAccelS = 0.005, maxStepTy = 5, maxAccelTy = 1.5;
  let prev = null, prevDs = null, prevDty = null;
  const ext = { max_ds: 0, max_d2s: 0, max_dty: 0, max_d2ty: 0, s_end: null, ty_end: null, first_moving_frame: null, last_moving_frame: null };
  for (let f = 0; f <= end; f++) {
    const c = cameraAt(f);
    if (f <= still && (c.s !== 1 || c.ty !== 0)) v(f, 'camera', 'still', `s ${c.s} ty ${c.ty} before the run`);
    if (f >= settled && (Math.abs(c.s - C.s_end) > 1e-6)) v(f, 'camera', 'settled', `s ${c.s} after settling`);
    if (C.pivot_y != null && Math.abs(c.ty - C.pivot_y * (1 - c.s)) > 1e-6) v(f, 'camera', 'pivot', `ty ${c.ty} is not pivot (1 - s)`);
    if (prev) {
      const ds = c.s - prev.s, dty = c.ty - prev.ty;
      if (ds > eps) v(f, 'camera', 'monotone_s', `s rises ${round3(prev.s)} -> ${round3(c.s)}`);
      if (dty < -eps) v(f, 'camera', 'monotone_ty', `ty falls ${round3(prev.ty)} -> ${round3(c.ty)}`);
      if (Math.abs(ds) > maxStepS) v(f, 'camera', 'step_s', `ds ${ds}`);
      if (Math.abs(dty) > maxStepTy) v(f, 'camera', 'step_ty', `dty ${dty}`);
      if (prevDs !== null) {
        if (Math.abs(ds - prevDs) > maxAccelS) v(f, 'camera', 'kink_s', `d2s ${ds - prevDs}`);
        if (Math.abs(dty - prevDty) > maxAccelTy) v(f, 'camera', 'kink_ty', `d2ty ${dty - prevDty}`);
        ext.max_d2s = Math.max(ext.max_d2s, Math.abs(ds - prevDs)); ext.max_d2ty = Math.max(ext.max_d2ty, Math.abs(dty - prevDty));
      }
      ext.max_ds = Math.max(ext.max_ds, Math.abs(ds)); ext.max_dty = Math.max(ext.max_dty, Math.abs(dty));
      if (Math.abs(ds) > eps) { if (ext.first_moving_frame === null) ext.first_moving_frame = f; ext.last_moving_frame = f; }
      prevDs = ds; prevDty = dty;
    }
    prev = c;
  }
  ext.s_end = prev.s; ext.ty_end = round3(prev.ty);
  const landings = [];
  for (const r of S.numbers.rows) {
    if (r.k < 13) continue;
    const c = cameraAt(r.land);
    const width = r.width48 * c.s, base = rowBase(r.k) * c.s + c.ty;
    if (width > C.fit_width_px + 0.01) v(r.land, 'camera', 'fit', `row ${r.k} is ${round3(width)} px wide on screen`);   // 0.01 px: the targets round s to 6 decimals
    if (base < 1100 || base > C.newest_baseline_screen_y + 0.01) v(r.land, 'camera', 'newest_row', `row ${r.k} lands at y ${round3(base)}`);
    landings.push({ k: r.k, frame: r.land, s: round3(c.s), width: round3(width), baseline: round3(base) });
  }
  for (const key of ['max_ds', 'max_d2s', 'max_dty', 'max_d2ty']) ext[key] = Math.round(ext[key] * 1e6) / 1e6;
  return { still_until: still, settled_at: settled, still_after: end, pivot_y: C.pivot_y, ...ext, landings };
};

return LN;
});
