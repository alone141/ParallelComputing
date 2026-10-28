/* Three Doors: layout(frame), a pure function of the frame number.

   Loaded by page.html (where render() draws what layout() returns) and usable from Node for tests.
   Everything here reads build/story.json (passed to init) and a measuring function; it never touches a
   clock, a random source or the DOM. The same layout() feeds the renderer and the checker, so the checks
   cannot drift from the picture.

   Items (all in screen pixels on the 1080 x 1920 frame):
     text   {id, face, size, bakeSize, baseline, runs: [{text, x, colour}], alpha, glow, ink, box, tracking}
     door   {id, index, x, y, w, h, colour, glow, alpha, open (0..1), panelAlpha, dim, prize: {alpha, glow} | null,
             shimmer}
   Glow: 0.5 is the rest halo baked into every glyph, 1.2 the stamp's bloom (the first two frames),
   smoothstepping back to 0.5 over 15 frames.
*/
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ThreeDoors = factory();
})(typeof self !== 'undefined' ? self : this, function () {
'use strict';

const TD = {};
let S = null;            // story.json
let measureRaw = null;   // (text, cssFont) -> {width, left, right, ascent, descent}
const mcache = new Map();
let byId = null;         // events by id (stamp / unstamp)
let rounds = null, sprint = null, tallyAt = null;

// ------------------------------------------------------------------------------------------ maths
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = v => clamp(v, 0, 1);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => { t = clamp01(t); return t * t * (3 - 2 * t); };
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
/* a deterministic hash in [0, 1) of two integers (the shimmer's flicker) */
function hash01(a, b) {
  let h = (a * 374761393 + b * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
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
TD.init = function (story, measureFn) {
  S = story; measureRaw = measureFn; mcache.clear();
  byId = Object.create(null);
  for (const e of S.events) {
    if (e.kind === 'stamp') (byId[e.id] = byId[e.id] || {}).stamp = e.frame;
    if (e.kind === 'unstamp') { const r = (byId[e.id] = byId[e.id] || {}); r.unstamp = e.frame; r.unstampFrames = e.frames; }
  }
  rounds = S.rounds; sprint = S.sprint;
  tallyAt = [];   // every tally frame with the counts shown from it (rounds 1-30), then the sprint ticks
  for (const r of rounds.slice(0, 30)) tallyAt.push({ frame: r.tally_f, stay: r.stay, switch: r.switch, round: r.k, outcome: r.outcome });
  for (let i = 1; i < sprint.count_by_frame.length; i++) {
    const c = sprint.count_by_frame[i];
    if (c > sprint.count_by_frame[i - 1]) {
      const r = rounds[c - 1];
      tallyAt.push({ frame: sprint.start + i, stay: r.stay, switch: r.switch, round: c, outcome: r.outcome, sprint: true });
    }
  }
  return TD;
};
TD.story = () => S;
TD.fontOf = fontOf;
TD.measure = measure;
TD.mixHex = mixHex;

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
/* a stamped text's {alpha, glow} at frame f, or null when it is absent */
function stampState(id, f) {
  const e = byId[id];
  if (!e || f < e.stamp) return null;
  if (e.unstamp != null && f >= e.unstamp) {
    const t = (f - e.unstamp) / e.unstampFrames;
    if (t >= 1) return null;
    return { alpha: 1 - smooth(t), glow: S.meta.stamp_bloom.rest_glow };
  }
  return { alpha: 1, glow: bloomGlow(f - e.stamp) };
}

// ------------------------------------------------------------------------------------------- text
/* A text item from runs [{text, colour}] set centred at x (or flush left at x), with its ink box. */
function textItem(id, faceKey, size, baseline, x, align, runs, st, opts) {
  opts = opts || {};
  const bake = opts.bakeSize || size;
  const tracking = opts.tracking || 0;
  const ms = runs.map(r => measure(r.text, faceKey, size));
  let width = 0;
  for (let i = 0; i < runs.length; i++) width += ms[i].width + (i < runs.length - 1 ? tracking : 0);
  let cx = align === 'centre' ? x - width / 2 : x;
  const out = [];
  let top = Infinity, bottom = -Infinity, left = Infinity, right = -Infinity;
  for (let i = 0; i < runs.length; i++) {
    out.push({ text: runs[i].text, x: round3(cx), colour: runs[i].colour, glow: runs[i].glow, alpha: runs[i].alpha });
    left = Math.min(left, cx - ms[i].left); right = Math.max(right, cx + ms[i].right);
    top = Math.min(top, baseline - ms[i].ascent); bottom = Math.max(bottom, baseline + ms[i].descent);
    cx += ms[i].width + tracking;
  }
  const g = glowRadius(size);
  return { kind: 'text', id, layer: 'screen', face: faceKey, size, bakeSize: bake, baseline: round3(baseline), runs: out,
           alpha: round3(st.alpha), glow: round3(st.glow), ink: true, width: round3(width),
           box: { x0: round3(left - g), y0: round3(top - g), x1: round3(right + g), y1: round3(bottom + g) } };
}
function glyphRuns(text, colourHex) { return [...text].map(ch => ({ text: ch, colour: colourHex })); }

// ------------------------------------------------------------------------------------- the doors
/* The state of the three doors at frame f: colour (lavender at rest, cyan when picked, mist when the host
   has opened it), open 0..1 (the panel slid up), the prize (alpha, glow) in one door, and the shimmer. */
function doorsAt(f) {
  const G = S.geometry.doors;
  const lav = colour('lavender'), cyan = colour('cyan'), mist = colour('mist');
  const B = S.meta.stamp_bloom;
  const doors = [];
  for (let d = 0; d < 3; d++) doors.push({ colour: lav, glow: B.rest_glow, open: 0, dim: 0, prize: null, shimmer: 0, pick: 0 });
  // a played round (1 to 30)?
  let r = null;
  for (const rr of rounds) { if (rr.start == null) break; if (f >= rr.start && f < rr.end) { r = rr; break; } }
  if (r) {
    const closing = f >= r.close_f ? smooth((f - r.close_f) / Math.max(1, r.end - r.close_f)) : 0;
    const keep = 1 - closing;
    // your pick
    if (f >= r.pick_f) {
      const D = doors[r.pick];
      D.pick = keep;
      D.colour = mixHex(lav, cyan, keep);
      D.glow = Math.max(D.glow, lerp(B.rest_glow, bloomGlow(f - r.pick_f), keep));
    }
    // the host's door
    if (f >= r.host_f) {
      const D = doors[r.host];
      const o = smooth((f - r.host_f) / G.open_frames);
      D.open = o * keep;
      D.dim = o * keep;
      D.colour = mixHex(D.colour, mist, D.dim * 0.8);
      D.glow = Math.max(D.glow, lerp(B.rest_glow, bloomGlow(f - r.host_f, 0.9), keep));
    }
    // the reveal: the two remaining doors open, the prize shows
    if (f >= r.reveal_f) {
      const o = smooth((f - r.reveal_f) / G.open_frames);
      for (let d = 0; d < 3; d++) if (d !== r.host) doors[d].open = Math.max(doors[d].open, o * keep);
      const pa = smooth((f - r.reveal_f - 4) / 8) * keep;
      if (pa > 0) doors[r.prize].prize = { alpha: round3(pa), glow: round3(lerp(B.rest_glow, bloomGlow(f - r.reveal_f - 4), keep)) };
    }
  } else if (f >= sprint.start && f <= sprint.end) {
    // the sprint: the doors shimmer with the speed of the count, tinted by the round in progress
    const i = f - sprint.start, C = sprint.count_by_frame;
    const vmax = 1.35;
    const v = (C[Math.min(C.length - 1, i + 1)] - C[Math.max(0, i - 1)]) / 2;
    const amp = clamp01(v / vmax);
    const c = C[i];
    const rr = rounds[Math.max(0, c - 1)];
    for (let d = 0; d < 3; d++) {
      const D = doors[d];
      const h1 = hash01(f, d), h2 = hash01(f + 7919, d), h3 = hash01(f + 104729, d);
      const isHost = d === rr.host, isPick = d === rr.pick, isPrize = d === rr.prize;
      D.shimmer = round3(amp);
      D.open = round3(amp * (isHost ? 0.55 + 0.35 * h1 : 0.15 + 0.35 * h1));
      D.dim = round3(amp * (isHost ? 0.6 : 0.2) * h2);
      const pickMix = amp * (isPick ? 0.55 + 0.4 * h2 : 0.12 * h3);
      D.pick = round3(pickMix);
      D.colour = mixHex(mixHex(lav, cyan, pickMix), mist, D.dim * 0.5);
      D.glow = round3(B.rest_glow + amp * 0.25 * h3);
      const pa = amp * (isPrize ? 0.35 + 0.45 * h3 : 0.08 * h1);
      D.prize = pa > 0.02 ? { alpha: round3(pa), glow: round3(B.rest_glow + 0.3 * amp * h2) } : null;
    }
  }
  const out = [];
  for (let d = 0; d < 3; d++) {
    const D = doors[d];
    out.push({ kind: 'door', id: 'door' + d, index: d, x: G.x[d] - G.w / 2, y: G.top, w: G.w, h: G.h, radius: G.radius,
               colour: D.colour, glow: round3(D.glow), alpha: 1, open: round3(D.open), dim: round3(D.dim), pick: round3(D.pick),
               prize: D.prize, shimmer: D.shimmer,
               box: { x0: G.x[d] - G.w / 2, y0: G.top, x1: G.x[d] + G.w / 2, y1: G.top + G.h } });
  }
  return out;
}

// ------------------------------------------------------------------------------- the scoreboard
function scoreboardAt(f) {
  const SB = S.geometry.scoreboard, T = S.timeline, B = S.meta.stamp_bloom;
  const fadeIn = smooth((f - T.board_in) / SB.fade_frames);
  const outEv = S.events.find(e => e.kind === 'scoreboard_out');
  const fadeOut = f >= outEv.frame ? smooth((f - outEv.frame) / outEv.frames) : 0;
  const alpha = fadeIn * (1 - fadeOut);
  if (alpha <= 0) return [];
  // the counts shown: the last tally at or before f
  let stay = 0, sw = 0, round = 0, last = null;
  for (const t of tallyAt) { if (t.frame <= f) { stay = t.stay; sw = t.switch; round = t.round; last = t; } else break; }
  const d = last ? f - last.frame : Infinity;
  const inSprint = f >= sprint.start && f <= sprint.end;
  const punch = (hit) => (hit && !inSprint && d < 10) ? 1 + 0.12 * Math.sin(Math.PI * d / 10) : 1;
  const glowOf = (hit) => inSprint ? B.rest_glow + 0.2 : (hit ? bloomGlow(d, 1.1) : B.rest_glow);
  const endBloom = f >= sprint.end && f < sprint.end + 20 ? bloomGlow(f - sprint.end, 1.1) : null;
  const payoffBloom = f >= T.payoff_in && f < T.payoff_in + 20 ? bloomGlow(f - T.payoff_in, 1.2) : null;
  const items = [];
  const cols = [
    { id: 'stay', label: S.texts.labels.stay, count: stay, colour: colour('cyan'), hit: last && last.outcome === 'stay' },
    { id: 'round', label: S.texts.labels.round, count: round, colour: colour('starlight'), hit: !!last },
    { id: 'switch', label: S.texts.labels.switch, count: sw, colour: colour('amber'), hit: last && last.outcome === 'switch' },
  ];
  const st = { alpha, glow: B.rest_glow };
  cols.forEach((c, i) => {
    const x = SB.x[i];
    items.push(textItem('label_' + c.id, 'medium', SB.label_size, SB.label_baseline, x, 'centre', glyphRuns(c.label, colour('mist')), st,
                        { tracking: SB.label_tracking_em * SB.label_size }));
    let g = glowOf(c.hit);
    if (endBloom != null) g = Math.max(g, endBloom);
    if (payoffBloom != null && c.id === 'switch') g = Math.max(g, payoffBloom);
    const size = SB.count_size * punch(c.hit);
    items.push(textItem('count_' + c.id, 'monoB', round3(size), SB.count_baseline, x, 'centre', [{ text: String(c.count), colour: c.colour }],
                        { alpha, glow: g }, { bakeSize: SB.count_size }));
  });
  return items;
}

// ------------------------------------------------------------------------------------------ layout
TD.layout = function (f) {
  if (!(f >= 0 && f < S.meta.frames)) throw new Error('frame out of range: ' + f);
  const T = S.timeline, Gh = S.geometry.hook, Gr = S.geometry.rules, Gp = S.geometry.payoff, X = S.texts;
  const items = [];
  // the hook
  for (let i = 0; i < 4; i++) {
    const st = stampState('hook' + (i + 1), f);
    if (!st) continue;
    const line = X.hook[i];
    let runs;
    if (line === X.hook_target) runs = [{ text: line, colour: colour('lavender') }];
    else if (line.startsWith(X.hook_you + ' ')) runs = [{ text: X.hook_you, colour: colour('cyan') }, { text: line.slice(X.hook_you.length), colour: colour('starlight') }];
    else runs = [{ text: line, colour: colour('starlight') }];
    items.push(textItem('hook' + (i + 1), 'display', Gh.size, Gh.baselines[i], Gh.x, Gh.align, runs, st));
  }
  { const st = stampState('subline', f);
    if (st) items.push(textItem('subline', 'regular', Gh.subline_size, Gh.subline_baseline, Gh.x, Gh.align, [{ text: X.subline, colour: colour('mist') }], st)); }
  // the rules
  for (let i = 0; i < 3; i++) {
    const st = stampState('rule' + (i + 1), f);
    if (!st) continue;
    X.rules[i].forEach((line, j) => {
      items.push(textItem('rule' + (i + 1) + (X.rules[i].length > 1 ? 'ab'[j] : ''), 'medium', Gr.size, Gr.baselines[i][j], Gr.x, Gr.align,
                          [{ text: line, colour: colour('mist_light') }], st));
    });
  }
  // the scoreboard
  for (const it of scoreboardAt(f)) items.push(it);
  // the payoff
  { const st = stampState('payoff', f);
    if (st) items.push(textItem('payoff', 'display', Gp.size, Gp.baseline, Gp.x, Gp.align, [{ text: X.payoff, colour: colour('starlight') }], st)); }
  // the doors
  for (const d of doorsAt(f)) items.push(d);
  // the nebula: lavender, amber through the hold and the payoff, lavender again for the loop
  const N = S.geometry.nebula;
  let neb = colour('lavender');
  const a0 = sprint.end, a1 = T.board_out;
  if (f >= a0 - N.crossfade_frames && f < a1 + N.crossfade_frames) {
    const tin = smooth((f - (a0 - N.crossfade_frames)) / N.crossfade_frames);
    const tout = f >= a1 ? smooth((f - a1) / N.crossfade_frames) : 0;
    neb = mixHex(colour('lavender'), colour('amber'), tin * (1 - tout));
  }
  return { frame: f, items, nebula: neb };
};

// ------------------------------------------------------------------------------------------ stills
TD.STILLS = [
  ['frame0', 0], ['hook_in', 70], ['cover', 120], ['hook_out', 165], ['rule1', 230], ['rules', 450], ['board', 515],
  ['r1_pick', 520], ['r1_host', 552], ['r1_reveal', 586], ['r1_tally', 596], ['r1_close', 630],
  ['r2_reveal', 706], ['r2_tally', 716], ['r3_reveal', 826], ['pause1', 930], ['pause1_close', 990],
  ['r4', 1004], ['r6', 1060], ['r12', 1160], ['r22', 1245], ['pause2', 1350], ['sprint_in', 1440], ['sprint_mid', 1566],
  ['sprint_late', 1690], ['hold', 1740], ['payoff', 1765], ['payoff_hold', 1900], ['payoff_out', 1992], ['ending', 2030],
  ['loop_from', 2088], ['last', 2099],
];

// ------------------------------------------------------------------------------------------- check
const overlap = (a, b, m) => !(a.x1 + m <= b.x0 || b.x1 + m <= a.x0 || a.y1 + m <= b.y0 || b.y1 + m <= a.y0);

TD.checkFrames = function () {
  const set = new Set();
  for (const e of S.events) for (const d of [-1, 0, 1]) { const f = e.frame + d; if (f >= 0 && f < S.meta.frames) set.add(f); }
  for (let f = 0; f < S.meta.frames; f += 6) set.add(f);
  set.add(S.meta.frames - 1);
  return [...set].sort((a, b) => a - b);
};

TD.check = function () {
  const V = [], band = S.geometry.band, T = S.timeline;
  const say = (frame, check, id, detail) => V.push({ frame, check, id, detail });
  const strings = new Set();
  for (const l of S.texts.hook) strings.add(l);
  strings.add(S.texts.subline); strings.add(S.texts.payoff);
  for (const r of S.texts.rules) for (const l of r) strings.add(l);
  for (const k in S.texts.labels) strings.add(S.texts.labels[k]);
  const frames = TD.checkFrames();
  let items = 0, textItems = 0;
  const fitted = {};
  const loop0 = JSON.stringify(TD.layout(0));
  let loopIdentical = true;
  for (const f of frames) {
    const L = TD.layout(f);
    const texts = L.items.filter(it => it.kind === 'text' && it.alpha > 0);
    const doors = L.items.filter(it => it.kind === 'door');
    items += L.items.length; textItems += texts.length;
    for (const it of texts) {
      const b = it.box;
      if (it.baseline > band.baseline_max) say(f, 'baseline', it.id, `baseline ${it.baseline} > ${band.baseline_max}`);
      if (b.y1 > band.ink_max) say(f, 'ink_bottom', it.id, `ink bottom ${b.y1} > ${band.ink_max}`);
      if (b.x0 < band.x0 || b.x1 > band.x1) say(f, 'band', it.id, `ink x ${b.x0}..${b.x1} outside ${band.x0}..${band.x1}`);
      if (b.x1 > band.button_column_x && b.y1 > band.button_column_y[0] && b.y0 < band.button_column_y[1]) say(f, 'button_column', it.id, 'ink in the button column');
      if (f === T.cover && (b.y0 < band.cover_y[0] || b.y1 > band.cover_y[1])) say(f, 'cover_crop', it.id, `ink y ${b.y0}..${b.y1} outside the 3:4 crop`);
      const s = it.runs.map(r => r.text).join('');
      if (!strings.has(s) && !/^\d+$/.test(s)) say(f, 'string', it.id, `"${s}" is not a story text`);
      if (it.width > band.x1 - band.x0) say(f, 'width', it.id, `width ${it.width} > ${band.x1 - band.x0}`);
      fitted[it.id] = Math.max(fitted[it.id] || 0, it.width);
      for (const d of doors) if (overlap(b, d.box, 6)) say(f, 'text_over_door', it.id, `text ink over ${d.id}`);
    }
    for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
      if (overlap(texts[i].box, texts[j].box, 2)) say(f, 'text_overlap', texts[i].id + '+' + texts[j].id, 'ink boxes overlap');
    }
    // the scoreboard sums and the pauses
    const cnt = {};
    for (const it of texts) if (it.id.startsWith('count_')) cnt[it.id.slice(6)] = parseInt(it.runs[0].text, 10);
    if ('stay' in cnt && cnt.stay + cnt.switch !== cnt.round) say(f, 'scoreboard_sum', 'scoreboard', `STAY ${cnt.stay} + SWITCH ${cnt.switch} != ROUND ${cnt.round}`);
    const pauses = [[T.pause1, [1, 2, 3]], [T.pause2, [10, 20, 30]], [[T.sprint[1], T.payoff_out], [100, 200, 300]]];
    for (const [[a, b], [st, sw, rd]] of pauses) if (f >= a && f < b && 'stay' in cnt && (cnt.stay !== st || cnt.switch !== sw || cnt.round !== rd))
      say(f, 'pause_counts', 'scoreboard', `${cnt.stay}/${cnt.switch}/${cnt.round} during a pause expecting ${st}/${sw}/${rd}`);
    // the doors
    const prizes = doors.filter(d => d.prize && d.prize.alpha > 0.5);
    if (prizes.length > 1 && !doors[0].shimmer) say(f, 'two_prizes', 'doors', 'the prize shows in more than one door');
    for (const d of doors) if (d.prize && d.prize.alpha > 0.5 && d.dim > 0.5 && !d.shimmer) say(f, 'prize_in_host_door', d.id, 'the prize shows in the host\'s opened door');
    if (f >= T.loop_from && JSON.stringify({ ...L, frame: 0 }) !== loop0) loopIdentical = false;
  }
  // the loop frames, every one
  for (let f = T.loop_from; f < S.meta.frames; f++) if (JSON.stringify({ ...TD.layout(f), frame: 0 }) !== loop0) { loopIdentical = false; say(f, 'loop', 'frame', 'differs from frame 0'); break; }
  // the round bookkeeping
  for (const r of rounds.slice(0, 30)) {
    if (r.host === r.pick || r.host === r.prize) say(r.start, 'round', 'round' + r.k, 'the host opened your door or the prize door');
    if ((r.outcome === 'stay') !== (r.prize === r.pick)) say(r.start, 'round', 'round' + r.k, 'outcome disagrees with the doors');
  }
  return { frames_checked: frames.length, items_total: items, text_items_total: textItems, loop_identical: loopIdentical, fitted, violations: V };
};

return TD;
});
