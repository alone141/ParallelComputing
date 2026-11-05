// 45 — a rhythm game that plays the intro of Gojira's "The Art of Dying" (2008).
// Everything is a pure function of film time t (seconds), so frames can be
// stepped deterministically by the recorder.

'use strict';

// ---------- music model ----------
const BPM = 140;
const STEP = 15 / BPM;                 // one sixteenth, 107.14 ms
const CYCLE = 45;                      // sixteenths per lap
const RESTS = new Set([4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44]);
const GROUPS = [5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3];  // 24 + 21
const GROUP_START = [];                // cycle position where each group begins
{ let p = 0; for (const g of GROUPS) { GROUP_START.push(p); p += g; } }
const LAPS = 8;                        // laps played before the end card
const END_STEP = CYCLE * LAPS;         // step 360: second realignment, last hit
const DRUM_IN = CYCLE * 4;             // step 180: first realignment, drum enters
const T0 = 1.5;                        // film time of music step 0
const END_HOLD = 3.3;                  // end card length after the last hit

function musicTime(t) { return t - T0; }
function stepTime(n) { return T0 + n * STEP; }

// ---------- layout (1080 x 1920) ----------
const W = 1080, H = 1920;
const HIT_Y = 1340;
const SPAWN_Y = 400;
const PLAY_TOP = 390, PLAY_BOT = 1470;   // clip region for notes, lines, brackets
const APPROACH = 1.6;                  // seconds from spawn to hit line
const SPEED = (HIT_Y - SPAWN_Y) / APPROACH;   // px per second
const PX_PER_STEP = SPEED * STEP;
const LANE = [
  { x0: 150, x1: 490, name: 'CLICK', color: '#FFB13B', dim: '#5a3d12' },
  { x0: 540, x1: 880, name: 'PULSE', color: '#4FC3FF', dim: '#13405a' },
];
const LANE_X0 = LANE[0].x0, LANE_X1 = LANE[1].x1;
const COL = {
  bg: '#06060b', lap: '#FFFFFF', drift: '#FF4D6D', ok: '#5CFF8A',
  text: '#ECECF4', muted: '#8A8AA3', five: '#FFB13B', three: '#FF7A59',
};
const FONT_D = '"Inter Display", "Inter", sans-serif';
const FONT = '"Inter", sans-serif';
const MONO = '"JetBrains Mono", monospace';

// ---------- notes ----------
// A note: { n: step index, lane: 0|1, kind: 'click'|'ghost'|'drum' }
const NOTES = [];
for (let n = 0; n <= END_STEP; n++) {
  const c = n % CYCLE;
  if (!RESTS.has(c)) NOTES.push({ n, lane: 0, kind: 'click' });
  if (n % 4 === 0) NOTES.push({ n, lane: 1, kind: n < DRUM_IN ? 'ghost' : 'drum' });
}
// rests after the last hit are never shown; the film ends on step 360.

function noteY(n, t) { return HIT_Y - (stepTime(n) - t) * SPEED; }

function hitsBefore(t) {
  // combo = every note whose time has passed
  const m = musicTime(t);
  if (m < 0) return 0;
  const n = Math.min(Math.floor(m / STEP + 1e-6), END_STEP);
  let count = 0;
  for (let i = 0; i <= n; i++) {
    if (!RESTS.has(i % CYCLE)) count++;
    if (i % 4 === 0) count++;
  }
  return count;
}

// ---------- helpers ----------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ease = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
const smooth = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
// 1 at t=t0 decaying to 0 over d
const pulse = (t, t0, d) => (t < t0 ? 0 : Math.max(0, 1 - (t - t0) / d));
// 0 before a, ramps to 1 by b, holds, ramps to 0 from c to d
function window_(t, a, b, c, d) {
  if (t < a || t > d) return 0;
  if (t < b) return smooth((t - a) / (b - a));
  if (t < c) return 1;
  return 1 - smooth((t - c) / (d - c));
}
function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r); ctx.closePath();
}
// shrink a font until the string fits maxW; font is like '600 36px Inter'
function fitFont(ctx, s, font, maxW) {
  const m = font.match(/^(.*?)(\d+)px(.*)$/);
  let size = Number(m[2]);
  for (;;) {
    const f = `${m[1]}${size}px${m[3]}`;
    ctx.font = f;
    if (ctx.measureText(s).width <= maxW || size <= 20) return f;
    size -= 1;
  }
}
function text(ctx, s, x, y, font, color, align = 'center', alpha = 1) {
  ctx.save(); ctx.globalAlpha *= alpha; ctx.font = font; ctx.fillStyle = color;
  ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore();
}

// ---------- captions (music time, seconds) ----------
const LAP_S = CYCLE * STEP;
const CAPTIONS = [
  { a: 0.6,  b: 4.6,  l1: 'Rim clicks, in groups of 5 and 3.', l2: 'Every group skips its last step.' },
  { a: 5.2,  b: 9.4,  l1: 'A lap is 45 steps. The pulse ticks every 4.', l2: '45 = 11 × 4 + 1, so the line lands 1 step late.' },
  { a: 10.0, b: 14.2, l1: 'Lap three: two steps late.', l2: 'The pulse never changes. The lap drifts.' },
  { a: 14.8, b: 18.8, l1: 'Three steps late.', l2: 'One more lap and the drift is a whole pulse.' },
  { a: 22.6, b: 27.6, l1: 'On the record the big drum enters exactly here,', l2: 'on the first realignment. Then it slips again.' },
  { a: 29.6, b: 33.6, l1: 'The drum rides the 4. The clicks ride the 45.', l2: 'They agree once every 180 steps.' },
  { a: 34.2, b: 38.2, l1: 'Fans call it 21/16 or 45/16.', l2: 'The drummer calls it 4/4, with a spare beat.' },
];

// ---------- drawing ----------
function drawBackground(ctx, t) {
  ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, W, H);
  // lanes
  for (const L of LANE) {
    const g = ctx.createLinearGradient(0, PLAY_TOP, 0, PLAY_BOT);
    g.addColorStop(0, 'rgba(255,255,255,0)');
    g.addColorStop(0.2, 'rgba(255,255,255,0.035)');
    g.addColorStop(1, 'rgba(255,255,255,0.06)');
    ctx.fillStyle = g; ctx.fillRect(L.x0, PLAY_TOP, L.x1 - L.x0, PLAY_BOT - PLAY_TOP);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(L.x0, PLAY_TOP + 60); ctx.lineTo(L.x0, PLAY_BOT);
    ctx.moveTo(L.x1, PLAY_TOP + 60); ctx.lineTo(L.x1, PLAY_BOT); ctx.stroke();
  }
  // lane names at the bottom of the lanes (decorative; below is app UI)
  for (const L of LANE) {
    text(ctx, L.name, (L.x0 + L.x1) / 2, HIT_Y + 95, `600 30px ${FONT_D}`, L.color, 'center', 0.8);
  }
}

function drawHitLine(ctx, t) {
  // receptors
  for (let i = 0; i < 2; i++) {
    const L = LANE[i];
    const last = lastHitTime(t, i);
    const glow = last == null ? 0 : pulse(t, last, 0.18);
    ctx.save();
    ctx.shadowColor = L.color; ctx.shadowBlur = 20 + 40 * glow;
    ctx.strokeStyle = L.color; ctx.lineWidth = 3 + 3 * glow; ctx.globalAlpha = 0.6 + 0.4 * glow;
    rrect(ctx, L.x0 + 18, HIT_Y - 24, L.x1 - L.x0 - 36, 48, 14); ctx.stroke();
    ctx.restore();
  }
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(LANE_X0 - 60, HIT_Y); ctx.lineTo(LANE_X1 + 30, HIT_Y); ctx.stroke();
  ctx.restore();
}

// time of the last note hit in a lane before t (null if none)
function lastHitTime(t, lane) {
  const m = musicTime(t);
  if (m < 0) return null;
  let n = Math.min(Math.floor(m / STEP + 1e-6), END_STEP);
  for (let i = n; i >= 0 && i > n - 8; i--) {
    if (lane === 1 && i % 4 === 0) return stepTime(i);
    if (lane === 0 && !RESTS.has(i % CYCLE)) return stepTime(i);
  }
  return null;
}

function drawGroups(ctx, t) {
  // brackets along the left of the click lane, one per 5 or 3 group, with
  // dashed "skipped" slots at each rest.
  const L = LANE[0];
  const m = musicTime(t);
  const nMin = Math.floor((m - 0.6) / STEP), nMax = Math.ceil((m + APPROACH + 0.3) / STEP);
  for (let lap = Math.floor(Math.max(0, nMin) / CYCLE); lap * CYCLE <= Math.min(nMax, END_STEP); lap++) {
    GROUPS.forEach((g, gi) => {
      const s = lap * CYCLE + GROUP_START[gi];
      const e = s + g - 1;               // the rest slot
      if (e < nMin || s > nMax || s > END_STEP) return;
      const yTop = noteY(s, t), yBot = noteY(e, t);
      const past = t > stepTime(e);
      const alpha = (past ? 0.25 : 1) * clamp((yTop - PLAY_TOP) / 110 + 1, 0, 1);
      const color = g === 5 ? COL.five : COL.three;
      ctx.save(); ctx.globalAlpha = alpha;
      // bracket
      ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.lineCap = 'round';
      const bx = L.x0 - 34;
      ctx.beginPath();
      ctx.moveTo(bx + 14, yTop - 18); ctx.lineTo(bx, yTop - 18); ctx.lineTo(bx, yBot + 18); ctx.lineTo(bx + 14, yBot + 18);
      ctx.stroke();
      text(ctx, String(g), bx - 36, (yTop + yBot) / 2, `700 54px ${FONT_D}`, color);
      // dashed empty slot for the skipped step
      ctx.setLineDash([10, 10]); ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.globalAlpha = alpha * 0.45;
      rrect(ctx, L.x0 + 40, yBot - 17, L.x1 - L.x0 - 80, 34, 10); ctx.stroke();
      ctx.restore();
    });
  }
}

function drawNotes(ctx, t) {
  const m = musicTime(t);
  for (const note of NOTES) {
    const y = noteY(note.n, t);
    if (y < PLAY_TOP - 40 || y > PLAY_BOT + 40) continue;
    const L = LANE[note.lane];
    const dt = t - stepTime(note.n);     // >0 after hit
    ctx.save();
    if (dt > 0) {
      // hit: explode outwards and fade
      const k = clamp(dt / 0.22, 0, 1);
      ctx.globalAlpha = 1 - k;
      const grow = 1 + 1.2 * k;
      const cx = (L.x0 + L.x1) / 2;
      ctx.strokeStyle = L.color; ctx.lineWidth = 4;
      ctx.shadowColor = L.color; ctx.shadowBlur = 30;
      const w = (L.x1 - L.x0 - 80) * grow, h = 34 * grow;
      rrect(ctx, cx - w / 2, HIT_Y - h / 2, w, h, 12 * grow); ctx.stroke();
      ctx.restore();
      continue;
    }
    // approaching
    const fadeIn = clamp((y - PLAY_TOP) / 110, 0, 1);
    ctx.globalAlpha = fadeIn;
    if (note.kind === 'ghost') {
      ctx.strokeStyle = L.color; ctx.lineWidth = 3; ctx.setLineDash([]);
      ctx.globalAlpha = fadeIn * 0.8;
      rrect(ctx, L.x0 + 40, y - 17, L.x1 - L.x0 - 80, 34, 10); ctx.stroke();
      ctx.fillStyle = L.color; ctx.globalAlpha = fadeIn * 0.18;
      rrect(ctx, L.x0 + 40, y - 17, L.x1 - L.x0 - 80, 34, 10); ctx.fill();
    } else if (note.kind === 'drum') {
      ctx.fillStyle = L.color; ctx.shadowColor = L.color; ctx.shadowBlur = 28;
      rrect(ctx, L.x0 + 28, y - 24, L.x1 - L.x0 - 56, 48, 14); ctx.fill();
      ctx.shadowBlur = 0; ctx.fillStyle = '#06060b';
      rrect(ctx, L.x0 + 150, y - 5, L.x1 - L.x0 - 300, 10, 5); ctx.fill();
      if (note.n === DRUM_IN) text(ctx, 'DRUM', (L.x0 + L.x1) / 2, y - 48, `700 30px ${FONT_D}`, L.color);
    } else {
      ctx.fillStyle = L.color; ctx.shadowColor = L.color; ctx.shadowBlur = 18;
      rrect(ctx, L.x0 + 40, y - 17, L.x1 - L.x0 - 80, 34, 10); ctx.fill();
    }
    ctx.restore();
  }
}

function drawLapLines(ctx, t) {
  for (let lap = 0; lap <= LAPS; lap++) {
    const n = lap * CYCLE;
    const y = noteY(n, t);
    if (y < PLAY_TOP - 40 || y > PLAY_BOT + 40) continue;
    const dt = t - stepTime(n);
    const alpha = dt > 0 ? clamp(1 - dt / 0.5, 0, 1) * 0.5 : clamp((y - PLAY_TOP) / 110, 0, 1);
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.strokeStyle = COL.lap; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(LANE_X0 - 70, y); ctx.lineTo(LANE_X1, y); ctx.stroke();
    // label: a tag sitting on the line at the right end of the pulse lane
    const label = lap === 0 ? 'START' : `LAP ${lap + 1}`;
    ctx.fillStyle = COL.lap;
    rrect(ctx, LANE_X1 - 118, y - 36, 118, 34, 8); ctx.fill();
    text(ctx, label, LANE_X1 - 59, y - 18, `700 24px ${FONT_D}`, '#06060b');
    // drift bracket in the pulse lane: from the previous pulse up to this line
    const off = n % 4;                  // 0,1,2,3
    const L = LANE[1];
    const yPulse = noteY(n - off, t);
    const x = L.x0 + 10;
    if (off > 0) {
      ctx.strokeStyle = COL.drift; ctx.lineWidth = 5; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x + 12, yPulse); ctx.lineTo(x, yPulse); ctx.lineTo(x, y); ctx.lineTo(x + 12, y); ctx.stroke();
      // shaded gap
      ctx.fillStyle = COL.drift; ctx.globalAlpha = alpha * 0.18;
      ctx.fillRect(L.x0 + 20, y, L.x1 - L.x0 - 40, yPulse - y);
      ctx.globalAlpha = alpha;
      text(ctx, `+${off}`, L.x1 - 70, (y + yPulse) / 2, `800 40px ${FONT_D}`, COL.drift);
    } else if (lap > 0) {
      text(ctx, 'ALIGNED', (L.x0 + L.x1) / 2 - 60, y - 18, `800 28px ${FONT_D}`, COL.ok);
    }
    ctx.restore();
  }
}

function currentLap(t) {
  const m = musicTime(t);
  if (m < 0) return 0;
  return Math.min(Math.floor(m / LAP_S + 1e-9), LAPS) + 1;  // 1-based; lap 9 = end
}

function drawHUD(ctx, t) {
  const m = musicTime(t);
  const lap = Math.max(1, currentLap(t));
  const lapShown = Math.min(lap, LAPS);
  const lapInSet = ((lapShown - 1) % 4) + 1;    // 1..4
  const lapStart = stepTime((lap - 1) * CYCLE);
  const pop = lap > 1 ? pulse(t, lapStart, 0.35) : 0;
  const off = ((lap - 1) * CYCLE) % 4;          // drift of this lap's line in steps
  // title row
  text(ctx, 'GOJIRA · THE ART OF DYING · 140 BPM', 540, 170, `600 30px ${FONT_D}`, COL.muted);
  // lap
  ctx.save(); ctx.translate(170, 270); ctx.scale(1 + 0.12 * pop, 1 + 0.12 * pop);
  text(ctx, 'LAP', 0, -42, `600 28px ${FONT_D}`, COL.muted, 'left');
  text(ctx, `${lapInSet}`, 0, 18, `800 92px ${FONT_D}`, COL.text, 'left');
  text(ctx, `/4`, 64, 30, `600 40px ${FONT_D}`, COL.muted, 'left');
  ctx.restore();
  // drift
  const driftCol = off === 0 ? COL.ok : COL.drift;
  text(ctx, 'LINE VS PULSE', 540, 228, `600 28px ${FONT_D}`, COL.muted);
  ctx.save(); ctx.translate(540, 290); ctx.scale(1 + 0.15 * pop, 1 + 0.15 * pop);
  text(ctx, off === 0 ? 'ALIGNED' : `+${off} STEP${off > 1 ? 'S' : ''}`, 0, 0, `800 52px ${FONT_D}`, driftCol);
  ctx.restore();
  // four little lap boxes showing the drift progression
  for (let i = 0; i < 4; i++) {
    const x = 455 + i * 48, y = 350;
    const filled = i < lapInSet;
    ctx.save();
    ctx.globalAlpha = filled ? 1 : 0.25;
    ctx.fillStyle = i === 0 ? COL.ok : COL.drift;
    rrect(ctx, x - 16, y - 8, 32, 16, 4); ctx.fill();
    ctx.restore();
  }
  // step counter (right, but inside x<880)
  const stepNow = m < 0 ? 0 : Math.min(Math.floor(m / STEP + 1e-6), END_STEP);
  const inLap = stepNow === END_STEP ? 0 : stepNow % CYCLE;
  text(ctx, 'STEP', 880, 228, `600 28px ${FONT_D}`, COL.muted, 'right');
  text(ctx, String(inLap).padStart(2, '0'), 880, 290, `700 60px ${MONO}`, COL.text, 'right');
  text(ctx, `of ${CYCLE}`, 880, 345, `500 24px ${MONO}`, COL.muted, 'right');
}

function drawCombo(ctx, t) {
  const combo = hitsBefore(t);
  if (combo === 0) return;
  const last = Math.max(lastHitTime(t, 0) ?? -1, lastHitTime(t, 1) ?? -1);
  const pop = pulse(t, last, 0.14);
  const cx = (LANE_X0 + LANE_X1) / 2, cy = 1000;
  ctx.save();
  ctx.globalAlpha = 0.6;
  ctx.translate(cx, cy); ctx.scale(1 + 0.18 * pop, 1 + 0.18 * pop);
  ctx.shadowColor = 'rgba(255,255,255,0.6)'; ctx.shadowBlur = 10 + 30 * pop;
  text(ctx, String(combo), 0, 0, `800 130px ${FONT_D}`, COL.text);
  ctx.restore();
  text(ctx, 'COMBO', cx, cy + 88, `700 30px ${FONT_D}`, COL.muted, 'center', 0.7);
}

function drawJudgement(ctx, t) {
  const combo = hitsBefore(t);
  if (combo === 0) return;
  const last = Math.max(lastHitTime(t, 0) ?? -1, lastHitTime(t, 1) ?? -1);
  const j = pulse(t, last, 0.3);
  if (j > 0) {
    const lane = (lastHitTime(t, 1) ?? -1) >= (lastHitTime(t, 0) ?? -1) ? 1 : 0;
    const L = LANE[lane];
    text(ctx, 'PERFECT', (L.x0 + L.x1) / 2, HIT_Y - 62 - 20 * (1 - j), `800 34px ${FONT_D}`, L.color, 'center', j);
  }
}

function drawCaptions(ctx, t) {
  const m = musicTime(t);
  for (const c of CAPTIONS) {
    const a = window_(m, c.a, c.a + 0.3, c.b - 0.3, c.b);
    if (a <= 0) continue;
    ctx.save(); ctx.globalAlpha = a;
    ctx.fillStyle = 'rgba(6,6,11,0.86)';
    rrect(ctx, 120, 400, 920, 160, 22); ctx.fill();
    text(ctx, c.l1, 580, 452, fitFont(ctx, c.l1, `600 42px ${FONT}`, 870), COL.text);
    text(ctx, c.l2, 580, 510, fitFont(ctx, c.l2, `500 38px ${FONT}`, 870), COL.muted);
    ctx.restore();
  }
}

function drawHook(ctx, t) {
  const a = window_(t, 0, 0.01, 2.3, 2.9);
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a;
  ctx.fillStyle = 'rgba(6,6,11,0.72)'; ctx.fillRect(0, 560, W, 560);
  text(ctx, 'NOW PLAYING', 540, 640, `600 30px ${FONT_D}`, COL.muted);
  text(ctx, 'THE ART OF DYING', 540, 700, `700 48px ${FONT_D}`, COL.text);
  text(ctx, 'THE PULSE', 540, 830, `800 120px ${FONT_D}`, COL.text);
  text(ctx, 'SLIPS.', 540, 950, `800 120px ${FONT_D}`, COL.drift);
  text(ctx, 'Every lap. Watch it come home.', 540, 1050, `500 40px ${FONT}`, COL.muted);
  ctx.restore();
}

function drawCelebration(ctx, t, n, big) {
  const t1 = stepTime(n);
  const k = t - t1;
  if (k < -0.001 || k > 3.2) return;
  const cx = 540, cy = 860;
  // flash
  const flash = pulse(t, t1, 0.2);
  ctx.save(); ctx.globalAlpha = 0.55 * flash; ctx.fillStyle = COL.ok; ctx.fillRect(0, 0, W, H); ctx.restore();
  // rings
  for (let i = 0; i < 3; i++) {
    const kk = clamp((k - i * 0.12) / 1.1, 0, 1);
    if (kk <= 0 || kk >= 1) continue;
    ctx.save(); ctx.globalAlpha = (1 - kk) * 0.8; ctx.strokeStyle = i === 1 ? COL.text : COL.ok; ctx.lineWidth = 10 * (1 - kk) + 2;
    ctx.beginPath(); ctx.arc(cx, HIT_Y, 60 + 900 * ease(kk), 0, Math.PI * 2); ctx.stroke(); ctx.restore();
  }
  // sparks
  const rnd = (i) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
  for (let i = 0; i < 60; i++) {
    const kk = clamp((k - 0.02) / (0.9 + rnd(i + 7) * 0.9), 0, 1);
    if (kk >= 1) continue;
    const ang = rnd(i) * Math.PI * 2, sp = 500 + rnd(i + 1) * 900;
    const x = cx + Math.cos(ang) * sp * ease(kk), y = HIT_Y + Math.sin(ang) * sp * ease(kk) + 300 * kk * kk;
    ctx.save(); ctx.globalAlpha = 1 - kk; ctx.fillStyle = [COL.ok, COL.text, LANE[0].color, LANE[1].color][i % 4];
    ctx.beginPath(); ctx.arc(x, y, 6 + rnd(i + 3) * 8, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  }
  // text
  const a = window_(k, 0, 0.15, 2.5, 3.2);
  const s = 1 + 0.25 * pulse(t, t1, 0.3);
  ctx.save(); ctx.globalAlpha = a;
  ctx.fillStyle = 'rgba(6,6,11,0.8)'; rrect(ctx, 50, cy - 230, 980, 460, 26); ctx.fill();
  ctx.translate(cx, cy); ctx.scale(s, s);
  ctx.shadowColor = COL.ok; ctx.shadowBlur = 40;
  text(ctx, big ? 'REALIGNED' : 'ALIGNED AGAIN', 0, -120, `800 ${big ? 108 : 96}px ${FONT_D}`, COL.ok);
  ctx.shadowBlur = 0;
  if (big) {
    text(ctx, '4 LAPS = 180 STEPS = 45 PULSES', 0, -20, `700 40px ${FONT_D}`, COL.text);
    text(ctx, '45 = 11 × 4 + 1, so four laps slip exactly one pulse', 0, 50, fitFont(ctx, '45 = 11 × 4 + 1, so four laps slip exactly one pulse', `500 34px ${FONT}`, 900), COL.muted);
    text(ctx, 'DRUM UNLOCKED', 0, 140, `800 48px ${FONT_D}`, LANE[1].color);
  } else {
    text(ctx, 'FULL COMBO', 0, -20, `800 64px ${FONT_D}`, COL.text);
    text(ctx, `${hitsBefore(stepTime(END_STEP))} HITS · 8 LAPS · 90 PULSES`, 0, 50, `700 34px ${MONO}`, COL.muted);
    text(ctx, 'Every 4 laps they meet. 7 times in the intro.', 0, 140, `500 34px ${FONT}`, COL.text);
  }
  ctx.restore();
}

function drawEndFade(ctx, t) {
  const tEnd = stepTime(END_STEP) + END_HOLD;
  const a = clamp((t - (tEnd - 0.6)) / 0.6, 0, 1);
  if (a > 0) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, W, H); ctx.restore(); }
}

function renderFrame(t) {
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  drawBackground(ctx, t);
  // soft screen shake on drum hits after the realignment
  const lastDrum = lastHitTime(t, 1);
  const shake = (musicTime(t) >= DRUM_IN * STEP && lastDrum != null) ? 6 * pulse(t, lastDrum, 0.12) : 0;
  ctx.save();
  ctx.beginPath(); ctx.rect(0, PLAY_TOP, W, PLAY_BOT - PLAY_TOP); ctx.clip();
  if (shake > 0) ctx.translate(0, shake * Math.sin(t * 200));
  drawCombo(ctx, t);
  drawGroups(ctx, t);
  drawLapLines(ctx, t);
  drawNotes(ctx, t);
  drawHitLine(ctx, t);
  drawJudgement(ctx, t);
  ctx.restore();
  drawHUD(ctx, t);
  drawCaptions(ctx, t);
  drawCelebration(ctx, t, DRUM_IN, true);
  drawCelebration(ctx, t, END_STEP, false);
  drawHook(ctx, t);
  drawEndFade(ctx, t);
}

window.renderFrame = renderFrame;
window.FILM = { T0, STEP, CYCLE, LAPS, END_STEP, DRUM_IN, END_HOLD,
  duration: T0 + END_STEP * STEP + END_HOLD };
