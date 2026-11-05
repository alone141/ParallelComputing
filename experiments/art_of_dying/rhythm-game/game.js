// 45 — a rhythm game that plays the intro of Gojira's "The Art of Dying" (2008), round 3.
// Picture only, on the record's own clock (the owner muxes the recording). Every frame is a pure
// function of film time t, and every timing is a function of the sixteenth counter k:
// t(k) = TIMELINE.offset + k * TIMELINE.step (BPM and FIRST_CLICK_OFFSET_S live in rhythm.py).
'use strict';

const T = window.TIMELINE;
const STEP = T.step, OFFSET = T.offset, CYCLE = T.cycle, END = T.end, LAP_S = CYCLE * STEP;
const RESTS = new Set(T.rests), SNARES = new Set(T.snares), KICK_RESTS = new Set(T.kickRests);
const GROUPS = T.groups, GROUP_START = T.groupStarts;
const PULSE_IN = T.pulseIn, REALIGN_1 = T.realign1, CUT = T.cut, TURN = T.turn, TURN_PATTERN = T.turnPattern;
const DURATION = T.total;
const N_LAP_LINES = TURN / CYCLE;                 // 16 laps, then the 6/4

// ---------- layout (1080 x 1920) ----------
const W = 1080, H = 1920;
const HIT_Y = 1340, PLAY_TOP = 390, PLAY_BOT = 1470;
const APPROACH = 1.6;
const SPEED = (HIT_Y - 420) / APPROACH;
const LANE_X0 = 100, LW = 130;
const COL = {
  bg: '#06060b', lap: '#FFFFFF', drift: '#FF4D6D', ok: '#5CFF8A', text: '#ECECF4', muted: '#8A8AA3',
  five: '#FFB13B', three: '#FF7A59', six: '#C8F56A', snare: '#FFFFFF', kick: '#FF6B4A',
};
const LANE_COLOUR = { rim: '#FFB13B', pulse: '#4FC3FF', ks: '#FF6B4A', ride: '#C8F56A', bass: '#FF5CC8', gtr: '#9D8CFF' };
const LANES = T.layers.map((l, i) => ({ ...l, i, x0: LANE_X0 + i * LW, x1: LANE_X0 + (i + 1) * LW, color: LANE_COLOUR[l.key] }));
const BY_KEY = Object.fromEntries(LANES.map((l) => [l.key, l]));
const LANE_X1 = LANES[LANES.length - 1].x1;
const FONT_D = '"Inter Display", "Inter", sans-serif';
const FONT = '"Inter", sans-serif';
const MONO = '"JetBrains Mono", monospace';

// ---------- helpers ----------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ease = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
const smooth = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const pulse = (t, t0, d) => (t < t0 ? 0 : Math.max(0, 1 - (t - t0) / d));
function window_(t, a, b, c, d) {
  if (t < a || t > d) return 0;
  if (t < b) return smooth((t - a) / (b - a));
  if (t < c) return 1;
  return 1 - smooth((t - c) / (d - c));
}
function rrect(ctx, x, y, w, h, r) {
  r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r); ctx.closePath();
}
function fitFont(ctx, s, font, maxW) {
  const m = font.match(/^(.*?)(\d+)px(.*)$/);
  let size = Number(m[2]);
  for (;;) {
    const f = `${m[1]}${size}px${m[3]}`;
    ctx.font = f;
    if (ctx.measureText(s).width <= maxW || size <= 18) return f;
    size -= 1;
  }
}
function text(ctx, s, x, y, font, color, align = 'center', alpha = 1) {
  ctx.save(); ctx.globalAlpha *= alpha; ctx.font = font; ctx.fillStyle = color;
  ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore();
}
const stepTime = (k) => OFFSET + k * STEP;
const stepAt = (t) => Math.floor((t - OFFSET) / STEP + 1e-6);
const noteY = (k, t) => HIT_Y - (stepTime(k) - t) * SPEED;

// ---------- the music model (mirrors rhythm.py) ----------
function hitsAt(key, k) {
  if (k >= TURN) {
    const p = (k - TURN) % 6;
    if (key === 'ks') return { K: ['kick'], S: ['snare'], '.': [] }[TURN_PATTERN[p]];
    if (key === 'bass') return [p < 3 ? 'note_c1' : 'note_c2'];
    if (key === 'gtr') return [p < 3 ? 'chug_c2' : 'chug_eb2'];
    if (key === 'ride') return k % 4 === 0 ? ['bell'] : k % 4 === 2 ? ['ride'] : [];
    return [];
  }
  const c = k % CYCLE;
  if (key === 'rim') return RESTS.has(c) ? [] : ['click'];
  if (key === 'bass') return RESTS.has(c) ? [] : ['note'];
  if (key === 'pulse') return k % 4 === 0 ? ['pulse'] : [];
  if (key === 'ks') return SNARES.has(c) ? ['snare'] : KICK_RESTS.has(c) ? [] : ['kick'];
  if (key === 'ride') return k === CUT ? ['crash'] : k % 4 === 0 ? ['bell'] : k % 4 === 2 ? ['ride'] : [];
  if (key === 'gtr') return ['chug'];
  return [];
}
const present = (l, k) => k >= l.entry && k < l.exit;
function presence(l, t) {
  // 0..1 for the picture: in at its entry, out over 0.45 s after its exit
  if (t < stepTime(l.entry)) return 0;
  return 1 - smooth((t - stepTime(l.exit)) / 0.45);
}
const beatLane = (k) => (k < CUT ? BY_KEY.pulse : BY_KEY.ride);
const patternLane = (k) => (k < CUT ? BY_KEY.rim : BY_KEY.bass);   // who carries the 5s and 3s

// notes
const NOTES = [];
{
  let foot = 0;
  for (let k = 0; k < END; k++) {
    for (const l of LANES) {
      const on = present(l, k);
      if (l.key === 'pulse' && k < CUT && k % 4 === 0) { NOTES.push({ k, lane: l, kind: on ? 'pulse' : 'ghost' }); continue; }
      if (!on) continue;
      for (const kind of hitsAt(l.key, k)) {
        const note = { k, lane: l, kind };
        if (kind === 'kick') { note.foot = foot; foot ^= 1; }
        if (kind === 'note' && RESTS.has((k + 1) % CYCLE)) note.hold = 1;
        NOTES.push(note);
      }
    }
  }
}
const CUM = [0];
for (let k = 0; k < END; k++) {
  let c = 0;
  for (const l of LANES) if (present(l, k)) c += hitsAt(l.key, k).length;
  CUM.push(CUM[k] + c);
}
function combo(t) {
  const k = stepAt(t);
  if (k < 0) return 0;
  return CUM[Math.min(k + 1, END)];
}
function lastHit(l, t) {
  const k = stepAt(t);
  for (let i = Math.min(k, END - 1); i >= 0 && i > k - 8; i--) {
    if (l.key === 'pulse' && i < CUT ? i % 4 === 0 : present(l, i) && hitsAt(l.key, i).length) return stepTime(i);
  }
  return null;
}

// ---------- captions (film seconds at 140 bpm; scaled with the step so they follow a re-timed film) ----------
const S = STEP / (15 / 140);
const CAPTIONS = [
  { a: 2.6,  b: 4.7,  l1: 'Rim clicks in 5s and 3s.', l2: 'Every group skips its last step.' },
  { a: 5.4,  b: 9.4,  l1: 'A lap is 45 steps. The beat is every 4.', l2: '45 = 11 × 4 + 1: the line lands one step later each lap.' },
  { a: 10.3, b: 14.0, l1: 'Two steps late.', l2: 'The beat never moves. The lap drifts.' },
  { a: 15.0, b: 18.8, l1: 'Three steps late.', l2: 'One more lap and the slip is a whole beat.' },
  { a: 21.6, b: 25.2, l1: 'The tribal drum enters on the beat,', l2: 'exactly where the lap comes home. Then it slips too.' },
  { a: 29.6, b: 33.6, l1: 'Fans call this 21/16 or 45/16.', l2: 'The drummer calls it 4/4, with one spare beat.' },
  { a: 41.2, b: 44.8, l1: 'Every 4 laps they meet.', l2: 'Watch it happen again.' },
  { a: 49.0, b: 53.0, l1: 'Four more laps.', l2: 'The band is waiting for the next PERFECT.' },
  { a: 60.6, b: 64.4, l1: 'Kick kick kick snare. Kick kick snare.', l2: 'The kit spells the same 5s and 3s. Feet alternate.' },
  { a: 65.2, b: 69.0, l1: 'Bass plays the clicks on C1 and holds the last note.', l2: 'Two guitars chug every sixteenth on C2.' },
  { a: 70.0, b: 74.5, l1: 'Clicks and tribal drum are gone.', l2: 'The ride bell keeps the beat. Still slipping.' },
  { a: 77.4, b: 79.2, l1: 'Four laps, then one bar of 6/4.', l2: 'The riff turns around.' },
].map((c) => ({ ...c, a: OFFSET + c.a * S, b: OFFSET + c.b * S }));

// ---------- drawing ----------
function drawLock(ctx, cx, cy, alpha) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = COL.muted; ctx.fillStyle = COL.muted; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.arc(cx, cy - 8, 16, Math.PI, 0); ctx.stroke();
  rrect(ctx, cx - 24, cy - 8, 48, 36, 6); ctx.fill();
  ctx.restore();
}
function drawBackground(ctx, t) {
  ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, W, H);
  for (const l of LANES) {
    const op = presence(l, t);
    const cx = l.x0 + LW / 2;
    ctx.save();
    const g = ctx.createLinearGradient(0, PLAY_TOP, 0, PLAY_BOT);
    g.addColorStop(0, 'rgba(255,255,255,0)');
    g.addColorStop(0.25, `rgba(255,255,255,${0.02 + 0.03 * op})`);
    g.addColorStop(1, `rgba(255,255,255,${0.04 + 0.05 * op})`);
    ctx.fillStyle = g; ctx.fillRect(l.x0, PLAY_TOP, LW, PLAY_BOT - PLAY_TOP);
    ctx.restore();
    ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(l.x0, PLAY_TOP + 60); ctx.lineTo(l.x0, PLAY_BOT); ctx.stroke();
    const f = fitFont(ctx, l.name, `700 24px ${FONT_D}`, LW - 12);
    if (op > 0.01) text(ctx, l.name, cx, PLAY_TOP + 2, f, l.color, 'center', 0.35 + 0.65 * op);
    else {
      text(ctx, l.name, cx, PLAY_TOP + 2, f, COL.muted, 'center', 0.35);
      if (t < stepTime(l.entry)) drawLock(ctx, cx, HIT_Y - 120, 0.35);
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.beginPath(); ctx.moveTo(LANE_X1, PLAY_TOP + 60); ctx.lineTo(LANE_X1, PLAY_BOT); ctx.stroke();
}

function drawHitLine(ctx, t) {
  for (const l of LANES) {
    const last = lastHit(l, t);
    const on = presence(l, t) > 0.01 || (l.key === 'pulse' && t < stepTime(CUT));
    const glow = last == null || !on ? 0 : pulse(t, last, 0.16);
    const op = on ? 0.45 + 0.55 * presence(l, t) : 0.25;
    ctx.save();
    ctx.shadowColor = l.color; ctx.shadowBlur = 16 + 36 * glow;
    ctx.strokeStyle = on ? l.color : COL.muted; ctx.lineWidth = 3 + 3 * glow; ctx.globalAlpha = op * (0.6 + 0.4 * glow);
    rrect(ctx, l.x0 + 14, HIT_Y - 20, LW - 28, 40, 12); ctx.stroke();
    ctx.restore();
  }
  ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(LANE_X0 - 70, HIT_Y); ctx.lineTo(LANE_X1 + 20, HIT_Y); ctx.stroke(); ctx.restore();
}

function drawGroups(ctx, t) {
  const kMin = stepAt(t - 0.5), kMax = Math.ceil((t + APPROACH + 0.3 - OFFSET) / STEP);
  const bracket = (s, e, label, color) => {
    if (e < kMin || s > kMax || s >= END) return;
    const yTop = noteY(s, t), yBot = noteY(e, t);
    const past = t > stepTime(e);
    const alpha = (past ? 0.25 : 1) * clamp((yTop - PLAY_TOP) / 110 + 1, 0, 1);
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.lineCap = 'round';
    const bx = LANE_X0 - 22;
    ctx.beginPath();
    ctx.moveTo(bx + 10, yTop - 16); ctx.lineTo(bx, yTop - 16); ctx.lineTo(bx, yBot + 16); ctx.lineTo(bx + 10, yBot + 16);
    ctx.stroke();
    text(ctx, label, bx - 32, (yTop + yBot) / 2, `700 44px ${FONT_D}`, color);
    ctx.restore();
    return { yBot, alpha, color };
  };
  for (let lap = Math.floor(Math.max(0, kMin) / CYCLE); lap * CYCLE <= Math.min(kMax, TURN - 1); lap++) {
    GROUPS.forEach((g, gi) => {
      const s = lap * CYCLE + GROUP_START[gi], e = s + g - 1;
      const r = bracket(s, e, String(g), g === 5 ? COL.five : COL.three);
      if (!r) return;
      // the skipped step, as a dashed hole in the lane that carries the pattern
      const L = patternLane(s);
      ctx.save(); ctx.globalAlpha = r.alpha * 0.45 * Math.max(0.3, presence(L, t));
      ctx.setLineDash([8, 8]); ctx.lineWidth = 2.5; ctx.strokeStyle = r.color;
      rrect(ctx, L.x0 + 18, r.yBot - 14, LW - 36, 28, 8); ctx.stroke();
      ctx.restore();
    });
  }
  for (let g = 0; g < 4; g++) bracket(TURN + g * 6, TURN + g * 6 + 5, '6', COL.six);
}

function drawNotes(ctx, t) {
  for (const note of NOTES) {
    const y = noteY(note.k, t);
    if (y < PLAY_TOP - 40 || y > PLAY_BOT + 40) continue;
    const l = note.lane;
    const dt = t - stepTime(note.k);
    const cx = l.x0 + LW / 2;
    const op = note.kind === 'ghost' ? 0.8 : Math.max(presence(l, t), dt > 0 ? 1 : 0);
    ctx.save();
    if (dt > 0) {
      if (note.kind === 'ghost' || note.kind === 'ride') { ctx.restore(); continue; }
      const big = note.kind === 'crash';
      const k = clamp(dt / (big ? 0.5 : 0.2), 0, 1);
      ctx.globalAlpha = (1 - k) * op;
      const grow = 1 + (big ? 3 : 1.1) * k;
      const color = note.kind === 'snare' ? COL.snare : note.kind === 'kick' ? COL.kick : l.color;
      ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.shadowColor = color; ctx.shadowBlur = 26;
      const w = (LW - 36) * grow, h = 28 * grow;
      rrect(ctx, cx - w / 2, HIT_Y - h / 2, w, h, 10 * grow); ctx.stroke();
      ctx.restore(); continue;
    }
    const fadeIn = clamp((y - PLAY_TOP) / 110, 0, 1);
    ctx.globalAlpha = fadeIn * op;
    const nw = LW - 36;
    switch (note.kind) {
      case 'click':
        ctx.fillStyle = l.color; ctx.shadowColor = l.color; ctx.shadowBlur = 16;
        rrect(ctx, cx - nw / 2, y - 14, nw, 28, 8); ctx.fill(); break;
      case 'ghost':
        ctx.strokeStyle = l.color; ctx.lineWidth = 2.5; ctx.globalAlpha = fadeIn * 0.75;
        rrect(ctx, cx - nw / 2, y - 14, nw, 28, 8); ctx.stroke();
        ctx.fillStyle = l.color; ctx.globalAlpha = fadeIn * 0.15; rrect(ctx, cx - nw / 2, y - 14, nw, 28, 8); ctx.fill(); break;
      case 'pulse':
        ctx.fillStyle = l.color; ctx.shadowColor = l.color; ctx.shadowBlur = 26;
        rrect(ctx, cx - nw / 2 - 6, y - 20, nw + 12, 40, 12); ctx.fill();
        ctx.shadowBlur = 0; ctx.fillStyle = COL.bg; rrect(ctx, cx - 16, y - 4, 32, 8, 4); ctx.fill(); break;
      case 'kick': {
        const kx = cx + (note.foot ? 22 : -22);
        ctx.fillStyle = COL.kick; ctx.shadowColor = COL.kick; ctx.shadowBlur = 14;
        rrect(ctx, kx - 26, y - 13, 52, 26, 8); ctx.fill();
        ctx.shadowBlur = 0; text(ctx, note.foot ? 'R' : 'L', kx, y + 1, `800 18px ${FONT_D}`, COL.bg); break; }
      case 'snare':
        ctx.fillStyle = COL.snare; ctx.shadowColor = COL.snare; ctx.shadowBlur = 22;
        rrect(ctx, cx - nw / 2, y - 16, nw, 32, 9); ctx.fill(); break;
      case 'bell':
        ctx.fillStyle = l.color; ctx.shadowColor = l.color; ctx.shadowBlur = 22;
        ctx.beginPath(); ctx.arc(cx, y, 20, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0; ctx.fillStyle = COL.bg; ctx.beginPath(); ctx.arc(cx, y, 6, 0, Math.PI * 2); ctx.fill(); break;
      case 'crash':
        ctx.fillStyle = l.color; ctx.shadowColor = l.color; ctx.shadowBlur = 30;
        ctx.beginPath(); ctx.arc(cx, y, 30, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0; text(ctx, 'CRASH', cx, y + 1, `800 14px ${FONT_D}`, COL.bg); break;
      case 'ride':
        ctx.strokeStyle = l.color; ctx.lineWidth = 2.5; ctx.globalAlpha = fadeIn * op * 0.55;
        ctx.beginPath(); ctx.arc(cx, y, 12, 0, Math.PI * 2); ctx.stroke(); break;
      case 'note': case 'note_c1': case 'note_c2': {
        if (note.hold) {
          ctx.fillStyle = l.color; ctx.globalAlpha = fadeIn * op * 0.45;
          rrect(ctx, cx - 18, y - 14 - SPEED * STEP, 36, SPEED * STEP + 14, 10); ctx.fill();
          ctx.globalAlpha = fadeIn * op;
        }
        ctx.fillStyle = l.color; ctx.shadowColor = l.color; ctx.shadowBlur = 16;
        if (note.kind === 'note_c2') ctx.globalAlpha = fadeIn * op * 0.7;
        rrect(ctx, cx - nw / 2, y - 14, nw, 28, 8); ctx.fill();
        if (note.kind !== 'note') { ctx.shadowBlur = 0; text(ctx, note.kind === 'note_c1' ? 'C1' : 'C2', cx, y + 1, `800 18px ${FONT_D}`, COL.bg); }
        break; }
      case 'chug': case 'chug_c2': case 'chug_eb2':
        ctx.fillStyle = l.color; ctx.shadowColor = l.color; ctx.shadowBlur = 10;
        if (note.kind === 'chug_eb2') ctx.globalAlpha = fadeIn * op * 0.7;
        rrect(ctx, cx - 36, y - 9, 30, 18, 5); ctx.fill();
        rrect(ctx, cx + 6, y - 9, 30, 18, 5); ctx.fill();
        if (note.kind === 'chug_eb2') { ctx.shadowBlur = 0; text(ctx, 'E♭', cx, y + 1, `800 16px ${FONT_D}`, COL.text); }
        break;
    }
    ctx.restore();
  }
}

function drawLapLines(ctx, t) {
  for (let lap = 0; lap <= N_LAP_LINES; lap++) {
    const k = lap * CYCLE;
    const y = noteY(k, t);
    if (y < PLAY_TOP - 40 || y > PLAY_BOT + 40) continue;
    const dt = t - stepTime(k);
    const alpha = dt > 0 ? clamp(1 - dt / 0.5, 0, 1) * 0.5 : clamp((y - PLAY_TOP) / 110, 0, 1);
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.strokeStyle = COL.lap; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(LANE_X0 - 70, y); ctx.lineTo(LANE_X1, y); ctx.stroke();
    const label = lap === 0 ? 'START' : lap === N_LAP_LINES ? '6/4' : `LAP ${lap + 1}`;
    ctx.fillStyle = COL.lap; rrect(ctx, LANE_X1 - 104, y - 34, 104, 30, 7); ctx.fill();
    text(ctx, label, LANE_X1 - 52, y - 18, `700 22px ${FONT_D}`, COL.bg);
    const off = k % 4;
    if (off > 0) {
      const B = beatLane(k);
      const yBeat = noteY(k - off, t);
      const x = B.x0 + 8;
      ctx.strokeStyle = COL.drift; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x + 10, yBeat); ctx.lineTo(x, yBeat); ctx.lineTo(x, y); ctx.lineTo(x + 10, y); ctx.stroke();
      ctx.fillStyle = COL.drift; ctx.globalAlpha = alpha * 0.2; ctx.fillRect(B.x0 + 14, y, LW - 28, yBeat - y);
      ctx.globalAlpha = alpha;
      text(ctx, `+${off}`, B.x0 + LW / 2 + 8, (y + yBeat) / 2, `800 34px ${FONT_D}`, COL.drift);
    }
    ctx.restore();
  }
}

function drawLapJudgement(ctx, t) {
  const lap = Math.floor((t - OFFSET) / LAP_S + 1e-9);
  if (lap < 1 || lap > N_LAP_LINES) return;
  const k = lap * CYCLE;
  const dk = t - stepTime(k);
  if (dk > 1.1) return;
  const B = beatLane(k);
  const off = k % 4;
  const ok = off === 0;
  const a = window_(dk, 0, 0.08, 0.8, 1.1);
  const s = 1 + 0.35 * pulse(t, stepTime(k), 0.25);
  ctx.save(); ctx.globalAlpha = a;
  ctx.translate(B.x0 + LW / 2, HIT_Y - 90); ctx.scale(s, s);
  ctx.shadowColor = ok ? COL.ok : COL.drift; ctx.shadowBlur = 20;
  if (ok) text(ctx, 'PERFECT', 0, 0, `800 28px ${FONT_D}`, COL.ok);
  else { text(ctx, `+${off}`, 0, -18, `800 40px ${FONT_D}`, COL.drift); text(ctx, 'LATE', 0, 20, `800 26px ${FONT_D}`, COL.drift); }
  ctx.restore();
}

const RESULTS_AT = stepTime(TURN + 12);          // the results card arrives halfway through the 6/4
function drawCombo(ctx, t) {
  const c = combo(t);
  if (c === 0) return;
  const hide = window_(t, RESULTS_AT, RESULTS_AT + 0.4, DURATION + 1, DURATION + 2);
  if (hide >= 1) return;
  let last = -1;
  for (const l of LANES) { const h = lastHit(l, t); if (h != null && presence(l, t) > 0.01) last = Math.max(last, h); }
  const pop = pulse(t, last, 0.12);
  const cx = (LANE_X0 + LANE_X1) / 2, cy = 1000;
  ctx.save();
  ctx.globalAlpha = 0.55 * (1 - hide);
  ctx.translate(cx, cy); ctx.scale(1 + 0.15 * pop, 1 + 0.15 * pop);
  ctx.shadowColor = 'rgba(255,255,255,0.6)'; ctx.shadowBlur = 8 + 24 * pop;
  text(ctx, String(c), 0, 0, `800 130px ${FONT_D}`, COL.text);
  ctx.restore();
  text(ctx, 'COMBO', cx, cy + 86, `700 28px ${FONT_D}`, COL.muted, 'center', 0.6 * (1 - hide));
}

function drawHUD(ctx, t) {
  const k = clamp(stepAt(t), 0, END - 1);
  const lap = Math.floor(k / CYCLE);
  const inTurn = k >= TURN;
  const lapStart = stepTime(Math.min(lap, N_LAP_LINES) * CYCLE);
  const pop = lap > 0 ? pulse(t, lapStart, 0.35) : 0;
  const off = (Math.min(lap, N_LAP_LINES) * CYCLE) % 4;
  text(ctx, 'GOJIRA · THE ART OF DYING · 140 BPM', 540, 165, `600 28px ${FONT_D}`, COL.muted);
  ctx.save(); ctx.translate(100, 262); ctx.scale(1 + 0.12 * pop, 1 + 0.12 * pop);
  text(ctx, inTurn ? 'METER' : 'LAP', 0, -42, `600 26px ${FONT_D}`, COL.muted, 'left');
  if (inTurn) text(ctx, '6/4', 0, 18, `800 80px ${FONT_D}`, COL.six, 'left');
  else { text(ctx, `${(lap % 4) + 1}`, 0, 18, `800 88px ${FONT_D}`, COL.text, 'left'); text(ctx, '/4', 62, 30, `600 36px ${FONT_D}`, COL.muted, 'left'); }
  ctx.restore();
  const driftCol = off === 0 ? COL.ok : COL.drift;
  text(ctx, 'LINE VS BEAT', 490, 220, `600 26px ${FONT_D}`, COL.muted);
  ctx.save(); ctx.translate(490, 280); ctx.scale(1 + 0.15 * pop, 1 + 0.15 * pop);
  text(ctx, off === 0 ? 'ALIGNED' : `+${off} STEP${off > 1 ? 'S' : ''}`, 0, 0, `800 50px ${FONT_D}`, driftCol);
  ctx.restore();
  for (let i = 0; i < 4; i++) {
    const x = 418 + i * 48;
    ctx.save(); ctx.globalAlpha = i < (inTurn ? 4 : (lap % 4) + 1) ? 1 : 0.25; ctx.fillStyle = i === 0 ? COL.ok : COL.drift;
    rrect(ctx, x - 16, 328, 32, 14, 4); ctx.fill(); ctx.restore();
  }
  // bar counter (the tab's bar numbers; the cut jumps from 33 to 42)
  const bar = T.bars[k];
  text(ctx, 'BAR', 880, 220, `600 26px ${FONT_D}`, COL.muted, 'right');
  text(ctx, String(bar), 880, 280, `800 60px ${FONT_D}`, COL.text, 'right');
  LANES.forEach((l, i) => {
    const x = 880 - (5 - i) * 30 - 8;
    ctx.save(); ctx.globalAlpha = presence(l, t) > 0.5 ? 1 : 0.22; ctx.fillStyle = l.color;
    ctx.beginPath(); ctx.arc(x, 336, 9, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  });
}

function drawCaptions(ctx, t) {
  for (const c of CAPTIONS) {
    const a = window_(t, c.a, c.a + 0.3, c.b - 0.3, c.b);
    if (a <= 0) continue;
    ctx.save(); ctx.globalAlpha = a;
    ctx.fillStyle = 'rgba(6,6,11,0.86)'; rrect(ctx, 110, 418, 940, 150, 20); ctx.fill();
    text(ctx, c.l1, 580, 468, fitFont(ctx, c.l1, `600 40px ${FONT}`, 890), COL.text);
    text(ctx, c.l2, 580, 522, fitFont(ctx, c.l2, `500 34px ${FONT}`, 890), COL.muted);
    ctx.restore();
  }
}

function banner(ctx, t, t0, big, small, color, len = 1.6) {
  const k = t - t0;
  if (k < 0 || k > len) return;
  const a = window_(k, 0, 0.1, len - 0.4, len), s = 1 + 0.2 * pulse(t, t0, 0.3);
  ctx.save(); ctx.globalAlpha = a; ctx.translate(490, 760); ctx.scale(s, s);
  ctx.fillStyle = 'rgba(6,6,11,0.85)'; rrect(ctx, -330, -80, 660, 160, 20); ctx.fill();
  ctx.shadowColor = color; ctx.shadowBlur = 30;
  text(ctx, big, 0, small ? -22 : 0, fitFont(ctx, big, `800 72px ${FONT_D}`, 620), color);
  ctx.shadowBlur = 0;
  if (small) text(ctx, small, 0, 40, fitFont(ctx, small, `700 28px ${MONO}`, 620), COL.text);
  ctx.restore();
}
function laneFlash(ctx, t, l, t0) {
  const f = pulse(t, t0, 0.4);
  if (f <= 0) return;
  ctx.save(); ctx.globalAlpha = 0.5 * f; ctx.fillStyle = l.color; ctx.fillRect(l.x0, PLAY_TOP, LW, PLAY_BOT - PLAY_TOP); ctx.restore();
}
function drawMoments(ctx, t) {
  // bar 12: the pulse joins, on the beat, exactly where the lap comes home
  laneFlash(ctx, t, BY_KEY.pulse, stepTime(PULSE_IN));
  banner(ctx, t, stepTime(PULSE_IN), 'PULSE IN', 'ON THE BEAT · LAP 5 · BAR 12', LANE_COLOUR.pulse, 1.8);
  // bar 42: the band drops in together; clicks and pulse go dark
  for (const key of ['ks', 'ride', 'bass', 'gtr']) laneFlash(ctx, t, BY_KEY[key], stepTime(CUT));
  // bar 53: the turnaround
  banner(ctx, t, stepTime(TURN), '6/4 TURNAROUND', 'KICK KICK KICK SNARE · SNARE, ×4', COL.six, 1.6);
}

function drawCelebration(ctx, t, k0, bandDrop) {
  const t1 = stepTime(k0), k = t - t1;
  if (k < -0.001 || k > 2.6) return;
  const cx = 540, cy = 860;
  const flash = pulse(t, t1, 0.2);
  ctx.save(); ctx.globalAlpha = 0.5 * flash; ctx.fillStyle = COL.ok; ctx.fillRect(0, 0, W, H); ctx.restore();
  for (let i = 0; i < 3; i++) {
    const kk = clamp((k - i * 0.12) / 1.1, 0, 1);
    if (kk <= 0 || kk >= 1) continue;
    ctx.save(); ctx.globalAlpha = (1 - kk) * 0.8; ctx.strokeStyle = i === 1 ? COL.text : COL.ok; ctx.lineWidth = 10 * (1 - kk) + 2;
    ctx.beginPath(); ctx.arc(cx, HIT_Y, 60 + 900 * ease(kk), 0, Math.PI * 2); ctx.stroke(); ctx.restore();
  }
  const rnd = (i) => { const x = Math.sin(i * 127.1 + 311.7 + k0) * 43758.5453; return x - Math.floor(x); };
  const sparkCols = bandDrop ? [COL.ok, COL.text, LANE_COLOUR.ks, LANE_COLOUR.ride, LANE_COLOUR.bass, LANE_COLOUR.gtr] : [COL.ok, COL.text, LANE_COLOUR.rim, LANE_COLOUR.pulse];
  for (let i = 0; i < (bandDrop ? 110 : 70); i++) {
    const kk = clamp((k - 0.02) / (0.9 + rnd(i + 7) * 0.9), 0, 1);
    if (kk >= 1) continue;
    const ang = rnd(i) * Math.PI * 2, sp = 500 + rnd(i + 1) * 900;
    const x = cx + Math.cos(ang) * sp * ease(kk), y = HIT_Y + Math.sin(ang) * sp * ease(kk) + 300 * kk * kk;
    ctx.save(); ctx.globalAlpha = 1 - kk; ctx.fillStyle = sparkCols[i % sparkCols.length];
    ctx.beginPath(); ctx.arc(x, y, 5 + rnd(i + 3) * 8, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  }
  const a = window_(k, 0, 0.15, 2.1, 2.6);
  const s = 1 + 0.25 * pulse(t, t1, 0.3);
  ctx.save(); ctx.globalAlpha = a;
  ctx.fillStyle = 'rgba(6,6,11,0.84)'; rrect(ctx, 50, cy - 230, 980, 460, 26); ctx.fill();
  ctx.translate(cx, cy); ctx.scale(s, s);
  ctx.shadowColor = COL.ok; ctx.shadowBlur = 40;
  text(ctx, 'REALIGNED', 0, -120, `800 108px ${FONT_D}`, COL.ok);
  ctx.shadowBlur = 0;
  text(ctx, '4 LAPS = 180 STEPS = 45 BEATS', 0, -20, `700 40px ${FONT_D}`, COL.text);
  const line = bandDrop ? 'Eight laps on the record, two homecomings' : '45 = 11 × 4 + 1: four laps slip exactly one beat';
  text(ctx, line, 0, 50, fitFont(ctx, line, `500 34px ${FONT}`, 900), COL.muted);
  if (bandDrop) {
    ctx.shadowColor = LANE_COLOUR.ks; ctx.shadowBlur = 24;
    text(ctx, 'BAND IN · BAR 42', 0, 140, `800 56px ${FONT_D}`, LANE_COLOUR.ks);
  } else {
    text(ctx, 'LAP 9 · BAR 23', 0, 140, `700 36px ${FONT_D}`, COL.muted);
  }
  ctx.restore();
}

function drawHook(ctx, t) {
  const a = window_(t, OFFSET, OFFSET + 0.01, OFFSET + 2.0, OFFSET + 2.5);
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a;
  ctx.fillStyle = 'rgba(6,6,11,0.78)'; ctx.fillRect(0, 580, W, 520);
  text(ctx, 'NOW PLAYING · THE ART OF DYING', 540, 650, `600 30px ${FONT_D}`, COL.muted);
  text(ctx, 'THE BEAT', 540, 770, `800 124px ${FONT_D}`, COL.text);
  text(ctx, 'SLIPS.', 540, 890, `800 124px ${FONT_D}`, COL.drift);
  text(ctx, 'The drums play 4. The riff plays 45. Watch them meet.', 540, 1000, fitFont(ctx, 'The drums play 4. The riff plays 45. Watch them meet.', `500 40px ${FONT}`, 960), COL.muted);
  ctx.restore();
}

function drawResults(ctx, t) {
  const a = window_(t, RESULTS_AT, RESULTS_AT + 0.5, DURATION + 1, DURATION + 2);
  if (a <= 0) return;
  const cx = 540, cy = 860;
  ctx.save(); ctx.globalAlpha = a;
  ctx.fillStyle = 'rgba(6,6,11,0.9)'; rrect(ctx, 50, cy - 250, 980, 500, 26); ctx.fill();
  ctx.shadowColor = COL.ok; ctx.shadowBlur = 40;
  text(ctx, 'FULL COMBO', 0 + cx, cy - 140, `800 96px ${FONT_D}`, COL.ok);
  ctx.shadowBlur = 0;
  text(ctx, `${CUM[END]} HITS · 16 LAPS + 6/4`, cx, cy - 40, `700 34px ${MONO}`, COL.text);
  text(ctx, '4 realignments · 45 = 11 × 4 + 1', cx, cy + 20, `500 32px ${FONT}`, COL.muted);
  text(ctx, 'GOJIRA · THE ART OF DYING', cx, cy + 120, `800 44px ${FONT_D}`, COL.text);
  text(ctx, 'The Way of All Flesh · 2008', cx, cy + 175, `500 30px ${FONT}`, COL.muted);
  ctx.restore();
}

function renderFrame(t) {
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  drawBackground(ctx, t);
  const k = clamp(stepAt(t), 0, END - 1);
  const B = beatLane(k);
  const lastBeat = lastHit(B, t);
  const shake = presence(B, t) > 0.5 && lastBeat != null ? 5 * pulse(t, lastBeat, 0.12) : 0;
  ctx.save();
  ctx.beginPath(); ctx.rect(0, PLAY_TOP, W, PLAY_BOT - PLAY_TOP); ctx.clip();
  if (shake > 0) ctx.translate(0, shake * Math.sin(t * 200));
  drawCombo(ctx, t);
  drawGroups(ctx, t);
  drawLapLines(ctx, t);
  drawNotes(ctx, t);
  drawHitLine(ctx, t);
  drawLapJudgement(ctx, t);
  ctx.restore();
  drawHUD(ctx, t);
  drawCaptions(ctx, t);
  drawMoments(ctx, t);
  drawCelebration(ctx, t, REALIGN_1, false);
  drawCelebration(ctx, t, CUT, true);
  drawResults(ctx, t);
  drawHook(ctx, t);
}

window.renderFrame = renderFrame;
window.FILM = { duration: DURATION, frames: T.frames, fps: T.fps, hits: CUM[END] };
