#!/usr/bin/env node
/* Three Doors: the Playwright driver.

   node render.mjs --still <frame> [--out path]      one PNG (under a second after warm-up)
   node render.mjs --frames 237,1100 [--out dir]      several PNGs for inspection (out/frames/)
   node render.mjs --slice i/4                        frames [525 i, 525 (i + 1)) piped as PNG into a lossless x264 slice
   node render.mjs --preview [i/4]                    every second frame at 540 x 960, 30 fps (one file, or one of 4 slices)
   node render.mjs --check                            the layout checks over every event frame (+-1) and every sixth frame
   node render.mjs --verify                           slice-boundary frames rendered in two fresh browsers, PNG bytes compared
   node render.mjs --stills                           the named moments, their 360 x 640 downscales and a contact sheet
   node render.mjs --time                             ms per frame (draw and PNG encode) on the heaviest frames

   Frames leave the page as canvas.toDataURL('image/png'); nothing here depends on the clock or on chance.
*/
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, 'out');
const STORY_PATH = path.join(HERE, 'build', 'story.json');
const LAUNCH_ARGS = ['--disable-gpu', '--font-render-hinting=none', '--disable-lcd-text', '--force-color-profile=srgb', '--hide-scrollbars'];

const argv = process.argv.slice(2);
const flag = name => argv.includes(name);
const opt = (name, dflt) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt; };

if (!fs.existsSync(STORY_PATH)) { console.error(`missing ${STORY_PATH}: run story.py first`); process.exit(2); }
const story = JSON.parse(fs.readFileSync(STORY_PATH, 'utf8'));
const FPS = story.meta.fps, FRAMES = story.meta.frames, SLICES = story.meta.slices, SLICE_FRAMES = story.meta.slice_frames;
fs.mkdirSync(OUT, { recursive: true });

const pngOf = url => Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');
const sha = buf => crypto.createHash('sha256').update(buf).digest('hex');
const secs = f => (f / FPS).toFixed(2);

async function openPage({ scale = 1 } = {}) {
  const browser = await chromium.launch({ args: LAUNCH_ARGS });
  const page = await browser.newPage({ viewport: { width: Math.round(1080 * scale), height: Math.round(1920 * scale) }, deviceScaleFactor: 1 });
  page.on('pageerror', e => { console.error('[pageerror]', e.message); });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.error('[page]', m.text()); });
  await page.goto('file://' + path.join(HERE, 'page.html'));
  const chk = await page.evaluate(() => window.ready());
  const fail = [];
  if (Math.abs(chk.mono0_100 - 60) > 0.1) fail.push(`JetBrains Mono '0' at 100 px = ${chk.mono0_100} (want 60.0 +- 0.1)`);
  if (Math.abs(chk.target56 - 401.4) > 1) fail.push(`Inter Display Bold 56 px 'Switch or stay?' = ${chk.target56} (want 401.4 +- 1)`);
  if (fail.length) { await browser.close(); throw new Error('font gate failed: ' + fail.join('; ')); }
  await page.evaluate(([s, sc]) => window.setup(s, sc), [story, scale]);
  return { browser, page, fonts: chk };
}

async function frame(page, f) {
  const r = await page.evaluate(f => window.frameDataURL(f), f);
  return { png: pngOf(r.url), drawMs: r.drawMs, encMs: r.encMs };
}

function ffmpeg(args) {
  const p = spawn('ffmpeg', ['-y', '-loglevel', 'error', ...args], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => { p.on('close', code => (code === 0 ? res() : rej(new Error('ffmpeg exited ' + code)))); p.on('error', rej); });
  return { p, done };
}
async function pipe(stream, buf) { if (!stream.write(buf)) await new Promise(r => stream.once('drain', r)); }

async function encodeFrames({ page, frames, fps, outFile, label }) {
  const { p, done } = ffmpeg(['-f', 'image2pipe', '-framerate', String(fps), '-i', '-', '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', '-threads', '1', outFile]);
  const t0 = Date.now();
  let draw = 0, enc = 0, n = 0;
  for (const f of frames) {
    const r = await frame(page, f);
    draw += r.drawMs; enc += r.encMs; n++;
    await pipe(p.stdin, r.png);
    if (n % 100 === 0) console.log(`${label}: ${n}/${frames.length} frames, ${((Date.now() - t0) / n).toFixed(0)} ms/frame`);
  }
  p.stdin.end();
  await done;
  const total = (Date.now() - t0) / 1000;
  console.log(`${label}: ${n} frames -> ${outFile} in ${total.toFixed(1)} s (${(1000 * total / n).toFixed(0)} ms/frame wall; draw ${(draw / n).toFixed(1)} ms, png ${(enc / n).toFixed(1)} ms)`);
}

function parseSlice(spec) {
  const m = /^(\d+)\/(\d+)$/.exec(spec || '');
  if (!m || +m[2] !== SLICES || +m[1] < 0 || +m[1] >= SLICES) throw new Error(`slice must be i/${SLICES} with 0 <= i < ${SLICES}`);
  return +m[1];
}

async function main() {
  if (flag('--still')) {
    const f = parseInt(opt('--still'), 10);
    if (!(f >= 0 && f < FRAMES)) throw new Error('frame out of range');
    const outFile = opt('--out', path.join(OUT, `still_${String(f).padStart(4, '0')}.png`));
    const { browser, page } = await openPage();
    await frame(page, f);  // warm-up: bakes the sprites this frame needs
    const t0 = Date.now();
    const r = await frame(page, f);
    const ms = Date.now() - t0;
    fs.mkdirSync(path.dirname(outFile), { recursive: true });
    fs.writeFileSync(outFile, r.png);
    console.log(`frame ${f} (${secs(f)} s) -> ${outFile}: ${ms} ms after warm-up (draw ${r.drawMs.toFixed(1)} ms, png ${r.encMs.toFixed(1)} ms)`);
    await browser.close();
    return;
  }
  if (flag('--frames')) {
    const list = opt('--frames').split(',').map(s => parseInt(s, 10));
    const dir = opt('--out', path.join(OUT, 'frames'));
    fs.mkdirSync(dir, { recursive: true });
    const { browser, page } = await openPage();
    for (const f of list) {
      const r = await frame(page, f);
      const file = path.join(dir, `frame_${String(f).padStart(4, '0')}.png`);
      fs.writeFileSync(file, r.png);
      console.log(`frame ${f} (${secs(f)} s) -> ${file} (draw ${r.drawMs.toFixed(1)} ms, png ${r.encMs.toFixed(1)} ms)`);
    }
    await browser.close();
    return;
  }
  if (flag('--slice')) {
    const i = parseSlice(opt('--slice'));
    const frames = []; for (let f = i * SLICE_FRAMES; f < (i + 1) * SLICE_FRAMES; f++) frames.push(f);
    const { browser, page } = await openPage();
    await encodeFrames({ page, frames, fps: FPS, outFile: path.join(OUT, `slice_${i}.mkv`), label: `slice ${i}` });
    await browser.close();
    return;
  }
  if (flag('--preview')) {
    const spec = opt('--preview', '');
    const all = []; for (let f = 0; f < FRAMES; f += 2) all.push(f);
    let frames = all, outFile = path.join(OUT, 'preview_frames.mkv'), label = 'preview';
    if (/^\d+\/\d+$/.test(spec)) {
      const i = parseSlice(spec);
      const per = all.length / SLICES;
      frames = all.slice(i * per, (i + 1) * per); outFile = path.join(OUT, `preview_slice_${i}.mkv`); label = `preview slice ${i}`;
    }
    const { browser, page } = await openPage({ scale: 0.5 });
    await encodeFrames({ page, frames, fps: FPS / 2, outFile, label });
    await browser.close();
    return;
  }
  if (flag('--check')) {
    const { browser, page, fonts } = await openPage();
    const t0 = Date.now();
    const report = await page.evaluate(() => window.runCheck());
    report.ms = Date.now() - t0;
    report.fonts = fonts;
    report.story = { frames: FRAMES, events: story.events.length, texts: Object.keys(story.texts).length };
    const file = path.join(OUT, 'layout_report.json');
    fs.writeFileSync(file, JSON.stringify(report, null, 1));
    const byCheck = {};
    for (const v of report.violations) byCheck[v.check] = (byCheck[v.check] || 0) + 1;
    console.log(`checked ${report.frames_checked} frames, ${report.items_total} items (${report.text_items_total} text), loop identical: ${report.loop_identical}, fitted: ${JSON.stringify(report.fitted)}, ${report.ms} ms`);
    console.log(`violations: ${report.violations.length} ${JSON.stringify(byCheck)} -> ${file}`);
    for (const v of report.violations.slice(0, 40)) console.log(`  frame ${v.frame} ${v.check} ${v.id}: ${v.detail}`);
    await browser.close();
    process.exitCode = report.violations.length ? 1 : 0;
    return;
  }
  if (flag('--verify')) {
    const frames = new Set([0, 1, story.meta.loop.cover_frame, FRAMES - 1]);
    for (let i = 1; i < SLICES; i++) { frames.add(i * SLICE_FRAMES - 1); frames.add(i * SLICE_FRAMES); }
    const list = [...frames].sort((a, b) => a - b);
    const hashes = [];
    for (let run = 0; run < 2; run++) {
      const { browser, page } = await openPage();
      const h = {};
      for (const f of list) h[f] = sha((await frame(page, f)).png);
      hashes.push(h);
      await browser.close();
    }
    let ok = true;
    for (const f of list) {
      const same = hashes[0][f] === hashes[1][f];
      ok = ok && same;
      console.log(`frame ${String(f).padStart(4)} ${hashes[0][f].slice(0, 16)} ${hashes[1][f].slice(0, 16)} ${same ? 'identical' : 'DIFFERENT'}`);
    }
    console.log(ok ? `verify: ${list.length} frames identical across two browser processes` : 'verify: FAILED');
    process.exitCode = ok ? 0 : 1;
    return;
  }
  if (flag('--stills')) {
    const dir = path.join(OUT, 'stills');
    fs.mkdirSync(dir, { recursive: true });
    const { browser, page } = await openPage();
    const stills = await page.evaluate(() => window.stillList());
    await page.evaluate(n => window.sheetInit(n, 6), stills.length);
    const t0 = Date.now();
    for (let i = 0; i < stills.length; i++) {
      const [name, f] = stills[i];
      const r = await page.evaluate(f => window.captureStill(f), f);
      const base = `${String(i).padStart(2, '0')}_${name}`;
      fs.writeFileSync(path.join(dir, `${base}.png`), pngOf(r.full));
      fs.writeFileSync(path.join(dir, `${base}_360.png`), pngOf(r.small));
      await page.evaluate(([i, cap]) => window.sheetAdd(i, cap), [i, `${String(i).padStart(2, '0')} ${name} · ${secs(f)} s`]);
      console.log(`${base}: frame ${f} (${secs(f)} s)`);
    }
    const sheet = await page.evaluate(() => window.sheetDataURL());
    fs.writeFileSync(path.join(dir, 'contact_sheet.png'), pngOf(sheet));
    console.log(`${stills.length} stills + 360x640 downscales + contact sheet -> ${dir} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    await browser.close();
    return;
  }
  if (flag('--time')) {
    const { browser, page } = await openPage();
    const list = (opt('--time', '') || '120,596,1100,1566,1770').split(',').map(s => parseInt(s, 10)).filter(n => n >= 0);
    for (const f of list) await frame(page, f);   // warm-up: bake every sprite these frames need
    const out = [];
    for (const f of list) {
      const t = await page.evaluate(([f, n]) => window.timeFrame(f, n), [f, 7]);
      out.push(t);
      console.log(`frame ${String(f).padStart(4)} (${secs(f)} s): draw ${t.draw_ms_median} ms (min ${t.draw_ms_min}), png ${t.enc_ms_median} ms (min ${t.enc_ms_min}), ${t.sprites} sprites`);
    }
    fs.writeFileSync(path.join(OUT, 'timing.json'), JSON.stringify(out, null, 1));
    await browser.close();
    return;
  }
  console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('*/')[0].split('\n').slice(1).join('\n'));
}

main().catch(e => { console.error(e.stack || String(e)); process.exit(1); });
