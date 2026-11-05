// Deterministic recorder: opens game.html in headless Chromium, steps the game
// clock one frame at a time, screenshots each frame and pipes the PNGs into
// ffmpeg. Usage:
//   node render.js video out/video_silent.mp4 [fps] [crf]
//   node render.js stills out/stills 1.0 7.3 20.9   (film times in seconds)
'use strict';
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

async function open() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await page.addInitScript(() => { window.__RECORDING__ = true; });
  await page.goto('file://' + path.resolve(__dirname, 'game.html'));
  await page.evaluate(() => document.fonts.ready);
  // make sure the three fonts are actually loaded before the first frame
  await page.evaluate(async () => {
    await Promise.all([
      document.fonts.load('800 100px "Inter Display"'),
      document.fonts.load('500 40px "Inter"'),
      document.fonts.load('700 60px "JetBrains Mono"'),
    ]);
  });
  const film = await page.evaluate(() => window.FILM);
  return { browser, page, film };
}

async function stills(outDir, times) {
  fs.mkdirSync(outDir, { recursive: true });
  const { browser, page } = await open();
  for (const t of times) {
    await page.evaluate((tt) => window.renderFrame(tt), t);
    const f = path.join(outDir, `still_${t.toFixed(2).replace('.', '_')}s.png`);
    await page.screenshot({ path: f, type: 'png' });
    console.log('wrote', f);
  }
  await browser.close();
}

async function video(outFile, fps, crf) {
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  const { browser, page, film } = await open();
  const nFrames = film.frames && fps === film.fps ? film.frames : Math.round(film.duration * fps);
  console.log(`duration ${film.duration.toFixed(3)} s, ${nFrames} frames at ${fps} fps`);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', String(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', outFile],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < nFrames; i++) {
    const t = i / fps;
    await page.evaluate((tt) => window.renderFrame(tt), t);
    const buf = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`frame ${i}/${nFrames}  ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  ff.stdin.end();
  await new Promise((res, rej) => ff.on('close', (c) => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))));
  await browser.close();
  console.log('wrote', outFile, `in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}

(async () => {
  const [mode, out, ...rest] = process.argv.slice(2);
  if (mode === 'stills') await stills(out, rest.map(Number));
  else if (mode === 'video') await video(out, Number(rest[0] || 60), Number(rest[1] || 20));
  else { console.error('usage: node render.js video|stills ...'); process.exit(1); }
})().catch((e) => { console.error(e); process.exit(1); });
