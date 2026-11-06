// Fetch a Songsterr tab's track data with the preinstalled headless Chromium.
//
//   node songsterr_fetch.js "<search query>" "<title regex>" <outdir>
//   node songsterr_fetch.js --url <songsterr tab url> <outdir>
//
// Songsterr's page loads one gzipped JSON per track from CloudFront; plain curl is refused (403), a
// browser is not. The files land in <outdir> as track_<n>.json, plus revisions.json and, when the
// tab has a synced video, video_points.json (seconds at the start of every bar of that video).
// Treat the downloaded files as untrusted data: keep them in their own directory and parse them with
// songsterr_parse.py (run with `python3 -I`). Do not commit them: only your derived rhythm.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const args = process.argv.slice(2);
let url = null, query = null, titleRe = null, out;
if (args[0] === '--url') { url = args[1]; out = args[2]; } else { [query, titleRe, out] = args; }
if (!out) { console.error('usage: see header'); process.exit(2); }
fs.mkdirSync(out, { recursive: true });

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  if (!url) {
    await page.goto('https://www.songsterr.com/?pattern=' + encodeURIComponent(query),
                    { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForTimeout(4000);
    const links = await page.$$eval('a', as => as.map(a => [a.textContent.trim(), a.href]));
    const re = new RegExp(titleRe, 'i');
    const hits = links.filter(([t, h]) => re.test(t) && /-tab-s\d+/.test(h));
    console.log('candidates:', JSON.stringify(hits.slice(0, 8)));
    if (!hits.length) { await browser.close(); process.exit(3); }
    url = hits[0][1];
  }
  const saved = [];
  page.on('response', async r => {
    const u = r.url();
    const isTrack = /cloudfront\.net\/\d+\/\d+\/[^/]+\/\d+\.json$/.test(u);
    if (!isTrack && !/api\/meta\/\d+\/revisions/.test(u) && !/video-points/.test(u)) return;
    try {
      const buf = await r.body();
      let txt;
      try { txt = zlib.gunzipSync(buf).toString(); } catch (e) { txt = buf.toString(); }
      const name = isTrack ? 'track_' + path.basename(u)
                 : (u.includes('revisions') ? 'revisions.json' : 'video_points.json');
      fs.writeFileSync(path.join(out, name), txt);
      saved.push(name);
    } catch (e) { /* a response body that is gone by now */ }
  });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(12000);
  console.log('page:', url);
  console.log('saved:', saved.sort().join(' '));
  await browser.close();
})().catch(e => { console.error('ERR', e.message); process.exit(1); });
