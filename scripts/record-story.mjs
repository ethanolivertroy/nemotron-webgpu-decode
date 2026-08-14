#!/usr/bin/env node
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const videoDir = path.join(root, 'video');
const rawDir = path.join(videoDir, 'raw');
const stillsDir = path.join(videoDir, 'stills');
const CHROME =
  process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = Number(process.env.PORT || 8765);
const DURATION_S = 24;
const stillsOnly = process.argv.includes('--stills');
const skipStills = process.argv.includes('--skip-stills');
const STILLS = [2.0, 5.5, 9.0, 12.2, 15.0, 18.0, 21.0, 23.5];

function contentType(file) {
  if (file.endsWith('.html')) return 'text/html; charset=utf-8';
  if (file.endsWith('.js')) return 'text/javascript; charset=utf-8';
  return 'application/octet-stream';
}

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const name = decodeURIComponent((req.url || '/').split('?')[0]);
      const file = path.join(videoDir, name === '/' ? 'story.html' : name.replace(/^\//, ''));
      if (!file.startsWith(videoDir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.statusCode = 404;
        res.end('not found');
        return;
      }
      res.setHeader('content-type', contentType(file));
      fs.createReadStream(file).pipe(res);
    });
    server.listen(PORT, '127.0.0.1', () => resolve(server));
  });
}

async function captureStills(browser) {
  fs.mkdirSync(stillsDir, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  for (const t of STILLS) {
    const out = path.join(stillsDir, `t${String(t).replace('.', 'p')}.png`);
    await page.goto(`http://127.0.0.1:${PORT}/story.html?t=${t}&record=1`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(200);
    await page.screenshot({ path: out, type: 'png' });
    console.log('still', out);
  }
  await page.close();
}

async function recordVideo(browser) {
  fs.mkdirSync(rawDir, { recursive: true });
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  await page.goto(`http://127.0.0.1:${PORT}/story.html?t=0&record=1`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => typeof window.draw === 'function');

  const fps = 60;
  const frames = Math.round(DURATION_S * fps);
  const mp4 = path.join(videoDir, 'nemotron-decode.mp4');
  const ff = spawn('ffmpeg', [
    '-y',
    '-f', 'image2pipe',
    '-framerate', String(fps),
    '-c:v', 'mjpeg',
    '-i', 'pipe:0',
    '-an',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-crf', '18',
    '-movflags', '+faststart',
    mp4,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });

  for (let i = 0; i <= frames; i++) {
    const t = i / fps;
    await page.evaluate((time) => window.draw(time), t);
    const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) {
      await new Promise((resolve) => ff.stdin.once('drain', resolve));
    }
    if (i % 30 === 0) console.log(`frame ${i}/${frames} t=${t.toFixed(2)}`);
  }
  ff.stdin.end();
  await new Promise((resolve, reject) => {
    ff.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg ${code}`))));
  });
  await page.close();
  console.log('wrote', mp4, fs.statSync(mp4).size, 'bytes');
}

const server = await startServer();
const browser = await chromium.launch({
  executablePath: CHROME,
  args: ['--no-first-run', '--no-default-browser-check'],
});
try {
  if (!skipStills) await captureStills(browser);
  if (!stillsOnly) await recordVideo(browser);
} finally {
  await browser.close();
  server.close();
}
