#!/usr/bin/env node
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CHROME =
  process.env.CHROME_PATH ||
  '/root/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome';
const PORT = process.env.PORT || '5174';
const outPng = path.join(root, 'charts', 'chart.png');

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

const vite = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', PORT], {
  cwd: root,
  stdio: ['ignore', 'pipe', 'pipe'],
});
let ready = false;
vite.stdout.on('data', (d) => {
  const s = d.toString();
  if (s.includes('Local:') || s.includes('localhost')) ready = true;
});
const t0 = Date.now();
while (!ready && Date.now() - t0 < 30000) await sleep(200);
if (!ready) {
  vite.kill();
  process.exit(1);
}

const chromeArgs = [
  '--headless=new',
  '--no-sandbox',
  '--disable-gpu-sandbox',
  '--window-size=1400,900',
  `--screenshot=${outPng}`,
  '--virtual-time-budget=8000',
  `http://127.0.0.1:${PORT}/charts/chart.html`,
];
console.log('screenshot', outPng);
const chrome = spawn(CHROME, chromeArgs, { stdio: 'inherit' });
await new Promise((resolve) => chrome.on('exit', resolve));
vite.kill();
if (!fs.existsSync(outPng)) {
  console.error('screenshot missing');
  process.exit(1);
}
console.log('wrote', outPng, fs.statSync(outPng).size, 'bytes');
