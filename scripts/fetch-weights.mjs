#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dest = path.join(root, 'harness', 'weights');
const base =
  'https://huggingface.co/onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX/resolve/main/onnx';
const files = ['model_q4.onnx_data', 'model_q4.onnx_data_1'];

fs.mkdirSync(dest, { recursive: true });
for (const f of files) {
  const out = path.join(dest, f);
  if (fs.existsSync(out) && fs.statSync(out).size > 1_000_000) {
    console.log('have', f, fs.statSync(out).size);
    continue;
  }
  console.log('fetch', f);
  const r = spawnSync(
    'curl',
    ['-L', '--retry', '5', '-C', '-', '-o', out, `${base}/${f}`],
    { stdio: 'inherit' },
  );
  if (r.status !== 0) process.exit(r.status ?? 1);
}
