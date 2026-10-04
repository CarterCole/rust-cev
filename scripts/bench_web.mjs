#!/usr/bin/env node
// Latency and memory of the wasm build, run under Node (the same V8 as
// Chrome, so the numbers match a desktop browser tab). One model per run:
//
//     scripts/build_web.sh
//     node scripts/bench_web.mjs web/models/Qwen3-0.6B
//
// Same request as scripts/bench.py (short state, 3 questions). "first" is the
// very first request (calibration not cached yet), "new state" changes the
// state every call, "same state" repeats it (prefix cached).
import { createReadStream, readFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import init, { Cev, WeightLoader } from '../web/pkg/cev_wasm.js';

const dir = process.argv[2];
if (!dir) {
  console.error('usage: node scripts/bench_web.mjs <model dir> [runs]');
  process.exit(2);
}
const runs = Number(process.argv[3]) || 3;

const QUESTIONS = {
  team: {
    type: 'choice',
    instructions: 'Which team should handle this ticket?',
    criteria: { billing: 'Payments, invoices, refunds', tech: 'Bugs, crashes, errors', sales: 'New purchases and upgrades' },
  },
  refund: { type: 'noul', instructions: 'Is the customer asking for a refund?' },
  severity: {
    type: 'score',
    instructions: 'How severe is the issue?',
    criteria: ['Cosmetic; no impact', 'Degraded, workaround exists', 'Blocking; no workaround'],
  },
};
const TICKET = { subject: 'App crashes on login', body: 'Since the update, the iOS app crashes as soon as I tap Log in.' };

const wasm = await init({ module_or_path: readFileSync(new URL('../web/pkg/cev_wasm_bg.wasm', import.meta.url)) });
const gb = () => (wasm.memory.buffer.byteLength / 2 ** 30).toFixed(2);
const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];

let t = performance.now();
const loader = new WeightLoader();
for await (const chunk of createReadStream(join(dir, 'model.safetensors'), { highWaterMark: 4 << 20 })) loader.push(chunk);
let tokenizerConfig = '';
try {
  tokenizerConfig = readFileSync(join(dir, 'tokenizer_config.json'), 'utf8');
} catch {}
const cev = Cev.load(basename(dir), readFileSync(join(dir, 'config.json')), readFileSync(join(dir, 'tokenizer.json')), tokenizerConfig, loader);
console.log(`${basename(dir)}: loaded in ${((performance.now() - t) / 1000).toFixed(1)} s, wasm memory ${gb()} GB`);

let nonce = 0;
const decide = (state, debias) => {
  const t = performance.now();
  const r = JSON.parse(cev.decide(JSON.stringify({ state, questions: QUESTIONS, debias, no_store: true })));
  return [performance.now() - t, r.usage.input_tokens];
};
for (const debias of ['calibrate', 'full']) {
  const fresh = () => ({ ...TICKET, nonce: `run-${nonce++}` });
  const [first, tokens] = decide(fresh(), debias);
  const uncached = Array.from({ length: runs }, () => decide(fresh(), debias)[0]);
  const state = fresh();
  decide(state, debias);
  const cached = Array.from({ length: runs }, () => decide(state, debias)[0]);
  console.log(
    `  ${debias.padEnd(9)} ${tokens} tokens: first ${first.toFixed(0)} ms, new state ${median(uncached).toFixed(0)} ms, same state ${median(cached).toFixed(0)} ms`,
  );
}
console.log(`  peak wasm memory ${gb()} GB`);
