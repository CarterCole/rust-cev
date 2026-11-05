// Runs cev (wasm) off the main thread. Messages: {id, op, args} in,
// {id, result | error} out, plus {progress} while a model loads.
import init, { Cev, ModelFiles, WeightLoader } from './pkg/cev_wasm.js';

let cev = null;
const ready = init().then(() => {
  cev = Cev.mock();
});

const progress = (text, done = 0, total = 0) => postMessage({ progress: { text, done, total } });

// What each model needs, repository by repository. `weights` is streamed
// into wasm memory tensor by tensor; the other files are read whole. Each
// repository is looked for in web/models/<repo name>/ first (a symlink to a
// Hugging Face snapshot directory works), then fetched from huggingface.co
// at the pinned revision.
const MODELS = {
  'qwen3-0.6b': {
    kind: 'qwen3',
    repo: 'Qwen/Qwen3-0.6B',
    parts: [{ repo: 'Qwen/Qwen3-0.6B', files: ['config.json', 'tokenizer.json'], optional: ['tokenizer_config.json'], weights: 'model.safetensors' }],
  },
  // The int8 copy made by `cev-model`'s quantize example; local only.
  'qwen3-0.6b-q8': {
    kind: 'qwen3',
    repo: 'Qwen/Qwen3-0.6B-q8',
    localOnly: true,
    parts: [{ repo: 'Qwen/Qwen3-0.6B-q8', files: ['config.json', 'tokenizer.json'], optional: ['tokenizer_config.json'], weights: 'model.safetensors' }],
  },
  // A LoRA adapter and pointer head over Qwen3.5-0.8B-Base, merged in wasm.
  'kev-0.8b': {
    kind: 'kev',
    repo: 'jaredpalmer/kev-0.8b',
    parts: [
      { repo: 'jaredpalmer/kev-0.8b', rev: 'bf75a6a8848ea6960ff2ed108d9ed44c2941174f', files: ['adapter_config.json', 'adapter_model.safetensors', 'head.pt'] },
      {
        repo: 'Qwen/Qwen3.5-0.8B-Base',
        rev: 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
        files: ['config.json', 'tokenizer.json'],
        weights: 'model.safetensors-00001-of-00001.safetensors',
      },
    ],
  },
  'd1-omni-600m': {
    kind: 'd1-omni',
    repo: 'LiquidAI/d1-omni-600M',
    parts: [{ repo: 'LiquidAI/d1-omni-600M', rev: '02b55d7076f15129e59ab3f94783f32c4b088674', files: ['config.json', 'tokenizer.json'], weights: 'model.safetensors' }],
  },
};

// Files from huggingface.co are kept in Cache Storage, so a reload does not
// download them again. Local files (web/models/) are just fetched.
async function get(url, cache) {
  const hit = cache && (await cache.match(url));
  if (hit) return hit;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`GET ${url}: HTTP ${resp.status}`);
  if (!cache || !resp.body) return resp;
  const [a, b] = resp.body.tee();
  cache.put(url, new Response(b, { headers: resp.headers })).catch(() => {});
  return new Response(a, { headers: resp.headers });
}

const dirName = (repo) => repo.split('/').pop();
const isLocal = (repo, file = 'config.json') =>
  fetch(`models/${dirName(repo)}/${file}`, { method: 'HEAD' }).then((r) => r.ok, () => false);

// Where one repository's files come from.
async function source(part, localOnly) {
  const first = part.files[0];
  if (await isLocal(part.repo, first)) return { base: `models/${dirName(part.repo)}`, cache: null };
  if (localOnly) throw new Error(`web/models/${dirName(part.repo)}/ not found`);
  const cache = await caches.open('cev-models').catch(() => null);
  return { base: `https://huggingface.co/${part.repo}/resolve/${part.rev ?? 'main'}`, cache };
}

async function load(key) {
  const spec = MODELS[key];
  if (!spec) throw new Error(`unknown model ${key}`);
  const files = new ModelFiles();
  const streams = [];
  try {
    for (const part of spec.parts) {
      const { base, cache } = await source(part, spec.localOnly);
      for (const file of [...part.files, ...(part.optional ?? [])]) {
        progress(`Fetching ${file}…`);
        let resp;
        try {
          resp = await get(`${base}/${file}`, cache);
        } catch (e) {
          if (part.files.includes(file)) throw e;
          continue;
        }
        files.add(file, new Uint8Array(await resp.arrayBuffer()));
      }
      if (part.weights) streams.push(await get(`${base}/${part.weights}`, cache));
    }
  } catch (e) {
    files.free();
    throw e;
  }
  const total = streams.reduce((n, r) => n + (Number(r.headers.get('content-length')) || 0), 0);
  // Two models do not fit in wasm memory: drop the current one first.
  cev.free();
  cev = Cev.mock();
  const loader = WeightLoader.for_model(spec.kind);
  let done = 0;
  let shown = 0;
  try {
    for (const resp of streams) {
      const reader = resp.body.getReader();
      for (;;) {
        const { value, done: end } = await reader.read();
        if (end) break;
        loader.push(value);
        done += value.length;
        if (done - shown > 8 << 20) progress('Loading weights', (shown = done), total);
      }
    }
  } catch (e) {
    loader.free();
    files.free();
    throw e;
  }
  progress('Building model…', done, total);
  const next = Cev.load_files(spec.repo, files, loader);
  cev.free();
  cev = next;
  return JSON.parse(cev.model());
}

const ops = {
  load,
  // Models whose files are all in web/models/ (the int8 copy is offered only then).
  has_local: async (key) => {
    for (const p of MODELS[key].parts) if (!(await isLocal(p.repo, p.files[0]))) return false;
    return true;
  },
  model: () => JSON.parse(cev.model()),
  decide: (req) => JSON.parse(cev.decide(JSON.stringify(req))),
  feedback: (req) => JSON.parse(cev.feedback(JSON.stringify(req))),
  tasks: () => JSON.parse(cev.tasks()),
  reset_task: (task) => cev.reset_task(task),
  export: (labeled) => cev.export(labeled),
  stats: () => JSON.parse(cev.stats()),
};

onmessage = async ({ data: { id, op, args } }) => {
  try {
    await ready;
    postMessage({ id, result: await ops[op](...args) });
  } catch (e) {
    postMessage({ id, error: String(e?.message ?? e) });
  }
};
