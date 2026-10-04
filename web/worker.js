// Runs cev (wasm) off the main thread. Messages: {id, op, args} in,
// {id, result | error} out, plus {progress} while a model loads.
import init, { Cev, WeightLoader } from './pkg/cev_wasm.js';

let cev = null;
const ready = init().then(() => {
  cev = Cev.mock();
});

const progress = (text, done = 0, total = 0) => postMessage({ progress: { text, done, total } });

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

const isLocal = (name) => fetch(`models/${name}/config.json`, { method: 'HEAD' }).then((r) => r.ok, () => false);

// `variant` "q8" loads the int8 copy made by `cev-model`'s quantize example,
// which only exists locally (web/models/<name>-q8/).
async function load(repo, variant) {
  if (variant) repo = `${repo}-${variant}`;
  const name = repo.split('/').pop();
  let base = `models/${name}`;
  let cache = null;
  const local = await isLocal(name);
  if (!local && variant) throw new Error(`web/models/${name}/ not found`);
  if (!local) {
    base = `https://huggingface.co/${repo}/resolve/main`;
    cache = await caches.open('cev-models').catch(() => null);
  }
  progress('Fetching tokenizer…');
  const bytes = async (file) => new Uint8Array(await (await get(`${base}/${file}`, cache)).arrayBuffer());
  const config = await bytes('config.json');
  const tokenizer = await bytes('tokenizer.json');
  const tokenizerConfig = await get(`${base}/tokenizer_config.json`, cache).then((r) => r.text(), () => '');

  const resp = await get(`${base}/model.safetensors`, cache);
  const total = Number(resp.headers.get('content-length')) || 0;
  // Two models do not fit in wasm memory: drop the current one first.
  cev.free();
  cev = Cev.mock();
  const loader = new WeightLoader();
  const reader = resp.body.getReader();
  let done = 0;
  let shown = 0;
  try {
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      loader.push(value);
      done += value.length;
      if (done - shown > 8 << 20) {
        progress('Loading weights', (shown = done), total);
      }
    }
  } catch (e) {
    loader.free();
    throw e;
  }
  progress('Building model…', done, total);
  const next = Cev.load(repo, config, tokenizer, tokenizerConfig, loader);
  cev.free();
  cev = next;
  return JSON.parse(cev.model());
}

const ops = {
  load,
  has_local: isLocal,
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
