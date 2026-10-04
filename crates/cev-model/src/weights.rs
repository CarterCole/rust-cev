//! Incremental safetensors reader for weights that arrive in chunks (a
//! browser `fetch` stream). Each tensor is converted to the serving dtype as
//! soon as its bytes are complete, so the file is never held in memory whole.

use anyhow::{Context, Result, bail};
pub use candle_core::DType;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

struct Entry {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct StreamLoader {
    dtype: DType,
    /// Tensors by data offset; `None` until the header has arrived.
    entries: Option<Vec<Entry>>,
    next: usize,
    /// Header bytes so far, then the bytes of tensor `next` so far.
    buf: Vec<u8>,
    /// Offset into the data section of the next byte to arrive.
    pos: usize,
    received: usize,
    tensors: HashMap<String, Tensor>,
}

fn dtype_of(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F64" => DType::F64,
        "I64" => DType::I64,
        "U32" => DType::U32,
        "U8" => DType::U8,
        other => bail!("unsupported safetensors dtype `{other}`"),
    })
}

impl StreamLoader {
    /// Float tensors are converted to `dtype` (CPU).
    pub fn new(dtype: DType) -> Self {
        Self { dtype, entries: None, next: 0, buf: Vec::new(), pos: 0, received: 0, tensors: HashMap::new() }
    }

    /// Bytes pushed so far.
    pub fn received(&self) -> usize {
        self.received
    }

    /// Append the next bytes of the file, in order.
    pub fn push(&mut self, chunk: &[u8]) -> Result<()> {
        self.received += chunk.len();
        if self.entries.is_some() {
            return self.data(chunk);
        }
        self.buf.extend_from_slice(chunk);
        if self.buf.len() < 8 {
            return Ok(());
        }
        let n = u64::from_le_bytes(self.buf[..8].try_into().expect("8 bytes")) as usize;
        if n > 100 << 20 {
            bail!("not a safetensors file (header of {n} bytes)");
        }
        if self.buf.len() < 8 + n {
            return Ok(());
        }
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&self.buf[8..8 + n]).context("safetensors header")?;
        let mut entries = Vec::with_capacity(header.len());
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let field = |k: &str| v.get(k).with_context(|| format!("{name}: missing `{k}`"));
            let offsets: [usize; 2] = serde_json::from_value(field("data_offsets")?.clone())?;
            let shape: Vec<usize> = serde_json::from_value(field("shape")?.clone())?;
            let dtype = dtype_of(field("dtype")?.as_str().unwrap_or_default())?;
            if offsets[1] < offsets[0] || offsets[1] - offsets[0] != shape.iter().product::<usize>() * dtype.size_in_bytes() {
                bail!("{name}: bad data offsets");
            }
            entries.push(Entry { name, dtype, shape, start: offsets[0], end: offsets[1] });
        }
        entries.sort_by_key(|e| e.start);
        self.entries = Some(entries);
        let rest = self.buf.split_off(8 + n);
        self.buf = Vec::new();
        self.data(&rest)
    }

    fn data(&mut self, mut chunk: &[u8]) -> Result<()> {
        let entries = self.entries.as_ref().expect("header parsed");
        while let Some(e) = entries.get(self.next) {
            if self.pos < e.start {
                // Padding between tensors.
                let skip = (e.start - self.pos).min(chunk.len());
                chunk = &chunk[skip..];
                self.pos += skip;
                if self.pos < e.start {
                    break;
                }
            }
            let size = e.end - e.start;
            if self.buf.is_empty() {
                self.buf.reserve_exact(size);
            }
            let take = (size - self.buf.len()).min(chunk.len());
            self.buf.extend_from_slice(&chunk[..take]);
            chunk = &chunk[take..];
            self.pos += take;
            if self.buf.len() < size {
                break;
            }
            let t = Tensor::from_raw_buffer(&self.buf, e.dtype, &e.shape, &Device::Cpu)?;
            self.buf = Vec::new();
            let t = if e.dtype.is_float() { t.to_dtype(self.dtype)? } else { t };
            self.tensors.insert(e.name.clone(), t);
            self.next += 1;
        }
        Ok(())
    }

    /// All tensors, once the whole file has been pushed.
    pub fn finish(self) -> Result<HashMap<String, Tensor>> {
        match &self.entries {
            Some(e) if self.next == e.len() => Ok(self.tensors),
            Some(e) => bail!("weights truncated: {} of {} tensors after {} bytes", self.next, e.len(), self.received),
            None => bail!("weights truncated: no header after {} bytes", self.received),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file with padding between tensors, fed in awkward chunk sizes.
    #[test]
    fn streams_in_any_chunking() {
        let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let b = [half::bf16::from_f32(1.5), half::bf16::from_f32(-2.0)];
        let header = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"BF16","shape":[2],"data_offsets":[28,32]},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend(a.iter().flat_map(|x| x.to_le_bytes()));
        file.extend([0u8; 4]);
        file.extend(b.iter().flat_map(|x| x.to_le_bytes()));
        for chunk in [1, 3, 7, 64, file.len()] {
            let mut l = StreamLoader::new(DType::F32);
            for c in file.chunks(chunk) {
                l.push(c).unwrap();
            }
            assert_eq!(l.received(), file.len());
            let t = l.finish().unwrap();
            assert_eq!(t["a"].to_vec2::<f32>().unwrap(), vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]);
            assert_eq!(t["b"].to_vec1::<f32>().unwrap(), vec![1.5, -2.0]);
        }
        let mut l = StreamLoader::new(DType::F32);
        l.push(&file[..file.len() - 1]).unwrap();
        assert!(l.finish().is_err());
    }
}
