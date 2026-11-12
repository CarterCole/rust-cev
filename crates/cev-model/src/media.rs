//! Images and audio as the models take them: decoding, the resize rules of
//! each family's processor, and patch / frame layouts.

use anyhow::{Result, bail};
use cev_core::InputError;
use image::{DynamicImage, GenericImageView};

/// Separable resampling weights of `F.interpolate(..., antialias=True)`:
/// a triangle (bilinear) or Keys cubic (`a = -0.5`, bicubic) kernel whose
/// support widens by the downscale factor, taps clamped to the image and
/// normalised, as torchvision and PIL do.
pub fn antialias_weights(inp: usize, out: usize, bicubic: bool) -> Vec<Vec<(usize, f32)>> {
    let scale = inp as f64 / out as f64;
    let support = if bicubic { 2.0 } else { 1.0 } * scale.max(1.0);
    let kernel = |x: f64| -> f64 {
        if bicubic {
            let a = -0.5;
            let x = x.abs();
            if x < 1.0 {
                ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                (((x - 5.0) * x + 8.0) * x - 4.0) * a
            } else {
                0.0
            }
        } else {
            (1.0 - x.abs()).max(0.0)
        }
    };
    let inv = 1.0 / scale.max(1.0);
    (0..out)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            let lo = (center - support + 0.5).floor().max(0.0) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(inp);
            let mut taps: Vec<(usize, f32)> = (lo..hi).map(|i| (i, kernel(((i as f64 + 0.5) - center) * inv) as f32)).filter(|(_, w)| *w != 0.0).collect();
            let sum: f32 = taps.iter().map(|(_, w)| w).sum();
            for t in &mut taps {
                t.1 /= sum.max(1e-12);
            }
            taps
        })
        .collect()
}

/// Resize an RGB image to `w × h` the way torchvision's antialiased
/// `resize` does on 8-bit images (bilinear or bicubic): the width pass, its
/// result rounded back to whole levels, then the height pass. Returns
/// `h × w × 3` values in 0..=255, rounded.
pub fn resize_exact(img: &image::RgbImage, w: u32, h: u32, bicubic: bool) -> Vec<f32> {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let (w1, h1) = (w as usize, h as usize);
    let src = img.as_raw();
    // Width pass: (h0, w1, 3).
    let ww = antialias_weights(w0, w1, bicubic);
    let mut tmp = vec![0f32; h0 * w1 * 3];
    for y in 0..h0 {
        for (x, taps) in ww.iter().enumerate() {
            for c in 0..3 {
                tmp[(y * w1 + x) * 3 + c] = taps.iter().map(|&(i, wt)| wt * src[(y * w0 + i) * 3 + c] as f32).sum::<f32>().round().clamp(0.0, 255.0);
            }
        }
    }
    // Height pass: (h1, w1, 3).
    let hw = antialias_weights(h0, h1, bicubic);
    let mut out = vec![0f32; h1 * w1 * 3];
    for (y, taps) in hw.iter().enumerate() {
        for x in 0..w1 {
            for c in 0..3 {
                out[(y * w1 + x) * 3 + c] = taps.iter().map(|&(i, wt)| wt * tmp[(i * w1 + x) * 3 + c]).sum::<f32>().round().clamp(0.0, 255.0);
            }
        }
    }
    out
}

/// An image as f32 RGB planes `h × w × 3` in 0..=255, resized if needed.
pub struct Pixels {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

impl Pixels {
    /// The processors resize 8-bit images and keep 8-bit results, so the
    /// resampled values are rounded back to whole levels before rescaling.
    pub fn from_image(img: &DynamicImage, w: u32, h: u32, bicubic: bool) -> Self {
        let rgb = img.to_rgb8();
        let data = if (rgb.width(), rgb.height()) == (w, h) { rgb.as_raw().iter().map(|&v| v as f32).collect() } else { resize_exact(&rgb, w, h, bicubic) };
        Self { data, w: w as usize, h: h as usize }
    }

    fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[(y * self.w + x) * 3 + c]
    }

    /// Crop a `size × size` tile at (x0, y0).
    fn tile(&self, x0: usize, y0: usize, size: usize) -> Pixels {
        let mut data = Vec::with_capacity(size * size * 3);
        for y in 0..size {
            for x in 0..size {
                for c in 0..3 {
                    data.push(self.at(x0 + x, y0 + y, c));
                }
            }
        }
        Pixels { data, w: size, h: size }
    }
}

/// Decode an encoded image (PNG, JPEG, WebP, GIF, BMP, TIFF) into RGB.
///
/// Sides are capped at [`MAX_IMAGE_SIDE`] (checked from the header, before
/// the pixels are allocated) and the aspect ratio at [`MAX_ASPECT`], which
/// keeps every resize buffer downstream in the hundreds of MB at most.
pub fn decode_image(bytes: &[u8]) -> Result<DynamicImage> {
    let bad = |m: String| anyhow::Error::new(InputError(m));
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| bad(format!("cannot decode image: {e}")))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    reader.limits(limits);
    let img = reader.decode().map_err(|e| bad(format!("cannot decode image: {e}")))?;
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(bad("empty image".into()));
    }
    if w.max(h) as f64 / w.min(h) as f64 > MAX_ASPECT {
        return Err(bad(format!("image is {w}×{h}; the aspect ratio must be at most {MAX_ASPECT}")));
    }
    Ok(DynamicImage::ImageRgb8(img.to_rgb8()))
}

/// Longest side an input image may have (a 48 MP phone photo fits).
pub const MAX_IMAGE_SIDE: u32 = 8192;
/// Most elongated image accepted (Qwen-VL's processor has the same limit).
pub const MAX_ASPECT: f64 = 200.0;

/// Qwen-VL's `smart_resize`: both sides multiples of `factor`, the pixel
/// count within `[min_pixels, max_pixels]`, the aspect ratio kept as closely
/// as possible. Returns (height, width).
pub fn smart_resize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) -> Result<(u32, u32)> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        bail!("image aspect ratio must be below 200");
    }
    let f = factor as f64;
    let round = |x: f64| ((x / f).round() * f).max(f) as u64;
    let (mut hb, mut wb) = (round(h), round(w));
    if hb * wb > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = (((h / beta / f).floor() * f) as u64).max(factor as u64);
        wb = (((w / beta / f).floor() * f) as u64).max(factor as u64);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = ((h * beta / f).ceil() * f) as u64;
        wb = ((w * beta / f).ceil() * f) as u64;
    }
    Ok((hb as u32, wb as u32))
}

/// One image prepared for the Qwen3.5 vision tower: flattened patches in
/// spatial-merge-block order, normalised to [-1, 1], each patch duplicated
/// over the two temporal frames.
pub struct QwenPatches {
    pub pixels: Vec<f32>,
    pub grid: crate::qwen35_vision::Grid,
    pub patch_dim: usize,
}

pub struct QwenImageParams {
    pub patch: u32,
    pub merge: u32,
    pub temporal: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for QwenImageParams {
    fn default() -> Self {
        // d3's budget: 1.6 MP at most, 65,536 pixels at least.
        Self { patch: 16, merge: 2, temporal: 2, min_pixels: 65_536, max_pixels: 1_638_400, mean: [0.5; 3], std: [0.5; 3] }
    }
}

pub fn qwen_patches(img: &DynamicImage, p: &QwenImageParams) -> Result<QwenPatches> {
    let (w0, h0) = img.dimensions();
    let (h, w) = smart_resize(h0, w0, p.patch * p.merge, p.min_pixels, p.max_pixels)?;
    let resized = Pixels::from_image(img, w, h, true);
    let (gh, gw) = ((h / p.patch) as usize, (w / p.patch) as usize);
    let ps = p.patch as usize;
    let patch_dim = 3 * p.temporal as usize * ps * ps;
    let order = crate::qwen35_vision::block_order(gh, gw, p.merge as usize);
    let mut pixels = Vec::with_capacity(order.len() * patch_dim);
    for &(r, c) in &order {
        for ch in 0..3 {
            let plane: Vec<f32> = (0..ps)
                .flat_map(|py| (0..ps).map(move |px| (py, px)))
                .map(|(py, px)| {
                    let v = resized.at(c * ps + px, r * ps + py, ch) / 255.0;
                    (v - p.mean[ch]) / p.std[ch]
                })
                .collect();
            for _ in 0..p.temporal {
                pixels.extend_from_slice(&plane);
            }
        }
    }
    Ok(QwenPatches { pixels, grid: crate::qwen35_vision::Grid { t: 1, h: gh, w: gw }, patch_dim })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        image::GrayImage::new(w, h).write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn decode_bounds_sides_and_aspect() {
        assert!(decode_image(&png(64, 48)).is_ok());
        assert!(decode_image(&png(1, 201)).unwrap_err().to_string().contains("aspect"));
        // Rejected from the header, before 8193 × 8193 pixels are allocated.
        assert!(decode_image(&png(MAX_IMAGE_SIDE + 1, MAX_IMAGE_SIDE + 1)).is_err());
    }

    #[test]
    fn antialias_weights_match_torch() {
        // F.interpolate(16 -> 7, bilinear, antialias): output 1 reads source
        // column 3 with weight 0.4133 and output 2 with 0.0137.
        let w = antialias_weights(16, 7, false);
        let at = |o: usize, i: usize| w[o].iter().find(|t| t.0 == i).map_or(0.0, |t| t.1);
        assert!((at(1, 3) - 0.4133).abs() < 1e-3, "{}", at(1, 3));
        assert!((at(2, 3) - 0.0137).abs() < 1e-3, "{}", at(2, 3));
        // Upsampling 16 -> 32: output 1 is 0.75 of column 0 and 0.25 of column 1.
        let up = antialias_weights(16, 32, false);
        assert!((up[0][0].1 - 1.0).abs() < 1e-6);
        assert!((up[1][0].1 - 0.75).abs() < 1e-6 && (up[1][1].1 - 0.25).abs() < 1e-6);
        for taps in antialias_weights(100, 37, true) {
            let s: f32 = taps.iter().map(|t| t.1).sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn smart_resize_rules() {
        assert_eq!(smart_resize(480, 640, 32, 65_536, 1_638_400).unwrap(), (480, 640));
        // Too big: scaled down to about 1.6 MP, multiples of 32.
        let (h, w) = smart_resize(3000, 4000, 32, 65_536, 1_638_400).unwrap();
        assert!(h % 32 == 0 && w % 32 == 0 && (h as u64) * (w as u64) <= 1_638_400);
        assert!((w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.05);
        // Too small: scaled up to at least 65,536 pixels.
        let (h, w) = smart_resize(50, 50, 32, 65_536, 1_638_400).unwrap();
        assert!((h as u64) * (w as u64) >= 65_536 && h % 32 == 0);
    }

    #[test]
    fn patches_layout() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(64, 32, |x, y| image::Rgb([x as u8 * 4, y as u8 * 8, 0])));
        let p = QwenImageParams { min_pixels: 0, ..Default::default() };
        let q = qwen_patches(&img, &p).unwrap();
        assert_eq!((q.grid.h, q.grid.w), (2, 4));
        assert_eq!(q.pixels.len(), 8 * q.patch_dim);
        // Second patch in block order is (row 0, col 1): red channel starts at x = 16 -> 64/255 -> (0.251 - 0.5) / 0.5.
        let first_red_of_patch1 = q.pixels[q.patch_dim];
        assert!((first_red_of_patch1 - ((64.0 / 255.0 - 0.5) / 0.5)).abs() < 1e-5);
        // Temporal copies are identical.
        let plane = 16 * 16;
        assert_eq!(q.pixels[..plane], q.pixels[plane..2 * plane]);
    }
}


/// LFM2-VL's image layout: a small image is one crop at its "smart" size; a
/// large one is cut into a grid of 512 px tiles (2 to 10) plus a thumbnail.
/// Every crop becomes (h/16)·(w/16) SigLIP2 patches and, after the 2×2
/// projector, h·w/1024 tokens.
#[derive(Debug, Clone)]
pub struct LfmTileParams {
    pub tile: u32,
    pub patch: u32,
    pub downsample: u32,
    pub min_image_tokens: u64,
    pub max_image_tokens: u64,
    pub max_pixels_tolerance: f64,
    pub min_tiles: u32,
    pub max_tiles: u32,
    pub use_thumbnail: bool,
    pub bicubic: bool,
}

impl Default for LfmTileParams {
    fn default() -> Self {
        Self { tile: 512, patch: 16, downsample: 2, min_image_tokens: 64, max_image_tokens: 256, max_pixels_tolerance: 2.0, min_tiles: 2, max_tiles: 10, use_thumbnail: true, bicubic: true }
    }
}

/// One crop prepared for SigLIP2: raster-order patches, channel-last within
/// a patch, in [-1, 1]; the grid is `(h, w)` patches.
pub struct Crop {
    pub pixels: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// A tiled image: the tiles (row-major, `rows × cols`), then the thumbnail
/// (or the single crop when the image was not split).
pub struct LfmImage {
    pub crops: Vec<Crop>,
    pub rows: usize,
    pub cols: usize,
    pub tiled: bool,
}

fn lfm_smart_resize(height: u32, width: u32, p: &LfmTileParams) -> (u32, u32) {
    let factor = (p.patch * p.downsample) as f64;
    let (h, w) = (height as f64, width as f64);
    let min_px = (p.min_image_tokens * (p.patch as u64).pow(2) * (p.downsample as u64).pow(2)) as f64;
    let max_px = (p.max_image_tokens * (p.patch as u64).pow(2) * (p.downsample as u64).pow(2)) as f64;
    let round = |x: f64| ((x / factor).round() * factor).max(factor);
    let (mut hb, mut wb) = (round(h), round(w));
    if hb * wb > max_px {
        let beta = (h * w / max_px).sqrt();
        hb = ((h / beta / factor).floor() * factor).max(factor);
        wb = ((w / beta / factor).floor() * factor).max(factor);
    } else if hb * wb < min_px {
        let beta = (min_px / (h * w)).sqrt();
        hb = (h * beta / factor).ceil() * factor;
        wb = (w * beta / factor).ceil() * factor;
    }
    (hb as u32, wb as u32)
}

fn lfm_is_large(height: u32, width: u32, p: &LfmTileParams) -> bool {
    let factor = (p.patch * p.downsample) as f64;
    let round = |x: f64| ((x / factor).round() * factor).max(p.patch as f64);
    let (hb, wb) = (round(height as f64), round(width as f64));
    hb * wb > p.max_image_tokens as f64 * (p.patch as f64).powi(2) * (p.downsample as f64).powi(2) * p.max_pixels_tolerance
}

/// The tile grid (cols, rows) whose aspect ratio is closest to the image's
/// (`find_closest_aspect_ratio`, ties broken toward more tiles for big images).
fn lfm_grid(height: u32, width: u32, p: &LfmTileParams) -> (usize, usize) {
    let mut ratios: Vec<(u32, u32)> = Vec::new();
    for n in p.min_tiles..=p.max_tiles {
        for w in 1..=n {
            for h in 1..=n {
                if (p.min_tiles..=p.max_tiles).contains(&(w * h)) && !ratios.contains(&(w, h)) {
                    ratios.push((w, h));
                }
            }
        }
    }
    ratios.sort_by_key(|(w, h)| w * h);
    let aspect = width as f64 / height as f64;
    let area = (width as f64) * (height as f64);
    let mut best = (1u32, 1u32);
    let mut best_diff = f64::INFINITY;
    for (w, h) in ratios {
        let diff = (aspect - w as f64 / h as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (w, h);
        } else if diff == best_diff && area > 0.5 * (p.tile as f64).powi(2) * (w * h) as f64 {
            best = (w, h);
        }
    }
    (best.0 as usize, best.1 as usize)
}

fn siglip_patches(img: &Pixels, patch: u32) -> Crop {
    let (ph, pw) = (img.h / patch as usize, img.w / patch as usize);
    let ps = patch as usize;
    let mut pixels = Vec::with_capacity(ph * pw * ps * ps * 3);
    for r in 0..ph {
        for c in 0..pw {
            for py in 0..ps {
                for px in 0..ps {
                    for ch in 0..3 {
                        pixels.push((img.at(c * ps + px, r * ps + py, ch) / 255.0 - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    Crop { pixels, h: ph, w: pw }
}

/// Resize and tile an image the way LFM2-VL's processor does.
pub fn lfm_image(img: &DynamicImage, p: &LfmTileParams) -> LfmImage {
    let (w0, h0) = img.dimensions();
    let (nh, nw) = lfm_smart_resize(h0, w0, p);
    let split = p.min_tiles != 1 || p.max_tiles != 1;
    if lfm_is_large(h0, w0, p) && split {
        let (cols, rows) = lfm_grid(h0, w0, p);
        let big = Pixels::from_image(img, cols as u32 * p.tile, rows as u32 * p.tile, p.bicubic);
        let mut crops = Vec::with_capacity(rows * cols + 1);
        let t = p.tile as usize;
        for r in 0..rows {
            for c in 0..cols {
                crops.push(siglip_patches(&big.tile(c * t, r * t, t), p.patch));
            }
        }
        if p.use_thumbnail && rows * cols != 1 {
            crops.push(siglip_patches(&Pixels::from_image(img, nw, nh, p.bicubic), p.patch));
        }
        LfmImage { crops, rows, cols, tiled: true }
    } else {
        LfmImage { crops: vec![siglip_patches(&Pixels::from_image(img, nw, nh, p.bicubic), p.patch)], rows: 1, cols: 1, tiled: false }
    }
}

impl LfmImage {
    /// Tokens the language model sees for each crop (after the projector).
    pub fn tokens_per_crop(&self, downsample: usize) -> Vec<usize> {
        self.crops.iter().map(|c| c.h * c.w / (downsample * downsample)).collect()
    }

    /// The processor's expansion of one `<image>`: tile markers and image
    /// tokens, then the thumbnail, inside `<|image_start|>` / `<|image_end|>`.
    pub fn markup(&self, downsample: usize) -> String {
        let per = self.tokens_per_crop(downsample);
        let mut s = String::from("<|image_start|>");
        if self.tiled {
            let mut k = 0;
            for r in 0..self.rows {
                for c in 0..self.cols {
                    s.push_str(&format!("<|img_row_{}_col_{}|>", r + 1, c + 1));
                    s.push_str(&"<image>".repeat(per[k]));
                    k += 1;
                }
            }
            if k < per.len() {
                s.push_str("<|img_thumbnail|>");
                s.push_str(&"<image>".repeat(per[k]));
            }
        } else {
            s.push_str(&"<image>".repeat(per[0]));
        }
        s.push_str("<|image_end|>");
        s
    }
}

#[cfg(test)]
mod lfm_tests {
    use super::*;

    #[test]
    fn lfm_layout() {
        let p = LfmTileParams::default();
        // 640×480 (307k px) is under the 2× tolerance of the 256-token budget
        // (524k px), so it is one crop scaled down to 416×576: a 26×36 grid.
        let small = DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));
        let li = lfm_image(&small, &p);
        assert!(!li.tiled && li.crops.len() == 1);
        assert_eq!((li.crops[0].h, li.crops[0].w), (26, 36));
        assert_eq!(li.markup(2), format!("<|image_start|>{}<|image_end|>", "<image>".repeat(26 * 36 / 4)));
        let big = DynamicImage::ImageRgb8(image::RgbImage::new(2000, 1000));
        let li = lfm_image(&big, &p);
        assert!(li.tiled && (li.cols, li.rows) == (2, 1) || (li.cols, li.rows) == (4, 2) || li.cols * li.rows <= 10);
        assert_eq!(li.crops.len(), li.rows * li.cols + 1);
        assert!(li.markup(2).contains("<|img_row_1_col_1|>") && li.markup(2).contains("<|img_thumbnail|>"));
        assert_eq!(li.tokens_per_crop(2)[0], 256);
    }
}
