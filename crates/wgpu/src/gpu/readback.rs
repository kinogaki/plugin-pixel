//! Copies a texture to the CPU (blocking: for final frames, previews and
//! tests, never the viewport's path).

use kinogaki_pixel_render::{AovImage, AovPixels};

/// IEEE half to single precision.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        // Subnormal: value = frac × 2⁻²⁴.
        (0, _) => {
            let v = frac as f32 * (-24f32).exp2();
            return if sign != 0 { -v } else { v };
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// The bytes of `texture`'s texels, rows from the top, tightly packed
/// (waits for the GPU).
fn bytes(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let texel = texture.format().block_copy_size(None).unwrap_or(4);
    let row = (w * texel).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel readback"),
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        texture.size(),
    );
    let index = queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("a mapped readback"));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("the readback copy");
    let mapped = slice.get_mapped_range().expect("the mapped bytes");
    mapped
        .chunks(row as usize)
        .flat_map(|r| &r[..(w * texel) as usize])
        .copied()
        .collect()
}

/// The half floats (their bits) of an Rgba16Float `texture`: what a
/// `kinogaki_render_api::Frame` carries.
pub fn halves(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u16> {
    bytes(device, queue, texture)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
        .collect()
}

/// The texels of `texture` (Rgba16Float, R32Float or R32Uint).
pub fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> AovImage {
    let (width, height) = (texture.width(), texture.height());
    let bytes = bytes(device, queue, texture);
    let words = || bytes.as_chunks::<4>().0.iter().copied();
    let (channels, pixels) = match texture.format() {
        wgpu::TextureFormat::Rgba16Float => (
            4,
            AovPixels::F32(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| f16_to_f32(u16::from_le_bytes(b)))
                    .collect(),
            ),
        ),
        wgpu::TextureFormat::R32Float => {
            (1, AovPixels::F32(words().map(f32::from_le_bytes).collect()))
        }
        _ => (1, AovPixels::U32(words().map(u32::from_le_bytes).collect())),
    };
    AovImage {
        width,
        height,
        channels,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn halves_widen_exactly() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }
}
