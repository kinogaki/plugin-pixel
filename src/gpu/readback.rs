//! Textures read back to the CPU: a frame's pixels for a host that asks for
//! them, a final frame's outputs and a preview's picture. Each read waits
//! for the GPU, on the render thread, a band of rows at a time through one
//! buffer of at most `BAND` bytes (a large final's outputs are more than a
//! buffer of the device holds).
use kinogaki_render_api::display::wgpu;

/// The most bytes a band of rows reads through.
const BAND: u64 = 64 << 20;

/// The bytes of `texture` (a whole 2D texture of a 4-, 8- or 16-byte texel
/// format), rows from the top, tightly packed; zeros where the device was
/// lost.
pub fn bytes(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let texel = texture.format().block_copy_size(None).unwrap_or(4);
    let row = width * texel;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let most = BAND.min(device.limits().max_buffer_size);
    let rows = u32::try_from(most / u64::from(padded)).map_or(height, |r| r.clamp(1, height));
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel readback"),
        size: u64::from(padded) * u64::from(rows),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut out = Vec::with_capacity(row as usize * height as usize);
    for top in (0..height).step_by(rows as usize) {
        let band = rows.min(height - top);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pixel readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                origin: wgpu::Origin3d { x: 0, y: top, z: 0 },
                ..texture.as_image_copy()
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(band),
                },
            },
            wgpu::Extent3d {
                width,
                height: band,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = buffer.slice(..u64::from(padded) * u64::from(band));
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let mapped = device.poll(wgpu::PollType::wait_indefinitely()).is_ok();
        let data = slice.get_mapped_range();
        let end = out.len() + (row * band) as usize;
        if let (true, Ok(data)) = (mapped, data.as_ref()) {
            for y in 0..band {
                let from = (y * padded) as usize;
                out.extend_from_slice(&data[from..from + row as usize]);
            }
        }
        drop(data);
        buffer.unmap();
        out.resize(end, 0);
    }
    out
}

/// An RGBA half-float texture's values, their bits.
pub fn halves(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u16> {
    let bytes = bytes(device, queue, texture);
    (bytes.as_chunks::<2>().0.iter())
        .map(|b| u16::from_ne_bytes(*b))
        .collect()
}

/// A texture's values as floats, `channels` a texel (of its own): half and
/// single floats widened, integers as their bits.
pub fn floats(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    channels: usize,
) -> Vec<f32> {
    let bytes = bytes(device, queue, texture);
    let values: Vec<f32> = match texture.format() {
        wgpu::TextureFormat::Rgba16Float => (bytes.as_chunks::<2>().0.iter())
            .map(|b| kinogaki_render_api::geometry::image::half_to_f32(u16::from_ne_bytes(*b)))
            .collect(),
        wgpu::TextureFormat::R32Uint => (bytes.as_chunks::<4>().0.iter())
            .map(|b| f32::from_bits(u32::from_ne_bytes(*b)))
            .collect(),
        _ => (bytes.as_chunks::<4>().0.iter())
            .map(|b| f32::from_ne_bytes(*b))
            .collect(),
    };
    let own = match texture.format() {
        wgpu::TextureFormat::Rgba16Float => 4,
        _ => 1,
    };
    match own == channels {
        true => values,
        false => (values.chunks(own))
            .flat_map(|t| (0..channels).map(move |k| t.get(k).copied().unwrap_or(0.0)))
            .collect(),
    }
}
