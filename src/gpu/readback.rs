//! Textures read back to the CPU: a frame's pixels for a host that asks for
//! them, a final frame's outputs and a preview's picture. Each read waits
//! for the GPU, on the render thread.
use kinogaki_render_api::display::wgpu;

/// The bytes of `texture` (a whole 2D texture of a 4-, 8- or 16-byte texel
/// format), rows from the top, tightly packed.
pub fn bytes(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let texel = texture.format().block_copy_size(None).unwrap_or(4);
    let row = width * texel;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel readback"),
        size: u64::from(padded * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    // A lost device maps nothing: the read is all zeros.
    let mapped = device.poll(wgpu::PollType::wait_indefinitely()).is_ok();
    let mut out = Vec::with_capacity((row * height) as usize);
    let data = slice.get_mapped_range();
    if let (true, Ok(data)) = (mapped, data.as_ref()) {
        for y in 0..height {
            let from = (y * padded) as usize;
            out.extend_from_slice(&data[from..from + row as usize]);
        }
    } else {
        out.resize((row * height) as usize, 0);
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
