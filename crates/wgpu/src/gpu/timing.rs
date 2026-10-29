//! Optional GPU timestamps around each pass, when the device has
//! `TIMESTAMP_QUERY`.

/// The passes a frame times, in order.
pub const PASSES: [&str; 7] = [
    "gbuffer", "ambient", "direct", "indirect", "volume", "shade", "upscale",
];

/// One frame's GPU time per pass, in milliseconds: a diagnostic. Passes
/// that do not read each other's output run side by side on Apple GPUs
/// (the volume pass beside the lighting passes), and their timestamps then
/// overlap or, on the M4 Max, come out wrong; a frame's own time is its
/// wall time on the render thread, submission to completion.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTimes {
    /// `(pass name, ms)` in [`PASSES`] order.
    pub passes: Vec<(&'static str, f64)>,
}

impl PassTimes {
    /// The passes' sum.
    pub fn total(&self) -> f64 {
        self.passes.iter().map(|p| p.1).sum()
    }
}

/// The query set and its read-back buffers.
pub struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period_ns: f64,
}

const COUNT: u32 = PASSES.len() as u32 * 2;

impl Timer {
    /// A timer, when `device` can time passes.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Timer> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pixel timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: COUNT,
        });
        let size = u64::from(COUNT) * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel timestamps resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel timestamps read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Timer {
            queries,
            resolve,
            read,
            period_ns: f64::from(queue.get_timestamp_period()),
        })
    }

    /// The writes around compute pass `slot`.
    pub fn compute(&self, slot: usize) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(slot as u32 * 2),
            end_of_pass_write_index: Some(slot as u32 * 2 + 1),
        }
    }

    /// The writes around render pass `slot`.
    pub fn render(&self, slot: usize) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(slot as u32 * 2),
            end_of_pass_write_index: Some(slot as u32 * 2 + 1),
        }
    }

    /// Records the copy of this frame's stamps to the read-back buffer.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.queries, 0..COUNT, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, self.resolve.size());
    }

    /// The last submitted frame's times (waits for the GPU).
    pub fn read(&self, device: &wgpu::Device) -> Option<PassTimes> {
        let slice = self.read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let stamps: Vec<u64> = {
            let bytes = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice(&bytes).to_vec()
        };
        self.read.unmap();
        let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * self.period_ns / 1e6;
        Some(PassTimes {
            passes: PASSES
                .iter()
                .enumerate()
                .map(|(i, &name)| (name, ms(stamps[i * 2], stamps[i * 2 + 1])))
                .collect(),
        })
    }
}
