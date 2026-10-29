//! A compute kernel: its pipeline and its group-0 layout. Each view binds
//! its own frame resources in a group of that layout.

use super::layout;
use super::timing::Timer;

/// One compute pass over the film or the view.
pub struct Kernel {
    label: &'static str,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Kernel {
    /// A kernel of `source` (after the common code and `parts`) whose
    /// group 0 holds `entries`, with `more` layouts after it.
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        parts: &[&str],
        entries: &[wgpu::BindGroupLayoutEntry],
        more: &[&wgpu::BindGroupLayout],
    ) -> Kernel {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
        let module = layout::shader(device, label, parts);
        let mut layouts = vec![&layout];
        layouts.extend_from_slice(more);
        let pipeline = layout::compute_pipeline(device, label, &module, &layouts);
        Kernel {
            label,
            pipeline,
            layout,
        }
    }

    /// A group 0 of `resources`, in binding order.
    pub fn bind(
        &self,
        device: &wgpu::Device,
        resources: &[wgpu::BindingResource],
    ) -> wgpu::BindGroup {
        layout::bind_group(device, self.label, &self.layout, resources)
    }

    /// Records the kernel over `size` pixels with `groups` (group 0
    /// first), timed as pass `slot` when a timer runs.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        groups: &[&wgpu::BindGroup],
        timer: Option<&Timer>,
        slot: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: timer.map(|t| t.compute(slot)),
        });
        pass.set_pipeline(&self.pipeline);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *g, &[]);
        }
        let (x, y) = layout::groups(size);
        pass.dispatch_workgroups(x, y, 1);
    }
}
