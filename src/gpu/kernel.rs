//! The compute kernels' plumbing: bind group layout entries, shader assembly
//! (the common code, then the parts a pass reads, then its own source) and
//! a [`Kernel`]: a pipeline and its group 0, each view binding its own.
use super::scene::MAX_VOLUMES;
use crate::render::bvh::MAX_DEPTH;
use kinogaki_render_api::display::wgpu;

/// A uniform buffer.
pub fn uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A read-only storage buffer.
pub fn storage(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A texture read with `textureLoad` (never filtered): floats, or ids.
pub fn texture(binding: u32, ids: bool) -> wgpu::BindGroupLayoutEntry {
    let sample_type = match ids {
        true => wgpu::TextureSampleType::Uint,
        false => wgpu::TextureSampleType::Float { filterable: false },
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A write-only storage texture.
pub fn written(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// A bind group of `resources` in binding order.
pub fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = (resources.iter().enumerate())
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: r.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// The WGSL every pass starts with.
const COMMON: &str = include_str!("shaders/common.wgsl");

/// The shared parts of the passes' sources.
pub mod part {
    /// A surface at a point: its texture, tint and cutout.
    pub const SURFACE: &str = include_str!("shaders/surface.wgsl");
    /// The BVH's traversals (its stack is the kernel's to size), after
    /// [`SURFACE`].
    pub const BVH: &str = include_str!("shaders/bvh.wgsl");
    /// The volumes' crossings, density and transmittance.
    pub const MEDIUM: &str = include_str!("shaders/medium.wgsl");
    /// The gobo's mask.
    pub const GOBO: &str = include_str!("shaders/gobo.wgsl");
    /// The lights' arrival and irradiance.
    pub const LIGHT: &str = include_str!("shaders/light.wgsl");
    /// The fixed hemisphere pattern AO and GI share.
    pub const HEMISPHERE: &str = include_str!("shaders/hemisphere.wgsl");
}

/// A shader module of the common code, `parts`, then the pass's own source:
/// with the constants the parts read (the BVH's stack, AO's rays, the most
/// volumes).
pub fn shader(device: &wgpu::Device, label: &str, parts: &[&str]) -> wgpu::ShaderModule {
    let mut source = format!(
        "{COMMON}\nconst STACK: u32 = {}u;\nconst AO_RAYS: u32 = {AO_RAYS}u;\nconst MAX_VOLUMES: u32 = {MAX_VOLUMES}u;\n",
        MAX_DEPTH + 1
    );
    for part in parts {
        source.push('\n');
        source.push_str(part);
    }
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Ambient occlusion's rays a film pixel.
pub const AO_RAYS: u32 = 32;

/// One compute pass over the film or the view.
pub struct Kernel {
    label: &'static str,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Kernel {
    /// A kernel of `parts` (its own source last) whose group 0 holds
    /// `entries`, the scene's group after it when `scene` is given.
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        parts: &[&str],
        entries: &[wgpu::BindGroupLayoutEntry],
        scene: Option<&wgpu::BindGroupLayout>,
    ) -> Kernel {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
        let module = shader(device, label, parts);
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
            [Some(&layout)].into_iter().chain(scene.map(Some)).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
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
        bind_group(device, self.label, &self.layout, resources)
    }

    /// Records the kernel over `size` pixels with `groups` (group 0 first).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        groups: &[&wgpu::BindGroup],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *g, &[]);
        }
        // The shaders' workgroups are 8 × 8.
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
    }
}
