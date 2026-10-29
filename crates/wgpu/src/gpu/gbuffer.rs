//! The G-buffer pass: the meshes rasterized at film resolution into albedo,
//! normal, depth, id and transmittance (shaders/gbuffer.wgsl).

use kinogaki_pixel_render::geometry::Vertex;

use super::layout;
use super::scene::GpuScene;
use super::targets::{self, Targets};
use super::timing::Timer;

/// The raster pipeline and its group's layout.
pub struct GbufferPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32, 2 => Float32x3, 3 => Float32x3];

impl GbufferPass {
    /// The pipeline.
    pub fn new(device: &wgpu::Device) -> GbufferPass {
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel gbuffer"),
            entries: &[layout::uniform(0, stages), layout::storage(1, stages)],
        });
        let module = layout::shader(
            device,
            "pixel gbuffer",
            &[include_str!("../shaders/gbuffer.wgsl")],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pixel gbuffer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pixel gbuffer"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRIBUTES,
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: targets::DEPTH_TEST,
                depth_write_enabled: Some(true),
                // Reversed depth: nearer is larger.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[
                    target(targets::COLOR),
                    target(targets::COLOR),
                    target(targets::SCALAR),
                    target(targets::ID),
                    target(targets::COLOR),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        GbufferPass { pipeline, layout }
    }

    /// A view's group: its frame uniform and the object table.
    pub fn bind(
        &self,
        device: &wgpu::Device,
        frame: &wgpu::Buffer,
        scene: &GpuScene,
    ) -> wgpu::BindGroup {
        layout::bind_group(
            device,
            "pixel gbuffer",
            &self.layout,
            &[frame.as_entire_binding(), scene.objects.as_entire_binding()],
        )
    }

    /// Records the pass: clears every target (ids to `MISS`, depth to the
    /// far plane: 0, depth is reversed) and draws.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &Targets,
        scene: &GpuScene,
        group: &wgpu::BindGroup,
        timer: Option<&Timer>,
    ) {
        let clear = |view, color| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let miss = wgpu::Color {
            r: f64::from(u32::MAX),
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pixel gbuffer"),
            color_attachments: &[
                clear(&targets.albedo.view, wgpu::Color::TRANSPARENT),
                clear(&targets.normal.view, wgpu::Color::TRANSPARENT),
                clear(&targets.depth.view, wgpu::Color::TRANSPARENT),
                clear(&targets.id.view, miss),
                clear(&targets.transmit.view, wgpu::Color::TRANSPARENT),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_test.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timer.map(|t| t.render(0)),
            ..Default::default()
        });
        if scene.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_vertex_buffer(0, scene.vertices.slice(..));
        pass.draw(0..scene.vertex_count, 0..1);
    }
}
