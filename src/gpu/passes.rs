//! The seven passes' pipelines, a view's bind groups for them, and the order
//! they are recorded in. Every compute pass binds its view's frame and
//! textures as group 0; those that trace rays or light bind the scene as
//! group 1.
use super::kernel::{Kernel, bind_group, part, texture, uniform, written};
use super::records::Vertex;
use super::scene::GpuScene;
use super::targets::{COLOR, DEPTH_TEST, ID, Image, SCALAR, Targets};
use crate::outputs::Output;
use kinogaki_render_api::display::wgpu;

/// The pipelines.
pub struct Passes {
    gbuffer: wgpu::RenderPipeline,
    gbuffer_layout: wgpu::BindGroupLayout,
    ambient: Kernel,
    direct: Kernel,
    bounce: Kernel,
    volume: Kernel,
    shade: Kernel,
    upscale: Kernel,
}

/// A view's bind groups for its targets, but the upscale's (one a display
/// image).
pub struct Groups {
    gbuffer: wgpu::BindGroup,
    ambient: wgpu::BindGroup,
    direct: wgpu::BindGroup,
    bounce: wgpu::BindGroup,
    volume: wgpu::BindGroup,
    shade: wgpu::BindGroup,
}

/// A vertex's position, normal and tint.
const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];
/// The instance a drawn instance is.
const INSTANCE: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![3 => Uint32];

/// The G-buffer's raster pipeline: five targets, reversed depth, both sides.
fn gbuffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let module = super::kernel::shader(
        device,
        "pixel gbuffer",
        &[include_str!("shaders/gbuffer.wgsl")],
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pixel gbuffer"),
        bind_group_layouts: &[Some(layout), Some(scene)],
        immediate_size: 0,
    });
    let target = |format| {
        Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pixel gbuffer"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRIBUTES,
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &INSTANCE,
                }),
            ],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_TEST,
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
                target(COLOR),
                target(COLOR),
                target(SCALAR),
                target(ID),
                target(COLOR),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl Passes {
    /// The pipelines, binding the scene's group of `scene`'s layout.
    pub fn new(device: &wgpu::Device, scene: &wgpu::BindGroupLayout) -> Passes {
        let frame = uniform(0, wgpu::ShaderStages::COMPUTE);
        // What the passes reading the G-buffer bind first.
        let gbuffer_in = [
            frame,
            texture(1, false),
            texture(2, false),
            texture(3, true),
        ];
        let with = |more: &[wgpu::BindGroupLayoutEntry]| [&gbuffer_in[..], more].concat();
        let lit = [part::BVH, part::MEDIUM, part::GOBO, part::LIGHT];
        let source = |own: &'static str, extra: &[&'static str]| [&lit[..], extra, &[own]].concat();
        let gbuffer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel gbuffer"),
            entries: &[uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        Passes {
            gbuffer: gbuffer(device, &gbuffer_layout, scene),
            gbuffer_layout,
            ambient: Kernel::new(
                device,
                "pixel ambient",
                &[
                    part::BVH,
                    part::HEMISPHERE,
                    include_str!("shaders/ambient.wgsl"),
                ],
                &with(&[written(4, COLOR), written(5, SCALAR)]),
                Some(scene),
            ),
            direct: Kernel::new(
                device,
                "pixel direct",
                &source(include_str!("shaders/direct.wgsl"), &[]),
                &with(&[texture(4, false), texture(5, false), written(6, COLOR)]),
                Some(scene),
            ),
            bounce: Kernel::new(
                device,
                "pixel bounce",
                &source(include_str!("shaders/bounce.wgsl"), &[part::HEMISPHERE]),
                &with(&[written(4, COLOR)]),
                Some(scene),
            ),
            volume: Kernel::new(
                device,
                "pixel volume",
                &source(include_str!("shaders/volume.wgsl"), &[part::HEMISPHERE]),
                &[
                    frame,
                    texture(1, false),
                    texture(2, true),
                    written(3, COLOR),
                ],
                Some(scene),
            ),
            shade: Kernel::new(
                device,
                "pixel shade",
                &[include_str!("shaders/shade.wgsl")],
                &[
                    frame,
                    texture(1, false),
                    texture(2, false),
                    texture(3, false),
                    texture(4, false),
                    texture(5, true),
                    texture(6, false),
                    texture(7, false),
                    texture(8, false),
                    written(9, COLOR),
                    texture(10, false),
                ],
                None,
            ),
            upscale: Kernel::new(
                device,
                "pixel upscale",
                &[include_str!("shaders/upscale.wgsl")],
                &[frame, texture(1, false), written(2, COLOR)],
                None,
            ),
        }
    }

    /// A view's groups over `frame` and `t`.
    pub fn bind<'a>(
        &self,
        device: &wgpu::Device,
        frame: &'a wgpu::Buffer,
        t: &'a Targets,
    ) -> Groups {
        use wgpu::BindingResource::TextureView as Tex;
        let tex = |o: Output| Tex(t.view(o));
        let frame = frame.as_entire_binding();
        let gbuffer_in = || {
            vec![
                frame.clone(),
                tex(Output::Normal),
                tex(Output::Depth),
                tex(Output::ObjectId),
            ]
        };
        let with = |more: Vec<wgpu::BindingResource<'a>>| [gbuffer_in(), more].concat();
        Groups {
            gbuffer: bind_group(
                device,
                "pixel gbuffer",
                &self.gbuffer_layout,
                std::slice::from_ref(&frame),
            ),
            ambient: (self.ambient).bind(
                device,
                &with(vec![tex(Output::Ambient), tex(Output::Occlusion)]),
            ),
            direct: self.direct.bind(
                device,
                &with(vec![
                    Tex(&t.transmit.view),
                    tex(Output::Occlusion),
                    tex(Output::Direct),
                ]),
            ),
            bounce: self.bounce.bind(device, &with(vec![tex(Output::Indirect)])),
            volume: self.volume.bind(
                device,
                &[
                    frame.clone(),
                    tex(Output::Depth),
                    tex(Output::ObjectId),
                    tex(Output::Volume),
                ],
            ),
            shade: self.shade.bind(
                device,
                &[
                    frame,
                    tex(Output::Albedo),
                    tex(Output::Direct),
                    tex(Output::Ambient),
                    tex(Output::Indirect),
                    tex(Output::ObjectId),
                    tex(Output::Normal),
                    tex(Output::Depth),
                    tex(Output::Volume),
                    tex(Output::Beauty),
                    Tex(&t.transmit.view),
                ],
            ),
        }
    }

    /// The upscale's group into display image `image`.
    pub fn bind_upscale(
        &self,
        device: &wgpu::Device,
        frame: &wgpu::Buffer,
        t: &Targets,
        image: &Image,
    ) -> wgpu::BindGroup {
        use wgpu::BindingResource::TextureView as Tex;
        let beauty = Tex(t.view(Output::Beauty));
        (self.upscale).bind(
            device,
            &[frame.as_entire_binding(), beauty, Tex(&image.shown.view)],
        )
    }

    /// Records the passes in order.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        t: &Targets,
        scene: &GpuScene,
        groups: &Groups,
        upscale: &wgpu::BindGroup,
    ) {
        self.draw_gbuffer(encoder, t, scene, &groups.gbuffer);
        let size = t.film.size;
        for (kernel, group) in [
            (&self.ambient, &groups.ambient),
            (&self.direct, &groups.direct),
            (&self.bounce, &groups.bounce),
            (&self.volume, &groups.volume),
        ] {
            kernel.dispatch(encoder, size, &[group, &scene.group]);
        }
        self.shade.dispatch(encoder, size, &[&groups.shade]);
        self.upscale.dispatch(encoder, t.film.view, &[upscale]);
    }

    /// The G-buffer: every target cleared (ids to `MISS`, depth to the far
    /// plane, 0: depth is reversed), then each mesh drawn for its instances.
    fn draw_gbuffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        t: &Targets,
        scene: &GpuScene,
        group: &wgpu::BindGroup,
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
            ..wgpu::Color::TRANSPARENT
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pixel gbuffer"),
            color_attachments: &[
                clear(t.view(Output::Albedo), wgpu::Color::TRANSPARENT),
                clear(t.view(Output::Normal), wgpu::Color::TRANSPARENT),
                clear(t.view(Output::Depth), wgpu::Color::TRANSPARENT),
                clear(t.view(Output::ObjectId), miss),
                clear(&t.transmit.view, wgpu::Color::TRANSPARENT),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &t.depth_test.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        if scene.instances.draws.is_empty() {
            return;
        }
        pass.set_pipeline(&self.gbuffer);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, &scene.group, &[]);
        pass.set_vertex_buffer(0, scene.vertices.slice(..));
        pass.set_vertex_buffer(1, scene.drawn.slice(..));
        for (vertices, instances) in &scene.instances.draws {
            pass.draw(vertices.clone(), instances.clone());
        }
    }
}
