//! The per-frame uniform (`Frame` in shaders/common.wgsl).

use bytemuck::{Pod, Zeroable};

use kinogaki_pixel_render::camera::CameraFrame;
use kinogaki_pixel_render::{Dome, Fog, Rims, Settings, dither};

/// What every pass reads: the camera, the film and the settings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    back: [f32; 4],
    ambient: [f32; 4],
    ground: [f32; 4],
    world_up: [f32; 4],
    film: [u32; 4],
    view: [u32; 4],
    shade: [f32; 4],
    rim: [f32; 4],
    rim_all: [f32; 4],
    fog_color: [f32; 4],
    fog: [f32; 4],
    ao: [f32; 4],
    gi: [f32; 4],
    volume: [f32; 4],
    media: [u32; 4],
    bands: [f32; 4],
    bayer: [[f32; 4]; 16],
    ramp: [[f32; 4]; 16],
}

/// What a frame's uniform is made from.
pub struct FrameInputs<'a> {
    /// The camera over the film.
    pub camera: &'a CameraFrame,
    /// The view's size, in view pixels.
    pub view: [u32; 2],
    /// The settings, clamped.
    pub settings: &'a Settings,
    /// The domes, if any.
    pub dome: Option<Dome>,
    /// The rim lights.
    pub rims: Rims,
    /// The fog, if any.
    pub fog: Option<Fog>,
    /// Area lights in the light buffer.
    pub lights: u32,
    /// The scene's extent (its bounds' diagonal), for ray offsets.
    pub extent: f32,
    /// Volumes, and how many of them cast shadows.
    pub volumes: (u32, u32),
}

impl FrameUniform {
    /// The uniform for one frame.
    pub fn new(f: &FrameInputs) -> FrameUniform {
        let c = f.camera;
        let s = f.settings;
        let v = |v: kinogaki_pixel_render::math::Vec3, w: f64| {
            [v.x as f32, v.y as f32, v.z as f32, w as f32]
        };
        let vp = c.view_projection();
        let mut view_proj = [[0.0f32; 4]; 4];
        for (i, col) in view_proj.iter_mut().enumerate() {
            let k = vp.col(i);
            *col = [k.x as f32, k.y as f32, k.z as f32, k.w as f32];
        }
        // Ambient light from above and below the horizon: the domes'.
        let dome = f.dome.unwrap_or_default();
        let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
        // No fog: zero density.
        let fog = f.fog.unwrap_or(Fog {
            density: 0.0,
            ..Fog::default()
        });
        let mut bayer = [[0.0f32; 4]; 16];
        for (i, t) in dither::thresholds(s.dither).into_iter().enumerate() {
            bayer[i / 4][i % 4] = t;
        }
        FrameUniform {
            view_proj,
            eye: v(c.position, (f.extent * 1e-5).max(1e-6) as f64),
            right: v(c.axes[0], c.origin[0]),
            up: v(c.axes[1], c.origin[1]),
            back: v(c.axes[2], c.step),
            ambient: rgb(dome.above, s.ao_distance),
            ground: rgb(dome.below, f32::from(u8::from(dome.visible))),
            world_up: [s.up[0], s.up[1], s.up[2], 0.0],
            film: [c.size[0], c.size[1], s.pixel_size, f.lights],
            view: [
                f.view[0],
                f.view[1],
                s.dither.size(),
                u32::from(c.orthographic()),
            ],
            shade: [
                s.shade_levels as f32,
                s.dither_strength,
                s.shade_ramp.count as f32,
                0.0,
            ],
            rim: rgb(f.rims.lit, 0.0),
            rim_all: rgb(f.rims.all, 0.0),
            fog_color: rgb(fog.color, fog.density),
            fog: [fog.start, fog.height, fog.falloff, 0.0],
            ao: [s.ao_strength, s.ao_power, s.ao_direct, 0.0],
            gi: [s.gi, s.gi_distance, s.gi_samples as f32, 0.0],
            volume: [
                s.volume_steps as f32,
                s.volume_max_distance,
                f32::from(u8::from(s.volume_shadow_rays)),
                fog.anisotropy,
            ],
            media: [
                f.volumes.0,
                f.volumes.1,
                u32::from(fog.inscatter && fog.density > 0.0),
                0,
            ],
            bands: [s.volume_levels as f32, s.volume_dither, 0.0, 0.0],
            bayer,
            ramp: s.shade_ramp.stops,
        }
    }
}
