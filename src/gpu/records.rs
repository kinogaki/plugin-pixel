//! What the passes read, as the shaders lay it out: the G-buffer's vertices,
//! the instances, the lights, the volumes and the
//! per-frame uniform. Each packs from the renderer's scene.
//!
//! Each light kind lights a white Lambert surface as the direct pass reads it:
//! - **area** (rect, disk): a point emitter at its centre, power radiance ×
//!   area (UsdLux), falling off as 1 / (d² + area / π);
//! - **distant**: parallel light of its illuminance, no falloff: the
//!   radiance × π sin²(angle / 2), or the intensity itself when it
//!   normalizes or is a delta (angle 0);
//! - **spot**: a sphere seen from outside is a point of radiant intensity
//!   radiance × π r² (normalized: over its area 4π r²), falling off as 1 /
//!   max(d², r²), times the cone's falloff.
use crate::render::camera::CameraFrame;
use crate::render::settings::{MAX_LEVELS, Settings};
use crate::render::vector::V3;
use crate::render::{Ambient, Fog, Instance, Light, LightShape, Rims, Surface, Volume, dither};
use bytemuck::{Pod, Zeroable};

/// A G-buffer vertex: one per triangle corner, in its mesh's own space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Its position.
    pub position: [f32; 3],
    /// Its shading normal (not unit).
    pub normal: [f32; 3],
    /// What the material's color is multiplied by.
    pub tint: [f32; 3],
}

/// An instance as the passes read it (`Instance` in common.wgsl): its
/// placement both ways, its material and ids.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuInstance {
    /// Own space to world: three rows.
    pub to_world: [[f32; 4]; 3],
    /// World to own space: three rows (NaN for a flat placement, which no
    /// ray enters).
    pub to_own: [[f32; 4]; 3],
    /// rgb: the material's color.
    pub albedo: [f32; 4],
    /// rgb: what it lets through.
    pub transmit: [f32; 4],
    /// x: the object id; y: its mesh's BVH root node.
    pub id: [u32; 4],
}

impl GpuInstance {
    /// `instance` packed, its mesh's BVH rooted at node `root`, shading as
    /// `surface`.
    pub fn of(instance: &Instance, root: u32, surface: Surface) -> GpuInstance {
        let mut packed = GpuInstance {
            to_world: instance.to_world,
            to_own: instance.to_own.unwrap_or([[f32::NAN; 4]; 3]),
            id: [instance.id, root, 0, 0],
            ..GpuInstance::default()
        };
        packed.set_surface(surface);
        packed
    }

    /// Shades as `surface`; an unlit one's transmittance carries 1 in w.
    pub fn set_surface(&mut self, surface: Surface) {
        self.albedo = V3::of(surface.color).with_w(0.0);
        let unlit = f32::from(u8::from(surface.unlit));
        self.transmit = V3::of(surface.transmittance()).with_w(unlit);
    }
}

/// Kind codes (`normal.w` of a light).
const AREA: f32 = 0.0;
const DISTANT: f32 = 1.0;
const SPOT: f32 = 2.0;

/// A light as the passes read it (`Light` in light.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    /// xyz: the centre; w: the falloff's r².
    pub position: [f32; 4],
    /// xyz: where it shines (its -Z), unit; w: its kind.
    pub normal: [f32; 4],
    /// rgb: an area light's power, a spot's intensity, a distant light's
    /// illuminance.
    pub power: [f32; 4],
    /// x, y: a spot's cone cosines (outer, inner); z: the gobo's contrast;
    /// w: 1 with a gobo.
    pub cone: [f32; 4],
    /// The gobo's plane: xyz the light's X over the gobo's scale, w the
    /// offset's x over it.
    pub gobo_x: [f32; 4],
    /// The light's Y over the scale, w the offset's y over it.
    pub gobo_y: [f32; 4],
    /// x: the gobo's seed.
    pub seed: [u32; 4],
}

/// The axes and origin of three rows of an affine matrix.
fn frame_of(rows: &[[f32; 4]; 3]) -> [V3; 4] {
    std::array::from_fn(|i| V3::new(rows[0][i], rows[1][i], rows[2][i]))
}

impl GpuLight {
    /// `light` packed, its sizes in world space (its scale included).
    pub fn of(light: &Light) -> GpuLight {
        let [x, y, z, centre] = frame_of(&light.rows);
        let radiance = V3::of(light.radiance);
        let pi = std::f32::consts::PI;
        let mut cone = [1.0f32, 1.0, 0.0, 0.0];
        let (kind, r2, scale) = match light.shape {
            LightShape::Area { area } => {
                let area = area * x.length() * y.length();
                let scale = if light.normalize { 1.0 } else { area };
                (AREA, area / pi, scale)
            }
            LightShape::Distant { angle } => {
                let solid = pi * (angle.to_radians() * 0.5).sin().powi(2);
                let delta = angle <= 0.0;
                (
                    DISTANT,
                    0.0,
                    if light.normalize || delta { 1.0 } else { solid },
                )
            }
            LightShape::Spot {
                radius,
                cone_angle,
                cone_softness,
            } => {
                let r = radius * (x.length() + y.length() + z.length()) / 3.0;
                let angle = cone_angle.clamp(0.0, 180.0);
                cone[0] = angle.to_radians().cos();
                cone[1] = (angle * (1.0 - cone_softness.clamp(0.0, 1.0)))
                    .to_radians()
                    .cos();
                let normalized = match light.normalize && r > 0.0 {
                    true => 1.0 / (4.0 * pi * r * r),
                    false => 1.0,
                };
                (SPOT, r * r, normalized * pi * r * r)
            }
        };
        let mut packed = GpuLight {
            position: centre.with_w(r2),
            normal: (-z).unit_or(V3::new(0.0, 0.0, -1.0)).with_w(kind),
            power: (radiance * scale).with_w(0.0),
            cone,
            ..GpuLight::default()
        };
        // Area lights take no gobo.
        if let Some(g) = light.gobo.filter(|_| kind != AREA) {
            let s = g.scale.max(1e-6);
            let unit = |v: V3| v.unit_or(V3::ZERO) * (1.0 / s);
            packed.cone[2] = g.contrast.clamp(0.0, 1.0);
            packed.cone[3] = 1.0;
            packed.gobo_x = unit(x).with_w(g.offset[0] / s);
            packed.gobo_y = unit(y).with_w(g.offset[1] / s);
            packed.seed[0] = g.seed;
        }
        packed
    }
}

/// A volume as the passes read it (`Medium` in medium.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuMedium {
    /// World to the mesh's own space: three rows.
    pub to_own: [[f32; 4]; 3],
    /// xyz: the bounding box's minimum, own space; w: the edge falloff.
    pub lo: [f32; 4],
    /// xyz: its maximum; w: the density.
    pub hi: [f32; 4],
    /// xyz: scene units a unit of own space spans; w: the anisotropy.
    pub scale: [f32; 4],
    /// rgb: the scatter color; w: the noise amount.
    pub scatter: [f32; 4],
    /// rgb: the emission; w: the noise frequency.
    pub emission: [f32; 4],
    /// xyz: the noise offset.
    pub offset: [f32; 4],
    /// x: noise octaves; y: the noise seed; z: 1 when it casts shadows.
    pub flags: [u32; 4],
}

impl GpuMedium {
    /// `volume` packed.
    pub fn of(volume: &Volume) -> GpuMedium {
        let m = &volume.medium;
        let (lo, hi) = volume.own_bounds;
        let w = |v: [f32; 3], w: f32| V3::of(v).with_w(w);
        GpuMedium {
            to_own: volume.to_own,
            lo: w(lo, m.edge_falloff),
            hi: w(hi, m.density),
            scale: w(volume.scale, m.anisotropy),
            scatter: w(m.scatter_color, m.noise_amount),
            emission: w(m.emission, m.noise_frequency),
            offset: w(m.noise_offset, 0.0),
            flags: [m.noise_octaves, m.noise_seed, u32::from(m.shadows), 0],
        }
    }
}

/// What every pass reads (`Frame` in common.wgsl): the camera, the film and
/// the settings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Frame {
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
    bayer: [[f32; 4]; 16],
    shade_ramp: [[f32; 4]; MAX_LEVELS as usize],
    volume_ramp: [[f32; 4]; MAX_LEVELS as usize],
}

/// What a frame's uniform is made from besides the settings and camera.
pub struct Lighting {
    /// The domes.
    pub ambient: Ambient,
    /// The rim lights.
    pub rims: Rims,
    /// The fog.
    pub fog: Option<Fog>,
    /// Lights in the light buffer.
    pub lights: u32,
    /// Volumes, and how many of them cast shadows.
    pub volumes: (u32, u32),
    /// The scene's bounds' diagonal (ray offsets grow with it).
    pub extent: f32,
}

/// `colors` as a table of four floats each.
fn table(colors: &[[f32; 3]]) -> [[f32; 4]; MAX_LEVELS as usize] {
    std::array::from_fn(|k| colors.get(k).map_or([0.0; 4], |c| V3::of(*c).with_w(1.0)))
}

impl Frame {
    /// The uniform of one frame of `camera` at view size `view`.
    pub fn new(camera: &CameraFrame, view: [u32; 2], s: &Settings, l: &Lighting) -> Frame {
        let v = |a: [f64; 3], w: f64| [a[0] as f32, a[1] as f32, a[2] as f32, w as f32];
        let rgb = |c: [f32; 3], w: f32| V3::of(c).with_w(w);
        let mut bayer = [[0.0f32; 4]; 16];
        for (i, t) in dither::thresholds(s.dither).into_iter().enumerate() {
            bayer[i / 4][i % 4] = t;
        }
        let fog = l.fog.unwrap_or(Fog {
            color: [0.0; 3],
            density: 0.0,
            start: 0.0,
            height: 0.0,
            falloff: 0.0,
            lit: false,
            anisotropy: 0.0,
        });
        let levels = |ramp: &[[f32; 3]]| ramp.len().clamp(2, MAX_LEVELS as usize) as f32;
        let [right, up, back] = camera.axes;
        Frame {
            view_proj: camera.view_projection(),
            eye: v(camera.position, f64::from((l.extent * 1e-5).max(1e-6))),
            right: v(right, camera.origin[0]),
            up: v(up, camera.origin[1]),
            back: v(back, camera.step),
            ambient: rgb(l.ambient.sky, s.ao_distance),
            ground: rgb(l.ambient.ground, f32::from(u8::from(l.ambient.seen))),
            world_up: rgb(s.up, 0.0),
            film: [camera.size[0], camera.size[1], s.pixel_size, l.lights],
            view: [
                view[0],
                view[1],
                s.dither.size(),
                u32::from(camera.orthographic),
            ],
            shade: [
                levels(&s.shade_ramp),
                s.dither_strength,
                levels(&s.volume_ramp),
                s.volume_dither,
            ],
            rim: rgb(l.rims.lit, 0.0),
            rim_all: rgb(l.rims.all, 0.0),
            fog_color: rgb(fog.color, fog.density),
            fog: [fog.start, fog.height, fog.falloff, fog.anisotropy],
            ao: [s.ao_strength, s.ao_power, s.ao_direct, 0.0],
            gi: [s.gi, s.gi_distance, s.gi_samples as f32, 0.0],
            volume: [
                s.volume_steps as f32,
                s.volume_max_distance,
                f32::from(u8::from(s.volume_shadow_rays)),
                0.0,
            ],
            media: [
                l.volumes.0,
                l.volumes.1,
                u32::from(fog.lit && fog.density > 0.0),
                0,
            ],
            bayer,
            shade_ramp: table(&s.shade_ramp),
            volume_ramp: table(&s.volume_ramp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Gobo;

    const IDENTITY: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    fn light(shape: LightShape, radiance: f32) -> Light {
        Light {
            shape,
            radiance: [radiance; 3],
            normalize: false,
            rows: IDENTITY,
            gobo: None,
        }
    }

    #[test]
    fn area_lights_pack_their_power_and_direction() {
        let mut rows = IDENTITY;
        rows[0][0] = 2.0;
        let rect = Light {
            rows,
            gobo: Some(Gobo {
                scale: 1.0,
                contrast: 0.5,
                seed: 0,
                offset: [0.0; 2],
            }),
            ..light(LightShape::Area { area: 3.0 }, 10.0)
        };
        let g = GpuLight::of(&rect);
        assert_eq!(g.power[0], 10.0 * 6.0);
        assert!((g.position[3] - 6.0 / std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(g.normal, [0.0, 0.0, -1.0, AREA]);
        assert_eq!(g.cone[3], 0.0, "area lights take no gobo");
        let normalized = GpuLight::of(&Light {
            normalize: true,
            ..rect
        });
        assert_eq!(normalized.power[0], 10.0);
    }

    #[test]
    fn distant_lights_pack_their_illuminance() {
        let sun = |angle: f32, normalize| {
            let sun = light(LightShape::Distant { angle }, 50_000.0);
            GpuLight::of(&Light { normalize, ..sun })
        };
        // UsdLux's sun: radiance 50,000 over a 0.53° disk is about 3.4.
        assert!((sun(0.53, false).power[0] - 3.36).abs() < 0.01);
        assert_eq!(sun(0.53, true).power[0], 50_000.0);
        assert_eq!(sun(0.0, false).power[0], 50_000.0, "a delta's illuminance");
        assert_eq!(sun(0.53, false).normal[3], DISTANT);
    }

    #[test]
    fn spots_pack_their_intensity_cone_and_gobo() {
        let spot = Light {
            gobo: Some(Gobo {
                scale: 2.0,
                contrast: 0.75,
                seed: 9,
                offset: [1.0, -4.0],
            }),
            ..light(
                LightShape::Spot {
                    radius: 0.5,
                    cone_angle: 30.0,
                    cone_softness: 0.5,
                },
                4.0,
            )
        };
        let g = GpuLight::of(&spot);
        let pi = std::f32::consts::PI;
        assert!((g.power[0] - 4.0 * pi * 0.25).abs() < 1e-5);
        assert_eq!(g.position[3], 0.25);
        assert!((g.cone[0] - 30f32.to_radians().cos()).abs() < 1e-6);
        assert!((g.cone[1] - 15f32.to_radians().cos()).abs() < 1e-6);
        assert_eq!((g.cone[2], g.cone[3], g.seed[0]), (0.75, 1.0, 9));
        assert_eq!(g.gobo_x, [0.5, 0.0, 0.0, 0.5]);
        assert_eq!(g.gobo_y, [0.0, 0.5, 0.0, -2.0]);
    }

    #[test]
    fn the_frame_holds_each_level_of_both_ramps() {
        let settings = Settings::default();
        let camera = CameraFrame::new(
            &crate::render::camera::Camera::default(),
            &crate::render::film::Film::new(64, 32, 4),
            None,
        );
        let lighting = Lighting {
            ambient: Ambient::default(),
            rims: Rims::default(),
            fog: None,
            lights: 0,
            volumes: (0, 0),
            extent: 0.0,
        };
        let frame = Frame::new(&camera, [64, 32], &settings, &lighting);
        assert_eq!(frame.shade, [4.0, 1.0, 12.0, 0.25]);
        assert_eq!(frame.shade_ramp[3], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(frame.shade_ramp[4], [0.0; 4]);
        assert_eq!(frame.film, [16, 8, 4, 0]);
        assert_eq!(std::mem::size_of::<Frame>() % 16, 0);
    }
}
