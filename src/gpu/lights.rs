//! Lights as the passes read them (`Light` in light.wgsl). Each lights a
//! white Lambert surface as the direct pass reads it, a point of its shape's
//! power at its centre:
//! - **area** (rect, disk): power radiance × area (UsdLux), falling off as
//!   1 / (d² + area / π), times its cosine toward the point;
//! - **distant**: parallel light of its illuminance, no falloff: the
//!   radiance × π sin²(angle / 2), or the intensity itself when it
//!   normalizes or is a delta (angle 0);
//! - **sphere**: a point of radiant intensity radiance × π r² (normalized:
//!   over its area 4π r², a quarter of the radiance), falling off as 1 /
//!   max(d², r²) (a point's as 1 / d²);
//! - **cylinder**: a point of radiant intensity across its axis radiance ×
//!   2r × length (normalized: over its side's area, the radiance / π), times
//!   the sine of the angle to its axis, falling off as 1 / max(d², r²).
//!
//! Each but a distant light is shaped (its cone's smoothstep over angles,
//! focus toward its tint, an IES profile read at [`PROFILE`] vertical
//! angles) and shadowed as ShadowAPI says.
use crate::render::vector::V3;
use crate::render::{Light, LightShape, PROFILE};
use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

/// Kind codes (`normal.w` of a light).
pub const AREA: f32 = 0.0;
/// A distant light.
pub const DISTANT: f32 = 1.0;
/// A sphere (or point).
pub const SPHERE: f32 = 2.0;
/// A cylinder (or line).
pub const CYLINDER: f32 = 3.0;

/// A light as the passes read it (`Light` in light.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    /// xyz: the centre; w: the falloff's r².
    pub position: [f32; 4],
    /// xyz: where it shines (its -Z), unit; w: its kind.
    pub normal: [f32; 4],
    /// rgb: an area light's power, a sphere's or cylinder's intensity, a
    /// distant light's illuminance.
    pub power: [f32; 4],
    /// xyz: a cylinder's axis (its X), unit.
    pub axis: [f32; 4],
    /// x: the cone's angle (radians, π none); y: where its softness starts;
    /// z: focus (0 none); w: the profile's angle scale.
    pub shaping: [f32; 4],
    /// rgb: the focus tint; w: 1 with an IES profile.
    pub tint: [f32; 4],
    /// The profile's candela at its vertical angles, four a row.
    pub profile: [[f32; 4]; PROFILE / 4],
    /// rgb: the shadow color; w: the shadow distance (-1 any, 0 none).
    pub shadow: [f32; 4],
    /// x: the shadow falloff (-1 none); y: its gamma; z: the gobo's
    /// contrast; w: 1 with a gobo.
    pub fade: [f32; 4],
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
        let widest = x.length().max(y.length()).max(z.length());
        let (kind, r2, scale) = match light.shape {
            LightShape::Area { .. } => {
                let area = light.area().unwrap_or(0.0);
                let scale = if light.normalize { 1.0 } else { area };
                (AREA, area / PI, scale)
            }
            LightShape::Distant { angle } => {
                let solid = PI * (angle.to_radians() * 0.5).sin().powi(2);
                let delta = angle <= 0.0;
                (
                    DISTANT,
                    0.0,
                    if light.normalize || delta { 1.0 } else { solid },
                )
            }
            LightShape::Sphere { radius, point } => {
                let r = radius * widest;
                let scale = if light.normalize { 0.25 } else { PI * r * r };
                (SPHERE, if point { 0.0 } else { r * r }, scale)
            }
            LightShape::Cylinder {
                length,
                radius,
                line,
            } => {
                let r = radius * y.length().max(z.length());
                let long = length * x.length();
                let scale = if light.normalize {
                    1.0 / PI
                } else {
                    2.0 * r * long
                };
                (CYLINDER, if line { 0.0 } else { r * r }, scale)
            }
        };
        let mut packed = GpuLight {
            position: centre.with_w(r2),
            normal: (-z).unit_or(V3::new(0.0, 0.0, -1.0)).with_w(kind),
            power: (radiance * scale).with_w(0.0),
            axis: x.unit_or(V3::new(1.0, 0.0, 0.0)).with_w(0.0),
            shaping: [PI, PI, 0.0, 0.0],
            ..GpuLight::default()
        };
        if let Some(s) = light.shaping.filter(|_| kind != DISTANT) {
            packed.shaping = [s.cone, s.start, s.focus, s.angle_scale];
            packed.tint = V3::of(s.tint).with_w(0.0);
            if let Some(profile) = s.profile {
                packed.tint[3] = 1.0;
                for (k, c) in profile.iter().enumerate() {
                    packed.profile[k / 4][k % 4] = *c;
                }
            }
        }
        let shadow = &light.shadow;
        packed.shadow = V3::of(shadow.color).with_w(shadow.reach);
        packed.fade = [shadow.falloff, shadow.gamma, 0.0, 0.0];
        // Distant and sphere lights take a gobo.
        if let Some(g) = light.gobo.filter(|_| kind == DISTANT || kind == SPHERE) {
            let s = g.scale.max(1e-6);
            let unit = |v: V3| v.unit_or(V3::ZERO) * (1.0 / s);
            packed.fade[2] = g.contrast.clamp(0.0, 1.0);
            packed.fade[3] = 1.0;
            packed.gobo_x = unit(x).with_w(g.offset[0] / s);
            packed.gobo_y = unit(y).with_w(g.offset[1] / s);
            packed.seed[0] = g.seed;
        }
        packed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{Gobo, Shadow, Shaping};

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
            shaping: None,
            shadow: Shadow::default(),
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
        assert!((g.position[3] - 6.0 / PI).abs() < 1e-6);
        assert_eq!(g.normal, [0.0, 0.0, -1.0, AREA]);
        assert_eq!(g.fade[3], 0.0, "area lights take no gobo");
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
    fn spheres_pack_their_intensity_shaping_and_gobo() {
        let shaped = Light {
            gobo: Some(Gobo {
                scale: 2.0,
                contrast: 0.75,
                seed: 9,
                offset: [1.0, -4.0],
            }),
            shaping: Some(Shaping {
                cone: 30f32.to_radians(),
                start: 15f32.to_radians(),
                focus: 2.0,
                tint: [1.0, 0.0, 0.0],
                profile: Some([3.0; PROFILE]),
                angle_scale: 0.5,
            }),
            ..light(
                LightShape::Sphere {
                    radius: 0.5,
                    point: false,
                },
                4.0,
            )
        };
        let g = GpuLight::of(&shaped);
        assert!((g.power[0] - 4.0 * PI * 0.25).abs() < 1e-5);
        assert_eq!(g.position[3], 0.25);
        assert_eq!(g.shaping[2..], [2.0, 0.5]);
        assert_eq!((g.tint, g.profile[3][3]), ([1.0, 0.0, 0.0, 1.0], 3.0));
        assert_eq!((g.fade[2], g.fade[3], g.seed[0]), (0.75, 1.0, 9));
        assert_eq!(g.gobo_x, [0.5, 0.0, 0.0, 0.5]);
        assert_eq!(g.gobo_y, [0.0, 0.5, 0.0, -2.0]);
        // A point has the sphere's power and no size to fall off within.
        let point = light(
            LightShape::Sphere {
                radius: 0.5,
                point: true,
            },
            4.0,
        );
        let p = GpuLight::of(&point);
        assert_eq!((p.power[0], p.position[3]), (g.power[0], 0.0));
    }

    #[test]
    fn cylinders_pack_their_intensity_across_their_axis_and_shadows() {
        let tube = Light {
            shadow: Shadow {
                color: [0.0, 0.0, 1.0],
                reach: 2.0,
                falloff: 0.5,
                gamma: 2.0,
            },
            ..light(
                LightShape::Cylinder {
                    length: 2.0,
                    radius: 0.25,
                    line: false,
                },
                3.0,
            )
        };
        let g = GpuLight::of(&tube);
        assert!((g.power[0] - 3.0 * 2.0 * 0.25 * 2.0).abs() < 1e-5);
        assert_eq!((g.axis, g.normal[3]), ([1.0, 0.0, 0.0, 0.0], CYLINDER));
        assert_eq!(
            (g.shadow, g.fade[..2].to_vec()),
            ([0.0, 0.0, 1.0, 2.0], vec![0.5, 2.0])
        );
        let normalized = GpuLight::of(&Light {
            normalize: true,
            ..tube
        });
        assert!((normalized.power[0] - 3.0 / PI).abs() < 1e-6);
        assert_eq!(std::mem::size_of::<GpuLight>() % 16, 0);
    }
}
