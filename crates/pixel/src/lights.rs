//! Lights packed for the direct pass (`Light` in shaders/direct.wgsl).
//!
//! Each kind lights a white Lambert surface as the direct pass reads it:
//! - **area** (rect, disk): a point emitter at its centre, power
//!   radiance × area (UsdLux), falling off as 1 / (d² + area / π);
//! - **distant**: parallel light of its illuminance, no falloff: the
//!   radiance × π sin²(angle / 2), or the intensity itself when it
//!   normalizes or is a delta (angle 0);
//! - **spot**: a sphere seen from outside is a point of radiant intensity
//!   radiance × π r² (normalized: over its area 4π r²), falling off as
//!   1 / max(d², r²), times the cone's falloff.

use crate::math::Vec3;
use bytemuck::{Pod, Zeroable};

use crate::geometry::matrix;
use crate::scene::{Light, LightShape};

/// Kind codes (`normal.w`).
pub const AREA: f32 = 0.0;
/// A distant light.
pub const DISTANT: f32 = 1.0;
/// A spot.
pub const SPOT: f32 = 2.0;

/// A light as the direct pass reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    /// xyz: the centre; w: the falloff's r².
    pub position: [f32; 4],
    /// xyz: where it shines (the light's -Z), unit; w: its kind.
    pub normal: [f32; 4],
    /// rgb: an area light's power, a spot's radiant intensity, a distant
    /// light's illuminance.
    pub power: [f32; 4],
    /// x, y: a spot's cone cosines (outer, inner); z: the gobo's contrast;
    /// w: 1 with a gobo.
    pub cone: [f32; 4],
    /// The gobo's plane: xyz the light's X over the gobo's scale; w the
    /// offset's x over it.
    pub gobo_x: [f32; 4],
    /// xyz the light's Y over the gobo's scale; w the offset's y over it.
    pub gobo_y: [f32; 4],
    /// x: the gobo's seed.
    pub seed: [u32; 4],
}

fn f32x4(v: Vec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

/// `light` packed, its sizes in world space (its transform's scale
/// included).
pub fn pack_light(light: &Light) -> GpuLight {
    let m = matrix(&light.transform);
    let axis = |v: Vec3| m.transform_vector3(v);
    let (x, y, z) = (axis(Vec3::X), axis(Vec3::Y), axis(Vec3::Z));
    let centre = m.transform_point3(Vec3::ZERO);
    let normal = (-z).normalize_or(-Vec3::Z);
    let radiance = f64::from(light.intensity) * f64::from(light.exposure).exp2();
    let pi = std::f64::consts::PI;
    let mut cone = [1.0f32, 1.0, 0.0, 0.0];
    let (kind, r2, scale) = match light.shape {
        LightShape::Rect { width, height } => {
            let area = f64::from(width) * f64::from(height) * x.length() * y.length();
            (AREA, area / pi, if light.normalize { 1.0 } else { area })
        }
        LightShape::Disk { radius } => {
            let r = f64::from(radius);
            let area = pi * r * r * x.length() * y.length();
            (AREA, area / pi, if light.normalize { 1.0 } else { area })
        }
        LightShape::Distant { angle } => {
            let half = f64::from(angle).to_radians() * 0.5;
            let solid = pi * half.sin().powi(2);
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
            let r = f64::from(radius) * (x.length() + y.length() + z.length()) / 3.0;
            let angle = cone_angle.clamp(0.0, 180.0);
            cone[0] = angle.to_radians().cos();
            cone[1] = (angle * (1.0 - cone_softness.clamp(0.0, 1.0)))
                .to_radians()
                .cos();
            let normalized = if light.normalize && r > 0.0 {
                1.0 / (4.0 * pi * r * r)
            } else {
                1.0
            };
            (SPOT, r * r, normalized * pi * r * r)
        }
    };
    let k = (radiance * scale) as f32;
    let mut packed = GpuLight {
        position: f32x4(centre, r2),
        normal: f32x4(normal, f64::from(kind)),
        power: [
            light.color[0] * k,
            light.color[1] * k,
            light.color[2] * k,
            0.0,
        ],
        cone,
        ..GpuLight::default()
    };
    // Area lights take no gobo.
    if let Some(g) = light.gobo.filter(|_| kind != AREA) {
        let s = f64::from(g.scale.max(1e-6));
        let unit = |v: Vec3| v.normalize_or(Vec3::ZERO) / s;
        packed.cone[2] = g.contrast.clamp(0.0, 1.0);
        packed.cone[3] = 1.0;
        packed.gobo_x = f32x4(unit(x), f64::from(g.offset[0]) / s);
        packed.gobo_y = f32x4(unit(y), f64::from(g.offset[1]) / s);
        packed.seed[0] = g.seed;
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Gobo, IDENTITY};

    #[test]
    fn area_lights_pack_their_power_and_direction() {
        let mut scaled = IDENTITY;
        scaled[0][0] = 2.0;
        let light = Light {
            shape: LightShape::Rect {
                width: 1.0,
                height: 3.0,
            },
            intensity: 5.0,
            exposure: 1.0,
            transform: scaled,
            gobo: Some(Gobo::default()),
            ..Light::default()
        };
        let g = pack_light(&light);
        assert_eq!(g.power[0], 5.0 * 2.0 * 6.0);
        assert!((g.position[3] - 6.0 / std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(&g.normal, &[0.0, 0.0, -1.0, AREA]);
        assert_eq!(g.cone[3], 0.0, "area lights take no gobo");
        let normalized = pack_light(&Light {
            normalize: true,
            ..light
        });
        assert_eq!(normalized.power[0], 10.0);
    }

    #[test]
    fn distant_lights_pack_their_illuminance() {
        let sun = |angle: f32, normalize| {
            pack_light(&Light {
                shape: LightShape::Distant { angle },
                intensity: 50_000.0,
                normalize,
                ..Light::default()
            })
        };
        // UsdLux's sun: radiance 50,000 over a 0.53° disk is about 3.4.
        let e = sun(0.53, false).power[0];
        assert!((e - 3.36).abs() < 0.01, "{e}");
        assert_eq!(sun(0.53, true).power[0], 50_000.0);
        assert_eq!(sun(0.0, false).power[0], 50_000.0, "a delta's illuminance");
        assert_eq!(sun(0.53, false).normal[3], DISTANT);
    }

    #[test]
    fn spots_pack_their_intensity_cone_and_gobo() {
        let spot = Light {
            shape: LightShape::Spot {
                radius: 0.5,
                cone_angle: 30.0,
                cone_softness: 0.5,
            },
            intensity: 4.0,
            gobo: Some(Gobo {
                scale: 2.0,
                contrast: 0.75,
                seed: 9,
                offset: [1.0, -4.0],
            }),
            ..Light::default()
        };
        let g = pack_light(&spot);
        let pi = std::f32::consts::PI;
        assert!((g.power[0] - 4.0 * pi * 0.25).abs() < 1e-5);
        assert_eq!(g.position[3], 0.25);
        assert!((g.cone[0] - 30f32.to_radians().cos()).abs() < 1e-6);
        assert!((g.cone[1] - 15f32.to_radians().cos()).abs() < 1e-6);
        assert_eq!(&g.cone[2..], &[0.75, 1.0]);
        assert_eq!(g.gobo_x, [0.5, 0.0, 0.0, 0.5]);
        assert_eq!(g.gobo_y, [0.0, 0.5, 0.0, -2.0]);
        assert_eq!(g.seed[0], 9);
        // Normalized: its power no longer grows with its size.
        let normalized = pack_light(&Light {
            normalize: true,
            ..spot
        });
        assert!((normalized.power[0] - 1.0).abs() < 1e-6);
    }
}
