//! Pixel's lights and fog: the prim types it declares, its object kind of
//! the one light only it has, and what each prim lights a frame with.
//!
//! - **Standard lights** (core's objects, `kinogaki.core.light`, offered
//!   under Render while an engine renders them): UsdLux's rect, disk,
//!   cylinder and dome lights; its distant and sphere lights, the standard
//!   ones with a **Gobo** section (a mask of dappled light, as if through
//!   leaves out of frame), which core's lights of those types show while
//!   Pixel is installed.
//! - **Rim Light** (`kinogaki.pixel.rim_light`, Pixel's own kind
//!   `kinogaki.pixel.light`, in its family): a Fresnel-like edge,
//!   (1 − n·v)⁴ × its emission, added where direct light reaches a surface
//!   (Lit) or all round (All), with no place or direction.
//! - **Fog** (core's object, `kinogaki.core.fog`): the standard
//!   exponential height fog, flat, or lit by the lights and marched with the
//!   volumes when In-scatter is on.
//!
//! Emissions become colors here, at luminance 1 (the SDK's
//! [`Emission::color`]): Pixel renders in RGB. A standard light becomes
//! Pixel's ([`light_of`]): a point of its shape's power at its centre (or
//! parallel light), shaped, shadowed, its shape drawn when the camera sees
//! it.
use crate::render::{Glow, GlowMesh, Gobo, Light, LightShape, PROFILE, Shadow, Shaping};
use kinogaki_render_api::geometry::Xform;
use kinogaki_render_api::geometry::image::TextureImage;
use kinogaki_render_api::standard::{self, Emission};
use kinogaki_render_api::value::{Choice, Condition, Number, Schema, Value, ValueKind};
use kinogaki_render_api::{Declared, Parameter, PrimType, PrimTypeId, Role, Section};
use kinogaki_studio_api::ObjectKindId;
use kinogaki_studio_api::menu::Home;
use kinogaki_studio_api::render::lights::DeclaredKind;

/// The Rim Light kind's id.
pub const LIGHT_KIND: ObjectKindId = ObjectKindId::declared("kinogaki.pixel.light");
/// The rim light's type.
pub const RIM_LIGHT: PrimTypeId = PrimTypeId::declared("kinogaki.pixel.rim_light");

/// The Gobo section's group.
const GOBO: &str = "gobo";

/// The Gobo section, opening with its toggle: the toggle, then the
/// pattern's values.
fn gobo_section() -> Schema {
    let on = || Condition::when("gobo", true);
    let shown = |d: Parameter| d.requires(on());
    let section = Section::new(GOBO, "Gobo", false).open_when(on());
    [
        Parameter::toggle("gobo", "Gobo", false)
            .group(GOBO)
            .help("Masks the light with dappled shade, as if through leaves"),
        shown(
            Parameter::length("goboScale", "Scale", (1e-3, 1e6), Some((0.1, 10.0)), 2, 1.0)
                .group(GOBO),
        ),
        shown(
            Parameter::slider("goboContrast", "Contrast", (0.0, 1.0), (0.0, 1.0), 2, 0.5)
                .group(GOBO),
        ),
        shown(Parameter::integer("goboSeed", "Seed", 0, 1 << 20, 0).group(GOBO)),
        shown(
            Parameter::new(
                "goboOffset",
                "Offset",
                ValueKind::Vector {
                    components: 2,
                    number: Number::length(),
                },
                Value::Numbers(vec![0.0, 0.0]),
            )
            .group(GOBO),
        ),
    ]
    .into_iter()
    .fold(Schema::default().section(section), Schema::parameter)
}

/// The rim light's type: Side, then the standard emission settings, a warm
/// white of intensity 0.25 by default.
fn rim_light() -> PrimType {
    let sides = vec![Choice::new("lit", "Lit"), Choice::new("all", "All")];
    let side = Parameter::new(
        "side",
        "Side",
        ValueKind::choice(sides),
        standard::choice("lit"),
    )
    .group(standard::LIGHT_GROUP)
    .help("Lit: only where direct light reaches; All: round the whole silhouette");
    let emission = (standard::emission_params(0.25).into_iter()).map(|mut d| {
        if &*d.name == "color" {
            d.value = Value::Color([1.0, 0.85, 0.6]);
        }
        d
    });
    let light = Schema::default().section(Section::new(standard::LIGHT_GROUP, "Light", true));
    let schema = [side]
        .into_iter()
        .chain(emission)
        .fold(light, Schema::parameter);
    PrimType::new(RIM_LIGHT, "Rim Light", Role::Light, schema)
}

/// Pixel's light types, in the order menus list them: the standard ones (a
/// Gobo on distant and sphere lights), then the rim light.
pub fn light_types() -> Vec<PrimType> {
    (standard::lights().into_iter())
        .map(|ty| match ty.id {
            standard::DISTANT_LIGHT | standard::SPHERE_LIGHT => ty.merged(gobo_section()),
            _ => ty,
        })
        .chain([rim_light()])
        .collect()
}

/// Pixel's own light, the Rim Light, as a placed kind in its family.
pub fn lights() -> DeclaredKind {
    DeclaredKind::lights(LIGHT_KIND, Home::family(), vec![rim_light()])
}

/// Geometry light `light` (slot `slot`) as Pixel lights by it: a sphere of
/// its meshes' world area `area` at their area-weighted centre `centre`,
/// whose power is theirs (π × radiance × area, whichever side faces);
/// none for a mesh of no area. Its meshes show as their materials.
pub fn mesh_light(
    light: &standard::Light,
    (area, centre): (f32, [f32; 3]),
    slot: u32,
) -> Option<Light> {
    if area.is_nan() || area <= 0.0 {
        return None;
    }
    let radius = (area / (4.0 * std::f32::consts::PI)).sqrt();
    let [x, y, z] = centre;
    Some(Light {
        shape: LightShape::Sphere {
            radius,
            point: false,
        },
        radiance: light.radiance().map(|c| c * light.diffuse.max(0.0)),
        normalize: light.normalize,
        rows: [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z]],
        gobo: None,
        shaping: None,
        shadow: Shadow::default(),
        slot,
    })
}

/// The color a rim's emission makes, × intensity × 2^exposure (a standard
/// light's is [`standard::Light::radiance`]).
pub fn emitted(prim: &Declared) -> [f32; 3] {
    let emission = Emission::of_key(prim.choice("emission")).unwrap_or_default();
    let color = prim.color("color").map(f64::from);
    let kelvin = f64::from(prim.float("colorTemperature"));
    let rgb = emission.color(color, kelvin).unwrap_or([1.0; 3]);
    let k = f64::from(prim.float("intensity")) * f64::from(prim.float("exposure")).exp2();
    rgb.map(|c| (c * k) as f32)
}

/// `prim`'s gobo, when its Gobo is on.
pub fn gobo(prim: &Declared) -> Option<Gobo> {
    let offset = prim
        .value("goboOffset")
        .and_then(Value::numbers)
        .unwrap_or(&[]);
    let at = |k: usize| offset.get(k).copied().unwrap_or(0.0) as f32;
    prim.bool("gobo").then(|| Gobo {
        scale: prim.float("goboScale"),
        contrast: prim.float("goboContrast"),
        seed: (prim.value("goboSeed").and_then(Value::integer))
            .map_or(0, |s| s.clamp(0, i64::from(u32::MAX)) as u32),
        offset: [at(0), at(1)],
    })
}

/// `prim` as a light with a place or a direction, and the shape of it the
/// camera sees (its Visible to Camera on), its images (a rect's color map,
/// an IES profile) by `texture`: none for a dome or another type. Both
/// shine a color map's mean; the light's radiance takes its Diffuse too
/// (Pixel shades diffuse light alone; Specular has nothing to scale).
pub fn light_of(
    prim: &Declared,
    texture: &dyn Fn(&str) -> Option<TextureImage>,
    slot: u32,
) -> Option<(Light, Option<Glow>)> {
    let light = standard::Light::decode(prim)?;
    let mut radiance = light.radiance();
    let rows = rows(&light.xform.at(0.0));
    let scaled = |sx: f32, sy: f32, sz: f32| rows.map(|r| [r[0] * sx, r[1] * sy, r[2] * sz, r[3]]);
    let (shape, glow) = match light.shape {
        standard::LightShape::Rect { width, height, .. } => {
            if let Some(map) = texture("texture") {
                let mean = mean_color(&map);
                radiance = std::array::from_fn(|k| radiance[k] * mean[k]);
            }
            let area = LightShape::Area {
                area: width * height,
            };
            (area, Some((GlowMesh::Rect, scaled(width, height, 1.0))))
        }
        standard::LightShape::Disk { radius } => {
            let area = std::f32::consts::PI * radius * radius;
            let d = 2.0 * radius;
            (
                LightShape::Area { area },
                Some((GlowMesh::Disk, scaled(d, d, 1.0))),
            )
        }
        standard::LightShape::Sphere { radius, point } => {
            let d = 2.0 * radius;
            let glow = (!point).then_some((GlowMesh::Sphere, scaled(d, d, d)));
            (LightShape::Sphere { radius, point }, glow)
        }
        standard::LightShape::Cylinder {
            length,
            radius,
            line,
        } => {
            let d = 2.0 * radius;
            let glow = (!line).then_some((GlowMesh::Tube, scaled(length, d, d)));
            (
                LightShape::Cylinder {
                    length,
                    radius,
                    line,
                },
                glow,
            )
        }
        standard::LightShape::Distant { angle } => (LightShape::Distant { angle }, None),
        standard::LightShape::Dome { .. }
        | standard::LightShape::Portal { .. }
        | standard::LightShape::Geometry { .. } => return None,
    };
    let shaping = light.shaping.as_ref().map(|s| {
        let cone = s.cone_angle.clamp(0.0, 180.0).to_radians();
        Shaping {
            cone,
            start: cone * (1.0 - s.cone_softness.clamp(0.0, 1.0)),
            focus: s.focus.max(0.0),
            tint: s.focus_tint,
            profile: texture("iesFile").map(|t| profile_of(&t, s.ies_normalize)),
            angle_scale: s.ies_angle_scale,
        }
    });
    let shadow = &light.shadow;
    let shadow = Shadow {
        color: shadow.color,
        reach: if shadow.enable { shadow.distance } else { 0.0 },
        falloff: shadow.falloff,
        gamma: shadow.falloff_gamma,
    };
    let (camera, diffuse) = (light.camera, light.diffuse.max(0.0));
    let light = Light {
        shape,
        radiance: radiance.map(|c| c * diffuse),
        normalize: light.normalize,
        rows,
        gobo: gobo(prim),
        shaping,
        shadow,
        slot,
    };
    let over = match light.normalize {
        true => 1.0 / light.area().unwrap_or(1.0).max(f32::MIN_POSITIVE),
        false => 1.0,
    };
    let glow = glow.filter(|_| camera).map(|(mesh, to_world)| Glow {
        mesh,
        to_world,
        radiance: radiance.map(|c| c * over),
    });
    Some((light, glow))
}

/// Three rows of `x`, single precision.
fn rows(x: &Xform) -> [[f32; 4]; 3] {
    x.rows().map(|r| r.map(|v| v as f32))
}

/// Texels a side a color map's mean reads at most.
const MEAN_SAMPLES: u32 = 64;

/// `map`'s mean color over a grid of at most [`MEAN_SAMPLES`]² texels.
fn mean_color(map: &TextureImage) -> [f32; 3] {
    let data = map.image.data();
    let ([width, height], [x0, y0]) = (data.size(), data.min());
    let step = |n: u32| n.div_ceil(MEAN_SAMPLES).max(1);
    let (sx, sy) = (step(width), step(height));
    let (mut sum, mut count) = ([0.0f64; 3], 0.0f64);
    for y in (0..height).step_by(sy as usize) {
        for x in (0..width).step_by(sx as usize) {
            let t = map.texel(x0 + x as i32, y0 + y as i32);
            sum = std::array::from_fn(|k| sum[k] + f64::from(t[k]));
            count += 1.0;
        }
    }
    sum.map(|c| (c / count.max(1.0)) as f32)
}

/// An IES profile's image (render-api's `standard::ies` layout: candela,
/// then over its power) as [`Shaping::profile`]: its candela at each of
/// [`PROFILE`] vertical angles, the mean around its axis.
fn profile_of(profile: &TextureImage, normalized: bool) -> [f32; PROFILE] {
    let data = profile.image.data();
    let ([width, height], [x0, y0]) = (data.size(), data.min());
    let channel = if normalized { 3 } else { 0 };
    std::array::from_fn(|k| {
        let theta = (k as f32 + 0.5) / PROFILE as f32;
        let y = ((theta * height as f32) as u32).min(height.saturating_sub(1));
        let sum: f32 = (0..width)
            .map(|x| profile.texel(x0 + x as i32, y0 + y as i32)[channel])
            .sum();
        sum / width.max(1) as f32
    })
}

/// Whether a rim light shows all round, rather than where direct light
/// reaches.
pub fn rim_all_round(prim: &Declared) -> bool {
    prim.choice("side") == "all"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadowless::NO_SLOT;

    fn declared(ty: &PrimType, values: &[(&str, Value)]) -> Declared {
        let mut prim = Declared::new(ty.id, ty.schema().values());
        for (name, v) in values {
            prim.values.insert((*name).to_owned(), v.clone());
        }
        prim
    }

    #[test]
    fn distant_and_sphere_lights_take_a_gobo_and_the_rim_comes_last() {
        let types = light_types();
        let ids: Vec<PrimTypeId> = types.iter().map(|t| t.id).collect();
        assert_eq!(
            ids,
            [
                standard::RECT_LIGHT,
                standard::DISK_LIGHT,
                standard::SPHERE_LIGHT,
                standard::CYLINDER_LIGHT,
                standard::DISTANT_LIGHT,
                standard::DOME_LIGHT,
                standard::PORTAL_LIGHT,
                standard::GEOMETRY_LIGHT,
                RIM_LIGHT
            ]
        );
        let gobo_of = |t: &PrimType| t.schema().parameters().any(|d| &*d.name == "goboScale");
        assert_eq!(
            types.iter().map(gobo_of).collect::<Vec<_>>(),
            [false, false, true, false, true, false, false, false, false]
        );
        // Off by default, decoded when on.
        let spot = declared(&types[2], &[]);
        assert_eq!(gobo(&spot), None);
        let on = declared(
            &types[2],
            &[
                ("gobo", Value::Bool(true)),
                ("goboSeed", Value::Integer(9)),
                ("goboOffset", Value::Numbers(vec![1.0, -2.0])),
            ],
        );
        let g = gobo(&on).unwrap();
        assert_eq!(
            (g.scale, g.contrast, g.seed, g.offset),
            (1.0, 0.5, 9, [1.0, -2.0])
        );
    }

    #[test]
    fn a_light_shows_its_shape_when_seen_its_color_maps_mean_in_both() {
        let sphere = |camera: bool, normalize: bool| standard::Light {
            shape: standard::LightShape::Sphere {
                radius: 0.5,
                point: false,
            },
            intensity: 2.0,
            diffuse: 0.5,
            normalize,
            camera,
            ..standard::Light::default()
        };
        let none = |_: &str| None;
        let (light, glow) = light_of(&sphere(false, false).declared(), &none, NO_SLOT).unwrap();
        assert_eq!(
            (light.radiance, glow),
            ([1.0; 3], None),
            "Diffuse on the light alone"
        );
        let (_, glow) = light_of(&sphere(true, false).declared(), &none, NO_SLOT).unwrap();
        let glow = glow.unwrap();
        assert_eq!(
            (glow.mesh, glow.radiance, glow.to_world[0][0]),
            (GlowMesh::Sphere, [2.0; 3], 1.0)
        );
        // Normalized, it shines its power over its area, 4π r².
        let (_, glow) = light_of(&sphere(true, true).declared(), &none, NO_SLOT).unwrap();
        assert!((glow.unwrap().radiance[0] - 2.0 / std::f32::consts::PI).abs() < 1e-6);
        // A point shows nothing; a rect's color map tints both by its mean.
        let point = standard::Light {
            shape: standard::LightShape::Sphere {
                radius: 0.5,
                point: true,
            },
            ..sphere(true, false)
        };
        assert_eq!(light_of(&point.declared(), &none, NO_SLOT).unwrap().1, None);
        let map = {
            use kinogaki_render_api::geometry::image::{
                Decoding, Encoding, Format, Image, Rect, Sample,
            };
            let format = Format::rgba(Sample::F32, Encoding::color("ACEScg", false));
            let texels = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
            let image = Image::builder(Rect::new([0, 0], [2, 1]).unwrap()).plane(format, &texels);
            TextureImage {
                image: image.build().unwrap(),
                decoding: Some(Decoding::IDENTITY),
            }
        };
        let rect = standard::Light {
            camera: true,
            ..standard::Light::default()
        };
        let mapped = |row: &str| (row == "texture").then(|| map.clone());
        let (light, glow) = light_of(&rect.declared(), &mapped, NO_SLOT).unwrap();
        assert_eq!(
            (light.radiance, glow.unwrap().radiance),
            ([0.5, 0.0, 0.5], [0.5, 0.0, 0.5])
        );
    }

    #[test]
    fn emissions_are_colors_at_their_intensity() {
        let rim = rim_light();
        let warm = declared(&rim, &[]);
        assert_eq!(emitted(&warm), [0.25, 0.2125, 0.15]);
        assert!(!rim_all_round(&warm));
        let doubled = declared(
            &rim,
            &[
                ("exposure", Value::Number(1.0)),
                ("color", Value::Color([1.0; 3])),
            ],
        );
        assert_eq!(emitted(&doubled), [0.5; 3]);
        let hot = declared(
            &rim,
            &[
                ("emission", standard::choice("temperature")),
                ("colorTemperature", Value::Number(2000.0)),
            ],
        );
        let c = emitted(&hot);
        assert!(c[0] > c[1] && c[1] > c[2], "{c:?}");
    }
}
