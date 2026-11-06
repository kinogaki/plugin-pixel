//! Pixel's lights and fog: the prim types it declares and its object kinds
//! of them, and what each prim lights a frame with.
//!
//! - **Lights** (`kinogaki.pixel.light`, placed): UsdLux's standard area
//!   and dome lights; its distant and spot lights, the standard ones with a
//!   **Gobo** section (a mask of dappled light, as if through leaves out of
//!   frame); and the **Rim Light** (`kinogaki.pixel.rim_light`), a
//!   Fresnel-like edge, (1 − n·v)⁴ × its emission, added where direct light
//!   reaches a surface (Lit) or all round (All), with no place or direction.
//! - **Fog** (`kinogaki.pixel.fog`, at the root): the standard exponential
//!   height fog, flat, or lit by the lights and marched with the volumes
//!   when In-scatter is on.
//!
//! Emissions become colors here, at luminance 1 ([`spectrum`]): Pixel
//! renders in RGB.
use crate::render::{Gobo, spectrum};
use kinogaki_render_api::standard::{self, Emission};
use kinogaki_render_api::value::{Choice, Condition, Number, NumberLook, Value, ValueKind};
use kinogaki_render_api::{Declared, PrimType, Role, SettingDescriptor, SettingGroup};
use kinogaki_studio_api::ObjectKindId;
use kinogaki_studio_api::render::DeclaredKind;

/// The Light kind's id.
pub const LIGHT_KIND: ObjectKindId = ObjectKindId::declared("kinogaki.pixel.light");
/// The Fog kind's id.
pub const FOG_KIND: ObjectKindId = ObjectKindId::declared("kinogaki.pixel.fog");
/// The rim light's type.
pub const RIM_LIGHT: &str = "kinogaki.pixel.rim_light";

/// The Gobo section's group.
const GOBO: &str = "gobo";

/// The gobo's parameters, under `GOBO`: a toggle, then the pattern's.
fn gobo_params() -> Vec<SettingDescriptor> {
    let on = || Condition::when("gobo", true);
    let shown = |d: SettingDescriptor| d.requires(on(), "Used with Gobo on");
    let slider = |min, max, soft: (f64, f64)| {
        ValueKind::Number(Number::plain().range(min, max).soft(soft.0, soft.1))
    };
    let look = |decimals| NumberLook::slider().decimals(decimals).speed(0.01);
    vec![
        SettingDescriptor::new("gobo", "Gobo", GOBO, ValueKind::Toggle, false)
            .tooltip("Masks the light with dappled shade, as if through leaves"),
        shown(
            SettingDescriptor::new(
                "goboScale",
                "Scale",
                GOBO,
                ValueKind::Number(Number::length().range(1e-3, 1e6).soft(0.1, 10.0)),
                1.0,
            )
            .look(NumberLook::default().decimals(2).speed(0.01)),
        ),
        shown(
            SettingDescriptor::new(
                "goboContrast",
                "Contrast",
                GOBO,
                slider(0.0, 1.0, (0.0, 1.0)),
                0.5,
            )
            .look(look(2)),
        ),
        shown(SettingDescriptor::new(
            "goboSeed",
            "Seed",
            GOBO,
            ValueKind::integer(0, 1 << 20),
            0i64,
        )),
        shown(SettingDescriptor::new(
            "goboOffset",
            "Offset",
            GOBO,
            ValueKind::Vector {
                components: 2,
                number: Number::length(),
            },
            Value::Numbers(vec![0.0, 0.0]),
        )),
    ]
}

/// The rim light's type: Side, then the standard emission settings, a warm
/// white of intensity 0.25 by default.
fn rim_light() -> PrimType {
    let sides = vec![Choice::new("lit", "Lit"), Choice::new("all", "All")];
    let side = SettingDescriptor::new(
        "side",
        "Side",
        standard::LIGHT_GROUP,
        ValueKind::choice(sides),
        standard::choice("lit"),
    )
    .tooltip("Lit: only where direct light reaches; All: round the whole silhouette");
    let emission = (standard::emission_params(0.25).into_iter()).map(|mut d| {
        if &*d.name == "color" {
            d.default = Value::Color([1.0, 0.85, 0.6]);
        }
        d
    });
    PrimType::new(RIM_LIGHT, "Rim Light", Role::Light).group(
        SettingGroup::new(standard::LIGHT_GROUP, "Light", true),
        [side].into_iter().chain(emission).collect(),
    )
}

/// Pixel's light types, in the order menus list them: the standard ones (a
/// Gobo on distant lights and spots), then the rim light.
pub fn light_types() -> Vec<PrimType> {
    let gobo = || SettingGroup::new(GOBO, "Gobo", false).open_when(Condition::when("gobo", true));
    (standard::lights().into_iter())
        .map(|ty| match &*ty.id {
            standard::DISTANT_LIGHT | standard::SPOT_LIGHT => ty.group(gobo(), gobo_params()),
            _ => ty,
        })
        .chain([rim_light()])
        .collect()
}

/// Pixel's lights, as a placed kind.
pub fn lights() -> DeclaredKind {
    DeclaredKind::lights(LIGHT_KIND, "Pixel Lights", light_types())
}

/// Pixel's fog, at the root.
pub fn fog() -> DeclaredKind {
    DeclaredKind::atmosphere(FOG_KIND, standard::fog())
}

/// The color `prim`'s emission makes, × intensity × 2^exposure (a light's
/// radiance, a rim's light).
pub fn emitted(prim: &Declared) -> [f32; 3] {
    let rgb = match Emission::of_key(prim.choice("emission")).unwrap_or_default() {
        Emission::Color => prim.color("color").map(f64::from),
        Emission::Temperature => {
            spectrum::blackbody(f64::from(prim.float("colorTemperature").max(1.0)))
        }
        Emission::Illuminant(name) => spectrum::illuminant(&name).unwrap_or([1.0; 3]),
    };
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

/// Whether a rim light shows all round, rather than where direct light
/// reaches.
pub fn rim_all_round(prim: &Declared) -> bool {
    prim.choice("side") == "all"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(ty: &PrimType, values: &[(&str, Value)]) -> Declared {
        let mut prim = Declared::new(&ty.id, ty.defaults());
        for (name, v) in values {
            prim.values.insert((*name).to_owned(), v.clone());
        }
        prim
    }

    #[test]
    fn distant_lights_and_spots_take_a_gobo_and_the_rim_comes_last() {
        let types = light_types();
        let ids: Vec<&str> = types.iter().map(|t| &*t.id).collect();
        assert_eq!(
            ids,
            [
                standard::AREA_LIGHT,
                standard::DOME_LIGHT,
                standard::DISTANT_LIGHT,
                standard::SPOT_LIGHT,
                RIM_LIGHT
            ]
        );
        let gobo_of = |t: &PrimType| t.params.iter().any(|d| &*d.name == "goboScale");
        assert_eq!(
            types.iter().map(gobo_of).collect::<Vec<_>>(),
            [false, false, true, true, false]
        );
        // Off by default, decoded when on.
        let spot = declared(&types[3], &[]);
        assert_eq!(gobo(&spot), None);
        let on = declared(
            &types[3],
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
