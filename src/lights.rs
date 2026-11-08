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
//! Emissions become colors here, at luminance 1 (the SDK's
//! [`Emission::color`]): Pixel renders in RGB.
use crate::render::Gobo;
use kinogaki_render_api::standard::{self, Emission};
use kinogaki_render_api::value::{Choice, Condition, Number, Schema, Value, ValueKind};
use kinogaki_render_api::{Declared, Parameter, PrimType, PrimTypeId, Role, Section};
use kinogaki_studio_api::ObjectKindId;
use kinogaki_studio_api::render::lights::DeclaredKind;

/// The Light kind's id.
pub const LIGHT_KIND: ObjectKindId = ObjectKindId::declared("kinogaki.pixel.light");
/// The Fog kind's id.
pub const FOG_KIND: ObjectKindId = ObjectKindId::declared("kinogaki.pixel.fog");
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
/// Gobo on distant lights and spots), then the rim light.
pub fn light_types() -> Vec<PrimType> {
    (standard::lights().into_iter())
        .map(|ty| match ty.id {
            standard::DISTANT_LIGHT | standard::SPOT_LIGHT => ty.merged(gobo_section()),
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

/// Whether a rim light shows all round, rather than where direct light
/// reaches.
pub fn rim_all_round(prim: &Declared) -> bool {
    prim.choice("side") == "all"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(ty: &PrimType, values: &[(&str, Value)]) -> Declared {
        let mut prim = Declared::new(ty.id, ty.schema().values());
        for (name, v) in values {
            prim.values.insert((*name).to_owned(), v.clone());
        }
        prim
    }

    #[test]
    fn distant_lights_and_spots_take_a_gobo_and_the_rim_comes_last() {
        let types = light_types();
        let ids: Vec<PrimTypeId> = types.iter().map(|t| t.id).collect();
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
        let gobo_of = |t: &PrimType| t.schema().parameters().any(|d| &*d.name == "goboScale");
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
