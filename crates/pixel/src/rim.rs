//! The rim light, Pixel's own light type ([`prim_type`]): a Fresnel-like
//! edge, (1 − n·v)⁴ × its emission's color × intensity × 2^exposure, added
//! to the light a surface takes, where direct light reaches it or all
//! round. It has no place or direction (its transform is unused). A host
//! sends it as a declared prim, decoded by [`Rim::decode`].

use std::sync::Arc;

use kinogaki_render_api::settings::set_value;
use kinogaki_render_api::standard::{self, Emission, LIGHT_GROUP, choice};
use kinogaki_render_api::{
    Choice, Declared, PrimType, Role, SettingDescriptor, SettingGroup, SettingKind, SettingValue,
};

/// The rim light's id.
pub const ID: &str = "kinogaki.pixel.rim_light";

/// Where a rim light shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum RimSide {
    /// Only where direct light reaches a surface.
    #[default]
    Lit,
    /// All round the silhouette.
    All,
}

impl RimSide {
    /// Its key in files.
    pub fn key(self) -> &'static str {
        match self {
            RimSide::Lit => "lit",
            RimSide::All => "all",
        }
    }
}

/// The rim light's type: Side, then the standard emission settings, a warm
/// white of intensity 0.25 by default.
pub fn prim_type() -> PrimType {
    let side = SettingDescriptor::new(
        "side",
        "Side",
        LIGHT_GROUP,
        SettingKind::Choice {
            choices: vec![Choice::new("lit", "Lit"), Choice::new("all", "All")],
            default: Arc::from(RimSide::Lit.key()),
            menu: false,
        },
    );
    let mut params = vec![side];
    params.extend(standard::emission_params(0.25).into_iter().map(|d| {
        if &*d.name == "color" {
            SettingDescriptor {
                kind: SettingKind::Color {
                    default: [1.0, 0.85, 0.6],
                },
                ..d
            }
        } else {
            d
        }
    }));
    PrimType {
        id: Arc::from(ID),
        label: Arc::from("Rim Light"),
        role: Role::Light,
        groups: vec![SettingGroup::new(LIGHT_GROUP, "Light", true)],
        params,
    }
}

/// A rim light, decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct Rim {
    /// Where it shows.
    pub side: RimSide,
    /// Where the spectrum comes from.
    pub emission: Emission,
    /// Its color, linear ACEScg: the spectrum's in [`Emission::Color`].
    pub color: [f32; 3],
    /// Its intensity.
    pub intensity: f32,
    /// The intensity is scaled by 2 to this power.
    pub exposure: f32,
    /// The blackbody's kelvin in [`Emission::Temperature`].
    pub color_temperature: f32,
}

impl Default for Rim {
    /// The type's defaults.
    fn default() -> Rim {
        Rim::decode(&Declared::new(ID, prim_type().defaults())).expect("a rim")
    }
}

impl Rim {
    /// `prim` as a rim light (none when it is of another type).
    pub fn decode(prim: &Declared) -> Option<Rim> {
        (&*prim.type_id == ID).then(|| Rim {
            side: match prim.choice("side") {
                "all" => RimSide::All,
                _ => RimSide::Lit,
            },
            emission: Emission::of_key(prim.choice("emission")).unwrap_or_default(),
            color: prim.color("color"),
            intensity: prim.float("intensity"),
            exposure: prim.float("exposure"),
            color_temperature: prim.float("colorTemperature"),
        })
    }

    /// It as a prim: its type's defaults, with its own values.
    pub fn declared(&self) -> Declared {
        let mut values = prim_type().defaults();
        let mut set = |name: &str, v: SettingValue| set_value(&mut values, name, v);
        let float = |v: f32| SettingValue::Float(f64::from(v));
        set("side", choice(self.side.key()));
        set("emission", choice(self.emission.key()));
        set("color", SettingValue::Color(self.color.map(f64::from)));
        set("intensity", float(self.intensity));
        set("exposure", float(self.exposure));
        set("colorTemperature", float(self.color_temperature));
        Declared::new(ID, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rim_round_trips_and_defaults_to_a_warm_quarter() {
        let d = Rim::default();
        assert_eq!(
            (d.side, d.color, d.intensity, d.exposure),
            (RimSide::Lit, [1.0, 0.85, 0.6], 0.25, 0.0)
        );
        let rim = Rim {
            side: RimSide::All,
            emission: Emission::of_key("D65").unwrap(),
            color: [0.2, 0.4, 0.6],
            intensity: 2.0,
            exposure: -1.0,
            color_temperature: 3000.0,
        };
        assert_eq!(Rim::decode(&rim.declared()), Some(rim));
        assert_eq!(Rim::decode(&standard::Fog::default().declared()), None);
        let names: Vec<_> = prim_type().params.iter().map(|d| d.name.clone()).collect();
        assert_eq!(
            names.iter().map(|n| &**n).collect::<Vec<_>>(),
            [
                "side",
                "emission",
                "color",
                "colorTemperature",
                "intensity",
                "exposure"
            ]
        );
    }
}
