//! Pixel's outputs: the beauty, then what a compositor separates it by.
//! Colors (beauty, albedo and the light passes, volume) are linear ACEScg;
//! occlusion, depth, normal and object id are data.
use kinogaki_render_api::aov::standard;
use kinogaki_render_api::{Aov, AovClass};

/// Pixel's own output: the domes' and rims' light.
const AMBIENT: &str = "ambient";

/// An output a frame writes, each a film-size texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Output {
    /// The shaded, dithered film.
    Beauty,
    /// The material's color × its tint.
    Albedo,
    /// Light from the lights, off a white surface.
    Direct,
    /// The domes' light × AO, off a white surface.
    Ambient,
    /// One diffuse bounce, off a white surface.
    Indirect,
    /// The volumes' light and opacity, premultiplied.
    Volume,
    /// The open share of the hemisphere.
    Occlusion,
    /// Distance along the view direction.
    Depth,
    /// The shading normal, facing the camera.
    Normal,
    /// The object id.
    ObjectId,
}

impl Output {
    /// Every output, the beauty first: the order hosts list them in.
    pub const ALL: [Output; 10] = [
        Output::Beauty,
        Output::Albedo,
        Output::Direct,
        Output::Ambient,
        Output::Indirect,
        Output::Volume,
        Output::Occlusion,
        Output::Depth,
        Output::Normal,
        Output::ObjectId,
    ];

    /// Its id in files.
    pub fn id(self) -> &'static str {
        match self {
            Output::Beauty => standard::BEAUTY,
            Output::Albedo => standard::ALBEDO,
            Output::Direct => standard::DIRECT,
            Output::Ambient => AMBIENT,
            Output::Indirect => standard::INDIRECT,
            Output::Volume => standard::VOLUME,
            Output::Occlusion => standard::OCCLUSION,
            Output::Depth => standard::DEPTH,
            Output::Normal => standard::NORMAL,
            Output::ObjectId => standard::OBJECT_ID,
        }
    }

    /// The output of id `id`.
    pub fn of_id(id: &str) -> Option<Output> {
        Output::ALL.into_iter().find(|o| o.id() == id)
    }

    /// The texture's label.
    pub fn label(self) -> &'static str {
        match self {
            Output::Beauty => "pixel beauty",
            Output::Albedo => "pixel albedo",
            Output::Direct => "pixel direct",
            Output::Ambient => "pixel ambient",
            Output::Indirect => "pixel indirect",
            Output::Volume => "pixel volume",
            Output::Occlusion => "pixel occlusion",
            Output::Depth => "pixel depth",
            Output::Normal => "pixel normal",
            Output::ObjectId => "pixel object id",
        }
    }

    /// The G-buffer pass draws it; the compute passes write the others.
    pub fn rasterized(self) -> bool {
        matches!(
            self,
            Output::Albedo | Output::Depth | Output::Normal | Output::ObjectId
        )
    }

    /// It as the SDK describes it: a standard output, or Ambient, Pixel's own.
    pub fn aov(self) -> Aov {
        match self {
            Output::Ambient => Aov::new(AMBIENT, "Ambient Light", AovClass::Color),
            other => standard::get(other.id()).expect("a standard output"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_output_is_described_by_its_id() {
        for output in Output::ALL {
            assert_eq!(&*output.aov().id, output.id());
            assert_eq!(Output::of_id(output.id()), Some(output));
        }
        assert!(Output::Beauty.aov().is_beauty());
        assert!(Output::ObjectId.aov().is_id());
        assert_eq!(Output::Volume.aov().channels(), 4);
    }
}
