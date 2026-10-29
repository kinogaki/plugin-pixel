//! The outputs a frame writes, at film resolution, and their CPU form.
//!
//! Colors (beauty, albedo, direct, ambient, indirect, volume) are linear ACEScg; the others
//! are data. Each is one of `kinogaki-render-api`'s standard outputs
//! ([`Aov::id`]), the contract compositing reads them by.

use kinogaki_render_api::aov::standard;

/// An output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Aov {
    /// Albedo × the quantized, dithered light; alpha 1 on geometry, the
    /// dome (alpha 1) or transparent black (alpha 0) elsewhere.
    Beauty,
    /// The Pixel Surface's color.
    Albedo,
    /// Light from the lights off a white Lambert surface, through
    /// translucent surfaces too, × AO as AO on Direct asks.
    Direct,
    /// The domes' light × AO.
    Ambient,
    /// One bounce of direct light off nearby surfaces (global
    /// illumination) off a white Lambert surface, × gi.
    Indirect,
    /// The light volumes and lit fog scatter toward the camera (rgb, plus
    /// emission), and their opacity (a: 1 − their
    /// transmittance to the first surface), premultiplied, before shade
    /// quantization.
    Volume,
    /// Ambient occlusion as applied: the open fraction of the
    /// cosine-weighted hemisphere within ao_distance, shaped by AO Power
    /// and Strength (1 channel).
    Ao,
    /// Distance along the view direction, in scene units; 0 off geometry
    /// (1 channel).
    Depth,
    /// The world-space unit normal facing the camera; 0 off geometry.
    Normal,
    /// The mesh's id; `u32::MAX` off geometry (1 channel).
    Id,
}

impl Aov {
    /// Every output.
    pub const ALL: [Aov; 10] = [
        Aov::Beauty,
        Aov::Albedo,
        Aov::Direct,
        Aov::Ambient,
        Aov::Indirect,
        Aov::Volume,
        Aov::Ao,
        Aov::Depth,
        Aov::Normal,
        Aov::Id,
    ];

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Aov::Beauty => "beauty",
            Aov::Albedo => "albedo",
            Aov::Direct => "direct",
            Aov::Ambient => "ambient",
            Aov::Indirect => "indirect",
            Aov::Volume => "volume",
            Aov::Ao => "ao",
            Aov::Depth => "depth",
            Aov::Normal => "normal",
            Aov::Id => "id",
        }
    }

    /// The standard output's id it is.
    pub fn id(self) -> &'static str {
        match self {
            Aov::Beauty => standard::BEAUTY,
            Aov::Albedo => standard::ALBEDO,
            Aov::Direct => standard::DIRECT,
            Aov::Ambient => standard::AMBIENT,
            Aov::Indirect => standard::INDIRECT,
            Aov::Volume => standard::VOLUME,
            Aov::Ao => standard::OCCLUSION,
            Aov::Depth => standard::DEPTH,
            Aov::Normal => standard::NORMAL,
            Aov::Id => standard::OBJECT_ID,
        }
    }

    /// The standard output it is.
    pub fn output(self) -> kinogaki_render_api::Aov {
        standard::get(self.id()).expect("a standard output")
    }

    /// The output of id `id`, if Pixel Render writes it.
    pub fn of_id(id: &str) -> Option<Aov> {
        Aov::ALL.into_iter().find(|a| a.id() == id)
    }

    /// Whether it holds ACEScg colors.
    pub fn is_color(self) -> bool {
        matches!(
            self,
            Aov::Beauty | Aov::Albedo | Aov::Direct | Aov::Ambient | Aov::Indirect | Aov::Volume
        )
    }
}

/// An output's values, `channels` a pixel, rows from the top.
#[derive(Clone, Debug, PartialEq)]
pub enum AovPixels {
    /// Floats (half floats widened).
    F32(Vec<f32>),
    /// Integers (ids).
    U32(Vec<u32>),
}

/// An output read back to the CPU.
#[derive(Clone, Debug, PartialEq)]
pub struct AovImage {
    /// Width, in pixels.
    pub width: u32,
    /// Height, in pixels.
    pub height: u32,
    /// Values a pixel: 4 (RGBA) or 1.
    pub channels: u32,
    /// The values.
    pub pixels: AovPixels,
}

impl AovImage {
    /// The float values (empty for ids).
    pub fn floats(&self) -> &[f32] {
        match &self.pixels {
            AovPixels::F32(v) => v,
            AovPixels::U32(_) => &[],
        }
    }

    /// The integer values (empty for floats).
    pub fn ints(&self) -> &[u32] {
        match &self.pixels {
            AovPixels::U32(v) => v,
            AovPixels::F32(_) => &[],
        }
    }
}
