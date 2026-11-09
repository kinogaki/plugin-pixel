//! The geometry lights an instance casts no shadow of. Pixel lights by a
//! point inside a geometry light's mesh, so the mesh's own instances must
//! cast none of its shadows: each geometry light takes a slot (at most
//! [`MOST`]; past them a light's mesh shadows it, said), recorded on its
//! light (`seed.y` in light.wgsl), and each instance holds the mask of the
//! slots whose light it casts no shadow of (`shadowless` in common.wgsl).
//! An instance answers for the prim placing it: its shape on its own, else
//! the outermost Instancer (the prims the host resolves object rows to).
use kinogaki_render_api::standard::{Light, LightShape};
use kinogaki_render_api::{Declared, Members, PrimId};

/// The most geometry lights with a slot: the mask's 128 bits.
pub const MOST: usize = 128;
/// The slot of a light whose shadows every instance casts.
pub const NO_SLOT: u32 = u32::MAX;

/// The geometry lights, each its slot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shadowless {
    /// In slot order: the light, and the prims of its mesh.
    lights: Vec<(PrimId, Members)>,
    /// How many more geometry lights there are, past [`MOST`].
    pub dropped: usize,
}

impl Shadowless {
    /// The geometry lights among `prims`.
    pub fn of<'a>(prims: impl IntoIterator<Item = (PrimId, &'a Declared)>) -> Shadowless {
        let mut held = Shadowless::default();
        for (id, prim) in prims {
            let geometry = Light::decode(prim).map(|l| l.shape);
            if !matches!(geometry, Some(LightShape::Geometry { .. })) {
                continue;
            }
            match held.lights.len() < MOST {
                true => held.lights.push((id, prim.members("geometry"))),
                false => held.dropped += 1,
            }
        }
        held
    }

    /// Light `light`'s slot ([`NO_SLOT`]: every instance casts its
    /// shadows).
    pub fn slot(&self, light: PrimId) -> u32 {
        (self.lights.iter())
            .position(|(id, _)| *id == light)
            .map_or(NO_SLOT, |at| at as u32)
    }

    /// The mask of an instance `placing` places: the slots of the geometry
    /// lights whose mesh it is, 32 a word.
    pub fn mask(&self, placing: PrimId) -> [u32; 4] {
        let mut mask = [0u32; 4];
        for (slot, (_, meshes)) in self.lights.iter().enumerate() {
            if meshes.holds(placing) {
                mask[slot / 32] |= 1 << (slot % 32);
            }
        }
        mask
    }
}
