//! AO's strength, power and share on direct light on the GPU, against the
//! traced AO of the same frame (`shading::occlusion`). `cargo test --
//! --ignored` runs them.

mod support;

use kinogaki_pixel_render::{Aov, AovImage, Settings, shading};
use support::{canonical, canonical_view, render, rgb};

fn rendered(settings: Settings) -> (AovImage, AovImage, AovImage) {
    let v = render(&canonical(), settings, &canonical_view(640, 360));
    let read = |aov| v.read_aov(aov).unwrap();
    (read(Aov::Ao), read(Aov::Direct), read(Aov::Id))
}

#[test]
#[ignore = "needs a GPU"]
fn ao_strength_and_power_shape_the_traced_ao() {
    let (traced, _, ids) = rendered(Settings::default());
    // The contact under the sphere is partly occluded.
    assert!(traced.floats().iter().any(|&a| a > 0.2 && a < 0.8));
    for (strength, power) in [(0.0, 1.0), (0.5, 1.0), (1.0, 2.0), (0.7, 0.5)] {
        let (shaped, _, _) = rendered(Settings {
            ao_strength: strength,
            ao_power: power,
            ..Settings::default()
        });
        for (i, &a) in traced.floats().iter().enumerate() {
            if ids.ints()[i] == u32::MAX {
                continue;
            }
            let want = shading::occlusion(a, strength, power);
            let got = shaped.floats()[i];
            assert!(
                (got - want).abs() < 1e-5,
                "{strength} {power}: {got} {want}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn ao_on_direct_darkens_direct_light_by_the_ao() {
    let (ao, direct, ids) = rendered(Settings::default());
    let half = Settings {
        ao_direct: 0.5,
        ..Settings::default()
    };
    let (_, darkened, _) = rendered(half);
    let mut touched = 0;
    for i in 0..ids.ints().len() {
        if ids.ints()[i] == u32::MAX {
            continue;
        }
        let k = 1.0 - 0.5 * (1.0 - ao.floats()[i]);
        let (want, got) = (rgb(&direct, i).map(|c| c * k), rgb(&darkened, i));
        // Half floats: 2⁻¹⁰ relative.
        assert!((0..3).all(|c| (got[c] - want[c]).abs() <= want[c] * 2e-3 + 1e-5));
        touched += usize::from(k < 0.95 && want[0] > 0.01);
    }
    assert!(touched > 20, "lit and occluded pixels: {touched}");
}
