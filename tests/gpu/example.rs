//! The repository's Pixel example (`studio/examples/scenes/pixel.kino`)
//! opened in the editor and rendered to a PNG through its Pixel Render
//! Output, against Pixel's own reference (`tests/references/example.png`;
//! `PIXEL_REFERENCES=1` writes it instead).
use kinogaki_harness::{Harness, Key, Modifiers, Selection, key};
use kinogaki_studio_api::ObjectKindId;
use std::path::{Path, PathBuf};

/// The rendered image's size.
const SIZE: [u32; 2] = [384, 216];
/// The largest mean difference of a channel from the reference.
const TOLERANCE: f64 = 0.004;

/// A folder of the build's for what this test writes, emptied.
fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("pixel-example");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a folder");
    dir
}

/// Row `label`'s part `part` set to `text`, as a person types it.
fn type_into(h: &mut Harness, label: &str, part: usize, text: &str) {
    let row = h.row(label).unwrap_or_else(|| panic!("no row {label}"));
    h.click(row.parts.get(part).copied().unwrap_or(row.rect).center());
    h.frame(vec![]);
    h.pressed(Key::A, Modifiers::COMMAND);
    h.type_text(text);
    h.frame(vec![key(Key::Enter)]);
    h.settle();
    h.frame(vec![]);
}

/// The 8-bit RGBA pixels of the PNG at `path`.
fn pixels(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let image = kinogaki_png::decode(&bytes, 1 << 24).expect("a PNG");
    assert_eq!([image.width, image.height], SIZE);
    match image.samples {
        kinogaki_png::Samples::U8(v) if image.channels == 4 => v,
        kinogaki_png::Samples::U8(v) => v.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        _ => panic!("8-bit"),
    }
}

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn the_example_renders_as_its_reference() {
    let Some(mut h) = Harness::start(
        Selection::names("")
            .with(kinogaki_mesh_kernel::package())
            .with(kinogaki_mesh::package())
            .with(kinogaki_pixel::package()),
    ) else {
        return;
    };
    kinogaki_gpu_turns::share();
    // On the GPU from here on: the harness draws on a device that renders.
    let _ = h.grab();
    let dir = scratch();
    let example =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../studio/examples/scenes/pixel.kino");
    let copy = dir.join("pixel.kino");
    std::fs::copy(&example, &copy).expect("the example");
    h.open(&copy);
    assert_eq!(h.error(), None);
    let outputs =
        h.cx(|cx| cx.objects_of_kind(ObjectKindId::declared("kinogaki.core.render_output")));
    let pixel = outputs[0];
    h.cx(|cx| cx.select_objects(&[pixel]));
    h.settle();
    h.frame(vec![]);
    let out = dir.join("pixel.png");
    type_into(&mut h, "Resolution", 0, &SIZE[0].to_string());
    type_into(&mut h, "Resolution", 1, &SIZE[1].to_string());
    type_into(&mut h, "Output", 0, &out.display().to_string());
    h.run_item("render_image");
    h.until("the final render", |h| !h.finals_running());
    assert_eq!(h.error(), None);
    let image = pixels(&out);
    let reference = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/references/example.png");
    if std::env::var_os("PIXEL_REFERENCES").is_some() {
        std::fs::copy(&out, &reference).expect("the reference");
        return;
    }
    let held = pixels(&reference);
    let (mut sum, mut worst) = (0.0, 0.0f64);
    for (pixel, reference) in image.chunks(4).zip(held.chunks(4)) {
        for k in 0..3 {
            let d = (f64::from(pixel[k]) - f64::from(reference[k])).abs() / 255.0;
            sum += d;
            worst = worst.max(d);
        }
    }
    let mean = sum / (image.len() / 4 * 3) as f64;
    eprintln!("example: mean {mean:.5}, worst {worst:.3}");
    assert!(
        mean < TOLERANCE,
        "the example differs from its reference: mean {mean}"
    );
}
