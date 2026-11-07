//! The texture a material's color reads, as its texture node says:
//! MaterialX's `image` and glTF's texture nodes (`gltf_image`,
//! `gltf_colorimage`, and Kinogaki's with glTF's minification filter), read
//! natively: the coordinate set and placement (a `texcoord` or
//! `geompropvalue` node, a `place2d` of one or of its own coordinates, a
//! glTF texture's own offset, rotation and scale, by the graphs MaterialX
//! defines them with), each axis's wrap, the filter (`closest` is the
//! nearest texel), the factors multiplying the texel and the vertex colors
//! and alpha glTF's color image multiplies it by.
use crate::render::Wrap;
use kinogaki_render_api::geometry::image::ImageSource;
use kinogaki_render_api::geometry::names;
use kinogaki_render_api::value::Value;
use kinogaki_render_api::{MaterialNetwork, PrimId};
use std::sync::Arc;

/// The texture nodes Pixel samples.
pub const NODES: [&str; 5] = [
    "image",
    "gltf_image",
    "gltf_colorimage",
    "kinogaki_gltf_image",
    "kinogaki_gltf_colorimage",
];

/// An affine map of texture coordinates, by rows: `(u, v, 1)` to `(s, t)`.
type Affine = [[f64; 3]; 2];

const IDENTITY: Affine = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

/// A texture as a material's node reads it; the host's Texture prim of its
/// image resolves it into the scene's [`Texture`](crate::render::Texture).
#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    /// The image the node reads, as authored (what the viewport shows).
    pub source: Option<ImageSource>,
    /// The Texture prim the host read it to (none: not read yet).
    pub image: Option<PrimId>,
    /// The primvar holding its coordinates.
    pub uv: Arc<str>,
    /// Where coordinates `(u, v)` read it: `transform · (u, v, 1)`.
    pub transform: [[f32; 3]; 2],
    /// How each axis reads past the edges.
    pub wrap: [Wrap; 2],
    /// The nearest texel, else bilinear.
    pub nearest: bool,
    /// What a texel is multiplied by.
    pub factor: [f32; 4],
    /// The attribute multiplying it too (glTF's vertex colors).
    pub tint: Option<Arc<str>>,
    /// The attribute multiplying its alpha (glTF's vertex colors' alpha,
    /// MaterialX's geometric color4's fourth: the mesh's `Alpha`).
    pub alpha: Option<Arc<str>>,
}

/// Whether node `at` of `network` is a texture node Pixel samples.
pub fn is_texture(network: &MaterialNetwork, at: u32) -> bool {
    NODES.contains(&kind(network, at))
}

/// The MaterialX node `at` is, by name; empty for another.
fn kind(network: &MaterialNetwork, at: u32) -> &str {
    (network.nodes()[at as usize].def.mtlx()).map_or("", |d| d.node.as_str())
}

/// What drives input `input` of `at`.
fn driver(network: &MaterialNetwork, at: u32, input: &str) -> Option<u32> {
    (network.links().iter())
        .find(|l| l.to == at && &*l.input == input)
        .map(|l| l.from)
}

/// Two numbers of value `name` of node `at`.
fn pair(network: &MaterialNetwork, at: u32, name: &str, default: [f64; 2]) -> [f64; 2] {
    match network.nodes()[at as usize].values.get(name) {
        Some(Value::Numbers(n)) if n.len() == 2 => [n[0], n[1]],
        Some(Value::Number(n)) => [*n; 2],
        _ => default,
    }
}

fn number(network: &MaterialNetwork, at: u32, name: &str, default: f64) -> f64 {
    let v = network.nodes()[at as usize].values.get(name);
    v.and_then(Value::number).unwrap_or(default)
}

fn integer(network: &MaterialNetwork, at: u32, name: &str) -> i64 {
    let v = network.nodes()[at as usize].values.get(name);
    v.and_then(Value::integer).unwrap_or(0)
}

/// Texture node `at` as Pixel samples it; none for another node.
pub fn sampled(network: &MaterialNetwork, at: u32) -> Option<Sampled> {
    let node_kind = kind(network, at);
    if !NODES.contains(&node_kind) {
        return None;
    }
    let node = &network.nodes()[at as usize];
    let choice = |name: &str| node.values.get(name).and_then(Value::choice);
    let wrap = |name: &str| match choice(name) {
        Some("clamp") => Wrap::Clamp,
        Some("mirror") => Wrap::Mirror,
        Some("constant") => Wrap::Black,
        _ => Wrap::Repeat,
    };
    let (placed, uv) = match driver(network, at, "texcoord") {
        Some(from) => affine(network, from)?,
        None => (IDENTITY, Arc::from(names::UV)),
    };
    let transform = match node_kind {
        "image" => placed,
        kind => then(
            placed,
            gltf_placement(network, at, kind.ends_with("colorimage")),
        ),
    };
    let factor = match (node_kind.ends_with("colorimage"), node.values.get("factor")) {
        (true, _) => four(node.values.get("color")),
        (false, Some(Value::Number(f))) => [*f as f32, *f as f32, *f as f32, 1.0],
        (false, factor) => four(factor),
    };
    let vertex = driver(network, at, "geomcolor").is_some();
    Some(Sampled {
        source: node.images.get("file").cloned(),
        image: node.textures.get("file").map(|id| id.prim()),
        uv,
        transform: transform.map(|row| row.map(|v| v as f32)),
        wrap: [wrap("uaddressmode"), wrap("vaddressmode")],
        nearest: choice("filtertype") == Some("closest"),
        factor,
        tint: vertex.then(|| Arc::from(names::CD)),
        alpha: vertex.then(|| Arc::from(names::ALPHA)),
    })
}

/// A color, color4 or number value as rgba (white where none).
fn four(value: Option<&Value>) -> [f32; 4] {
    match value {
        Some(Value::Color(c)) => [c[0] as f32, c[1] as f32, c[2] as f32, 1.0],
        Some(Value::Numbers(n)) => std::array::from_fn(|k| n.get(k).map_or(1.0, |v| *v as f32)),
        Some(Value::Number(n)) => [*n as f32, *n as f32, *n as f32, 1.0],
        _ => [1.0; 4],
    }
}

/// `b` after `a`.
fn then(a: Affine, b: Affine) -> Affine {
    let row = |r: [f64; 3]| {
        [
            r[0] * a[0][0] + r[1] * a[1][0],
            r[0] * a[0][1] + r[1] * a[1][1],
            r[0] * a[0][2] + r[1] * a[1][2] + r[2],
        ]
    };
    [row(b[0]), row(b[1])]
}

/// MaterialX's `rotate2d` by `degrees`.
fn rotation(degrees: f64) -> Affine {
    let (sa, ca) = degrees.to_radians().sin_cos();
    [[ca, sa, 0.0], [-sa, ca, 0.0]]
}

fn moved([x, y]: [f64; 2]) -> Affine {
    [[1.0, 0.0, x], [0.0, 1.0, y]]
}

fn scaled([x, y]: [f64; 2]) -> Affine {
    [[x, 0.0, 0.0], [0.0, y, 0.0]]
}

/// MaterialX's `place2d` (its graph): about `pivot`, divided by `scale`,
/// rotated by `rotate` degrees and moved back by `offset`, in that order
/// (`operationorder` 0), or the reverse (1).
fn place2d(pivot: [f64; 2], scale: [f64; 2], rotate: f64, offset: [f64; 2], order: i64) -> Affine {
    let steps = [
        scaled([1.0 / scale[0], 1.0 / scale[1]]),
        rotation(rotate),
        moved([-offset[0], -offset[1]]),
    ];
    let steps = match order {
        1 => [steps[2], steps[1], steps[0]],
        _ => steps,
    };
    let mut at = moved([-pivot[0], -pivot[1]]);
    for step in steps {
        at = then(at, step);
    }
    then(at, moved(pivot))
}

/// A glTF texture node's own placement, by MaterialX's graph of
/// `gltf_image`: `place2d` of its pivot, the inverse of its scale, its
/// rotation reversed and its offset with u reversed (glTF's rows run down);
/// `gltf_colorimage`'s graph places in order 0 whatever its own says.
fn gltf_placement(network: &MaterialNetwork, at: u32, color: bool) -> Affine {
    let scale = pair(network, at, "scale", [1.0; 2]);
    let offset = pair(network, at, "offset", [0.0; 2]);
    let order = match color {
        true => 0,
        false => integer(network, at, "operationorder"),
    };
    place2d(
        pair(network, at, "pivot", [0.0, 1.0]),
        [1.0 / scale[0], 1.0 / scale[1]],
        -number(network, at, "rotate", 0.0),
        [-offset[0], offset[1]],
        order,
    )
}

/// Node `at`'s coordinates as an affine map of a coordinate set's, and that
/// set's primvar: a `texcoord` (`uv`, `uv2`…), a `geompropvalue`, or a
/// `place2d` of either or of its own coordinates (its `texcoord` value,
/// (0, 0) by default, when nothing drives it: a map of no set's).
fn affine(network: &MaterialNetwork, at: u32) -> Option<(Affine, Arc<str>)> {
    match kind(network, at) {
        "texcoord" => {
            let set = match integer(network, at, "index") {
                0 => names::UV.to_owned(),
                n => format!("{}{}", names::UV, n + 1),
            };
            Some((IDENTITY, Arc::from(set)))
        }
        "geompropvalue" => {
            let name = network.nodes()[at as usize]
                .values
                .get("geomprop")?
                .text()?;
            Some((IDENTITY, Arc::from(name)))
        }
        "place2d" => {
            let (inner, set) = match driver(network, at, "texcoord") {
                Some(from) => affine(network, from)?,
                None => {
                    let [u, v] = pair(network, at, "texcoord", [0.0; 2]);
                    ([[0.0, 0.0, u], [0.0, 0.0, v]], Arc::from(names::UV))
                }
            };
            let placed = place2d(
                pair(network, at, "pivot", [0.0; 2]),
                pair(network, at, "scale", [1.0; 2]),
                number(network, at, "rotate", 0.0),
                pair(network, at, "offset", [0.0; 2]),
                integer(network, at, "operationorder"),
            );
            Some((then(inner, placed), set))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::material::mtlx::def;
    use kinogaki_render_api::materialx::{Value as M, library};
    use kinogaki_render_api::value::Parameters;
    use kinogaki_render_api::{MaterialNetwork, NetworkBuilder};

    /// The node `signature` holding `values`.
    fn node(built: &mut NetworkBuilder, signature: &str, values: &[(&str, Value)]) -> u32 {
        let values: Parameters = (values.iter())
            .map(|(n, v)| ((*n).to_owned(), v.clone()))
            .collect();
        built.node(&def(signature).unwrap(), &values).unwrap()
    }

    /// MaterialX's own graph of `signature`'s placement, evaluated at `uv`:
    /// what the affine map must agree with.
    fn evaluated(at: [f64; 2], placed: Affine) -> [f64; 2] {
        placed.map(|[a, b, c]| a * at[0] + b * at[1] + c)
    }

    #[test]
    fn a_gltf_texture_reads_its_placement_wraps_filter_and_factor() {
        let mut built = MaterialNetwork::builder();
        let texture = node(
            &mut built,
            "ND_kinogaki_gltf_image_color3",
            &[
                ("offset", Value::Numbers(vec![0.25, 0.5])),
                ("rotate", Value::Number(90.0)),
                ("scale", Value::Numbers(vec![2.0, 4.0])),
                ("uaddressmode", Value::Choice("clamp".into())),
                ("vaddressmode", Value::Choice("mirror".into())),
                ("filtertype", Value::Choice("closest".into())),
                ("factor", Value::Color([0.5, 0.25, 1.0])),
            ],
        );
        let set = node(
            &mut built,
            "ND_texcoord_vector2",
            &[("index", Value::Integer(1))],
        );
        built.link(set, "out", texture, "texcoord").unwrap();
        let network = built.build(texture).unwrap();
        let read = sampled(&network, texture).unwrap();
        assert_eq!((&*read.uv, read.wrap), ("uv2", [Wrap::Clamp, Wrap::Mirror]));
        assert!(read.nearest);
        assert_eq!(read.factor, [0.5, 0.25, 1.0, 1.0]);
        // The transform is MaterialX's graph of gltf_image: evaluate its
        // place2d step by step with the evaluator and compare.
        let library = library();
        let step = |name: &str, inputs: &[(&str, M)]| {
            let d = library.def(name).unwrap();
            d.evaluate(&|n| inputs.iter().find(|(k, _)| *k == n).map(|(_, v)| v.clone()))
                .unwrap()
        };
        let v = |x: f64, y: f64| M::Numbers(vec![x, y]);
        for uv in [[0.0, 0.0], [1.0, 0.0], [0.3, 0.7]] {
            // place2d(uv; pivot (0, 1), scale 1/scale, rotate -90, offset (-0.25, 0.5)).
            let s = step(
                "ND_subtract_vector2",
                &[("in1", v(uv[0], uv[1])), ("in2", v(0.0, 1.0))],
            );
            let s = step("ND_divide_vector2", &[("in1", s), ("in2", v(0.5, 0.25))]);
            let s = step(
                "ND_rotate2d_vector2",
                &[("in", s), ("amount", M::Numbers(vec![-90.0]))],
            );
            let s = step("ND_subtract_vector2", &[("in1", s), ("in2", v(-0.25, 0.5))]);
            let s = step("ND_add_vector2", &[("in1", s), ("in2", v(0.0, 1.0))]);
            let expected = s.numbers().unwrap().to_vec();
            let got = evaluated(uv, read.transform.map(|r| r.map(f64::from)));
            assert!(
                (0..2).all(|k| (got[k] - expected[k]).abs() < 1e-5),
                "{uv:?}: {got:?} {expected:?}"
            );
        }
    }

    /// A `place2d` whose coordinates nothing drives places its own value,
    /// as MaterialX defines: every point reads one texel.
    #[test]
    fn an_undriven_place2d_places_its_own_coordinates() {
        let mut built = MaterialNetwork::builder();
        let image = node(&mut built, "ND_image_color3", &[]);
        let place = node(
            &mut built,
            "ND_place2d_vector2",
            &[
                ("texcoord", Value::Numbers(vec![0.5, 0.0])),
                ("offset", Value::Numbers(vec![-0.25, -0.25])),
            ],
        );
        built.link(place, "out", image, "texcoord").unwrap();
        let network = built.build(image).unwrap();
        let read = sampled(&network, image).unwrap();
        for uv in [[0.0, 0.0], [0.3, 0.9], [1.0, 1.0]] {
            let at = evaluated(uv, read.transform.map(|r| r.map(f64::from)));
            assert!(
                (at[0] - 0.75).abs() < 1e-6 && (at[1] - 0.25).abs() < 1e-6,
                "{uv:?}: {at:?}"
            );
        }
    }

    #[test]
    fn an_image_reads_its_place2d_and_a_color_image_its_color_and_vertex_colors() {
        let mut built = MaterialNetwork::builder();
        let image = node(&mut built, "ND_image_color3", &[]);
        let place = node(
            &mut built,
            "ND_place2d_vector2",
            &[("scale", Value::Numbers(vec![2.0, 2.0]))],
        );
        let set = node(&mut built, "ND_texcoord_vector2", &[]);
        built.link(set, "out", place, "texcoord").unwrap();
        built.link(place, "out", image, "texcoord").unwrap();
        let network = built.build(image).unwrap();
        let read = sampled(&network, image).unwrap();
        assert_eq!(read.transform, [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]);
        assert_eq!((read.factor, read.tint.as_deref()), ([1.0; 4], None));
        let mut built = MaterialNetwork::builder();
        let color = [0.5, 0.25, 0.125, 0.75].map(f64::from).to_vec();
        let image = node(
            &mut built,
            "ND_kinogaki_gltf_colorimage",
            &[("color", Value::Numbers(color))],
        );
        let vertex = node(&mut built, "ND_geomcolor_color4", &[]);
        built.link(vertex, "out", image, "geomcolor").unwrap();
        let network = built.build(image).unwrap();
        let read = sampled(&network, image).unwrap();
        assert_eq!(
            (read.factor, read.tint.as_deref()),
            ([0.5, 0.25, 0.125, 0.75], Some("Cd"))
        );
        // No placement: the coordinates as they are.
        let id = evaluated([0.3, 0.6], read.transform.map(|r| r.map(f64::from)));
        assert!(
            (id[0] - 0.3).abs() < 1e-6 && (id[1] - 0.6).abs() < 1e-6,
            "{id:?}"
        );
    }
}
