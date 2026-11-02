//! A mesh's color attribute as corner tints: what a Pixel Surface's Color
//! is multiplied by in the G-buffer's albedo.
//!
//! The attribute a material names is read from the mesh prim's attributes
//! (a host sends the ones its materials name) and spread to three values a
//! triangle, which the raster interpolates:
//!
//! | Attribute class | Render domain | Across a triangle |
//! |---|---|---|
//! | Point | `Point` | blended by barycentric weight |
//! | Vertex | `Corner` | blended by barycentric weight |
//! | Primitive (face, piece) | `Triangle` | flat: one color |
//! | Detail | `Constant` | one color for the mesh |
//!
//! One component is a grey, two fill red and green, three or four give
//! red, green and blue (the fourth is ignored). A mesh without the
//! attribute, or with one of the wrong length, gets no tints: the Color
//! alone.

use std::sync::Arc;

use kinogaki_render_api::{AttributeDomain, Mesh as RenderMesh, MeshAttribute};

/// The corner tints of `triangles` (indices into `mesh`'s triangles) from
/// the attribute `name`; none when the name is empty or the mesh has no
/// usable attribute of that name.
pub fn corners(
    mesh: &RenderMesh,
    name: &str,
    triangles: &mut dyn Iterator<Item = usize>,
) -> Option<Arc<[[f32; 3]]>> {
    if name.is_empty() {
        return None;
    }
    let points = mesh.points.len();
    let count = mesh.triangles.len();
    let a = mesh
        .attributes
        .iter()
        .find(|a| &*a.name == name && a.is_consistent(points, count))?;
    let mut out = Vec::new();
    for t in triangles {
        let tri = mesh.triangles.get(t)?;
        for (k, &p) in tri.iter().enumerate() {
            let element = match a.domain {
                AttributeDomain::Point => p as usize,
                AttributeDomain::Corner => t * 3 + k,
                AttributeDomain::Triangle => t,
                AttributeDomain::Constant => 0,
            };
            out.push(color(a, element)?);
        }
    }
    Some(out.into())
}

/// Element `e`'s value as a color.
fn color(a: &MeshAttribute, e: usize) -> Option<[f32; 3]> {
    let n = a.components as usize;
    let v = a.values.get(e * n..e * n + n)?;
    Some(match v {
        [g] => [*g; 3],
        [r, g] => [*r, *g, 0.0],
        [r, g, b, ..] => [*r, *g, *b],
        [] => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::IDENTITY;

    fn mesh(attributes: Vec<MeshAttribute>) -> RenderMesh {
        RenderMesh {
            motion: kinogaki_render_api::Motion::none(),
            points: Arc::from(vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            triangles: Arc::from(vec![[0u32, 1, 2], [0, 2, 3]]),
            normals: None,
            uvs: None,
            material_slots: Arc::from([]),
            materials: Arc::from([]),
            attributes: attributes.into(),
            transform: IDENTITY,
            visible: true,
            id: 1,
            path: Arc::from("/m"),
        }
    }

    fn attribute(domain: AttributeDomain, components: u32, values: Vec<f32>) -> MeshAttribute {
        MeshAttribute {
            name: Arc::from("Cd"),
            domain,
            components,
            color: true,
            integer: false,
            values: values.into(),
        }
    }

    #[test]
    fn each_class_spreads_to_corners() {
        let point = mesh(vec![attribute(
            AttributeDomain::Point,
            3,
            (0..12).map(|v| v as f32).collect(),
        )]);
        let c = corners(&point, "Cd", &mut (0..2)).unwrap();
        // Triangle 1 is points 0, 2, 3.
        assert_eq!(
            c[3..],
            [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        );
        let primitive = mesh(vec![attribute(
            AttributeDomain::Triangle,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )]);
        let c = corners(&primitive, "Cd", &mut (1..2)).unwrap();
        assert_eq!(&*c, [[0.0, 1.0, 0.0]; 3]);
        let corner = mesh(vec![attribute(
            AttributeDomain::Corner,
            1,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )]);
        let c = corners(&corner, "Cd", &mut (0..2)).unwrap();
        assert_eq!(c[4], [0.5; 3]);
        let detail = mesh(vec![attribute(
            AttributeDomain::Constant,
            2,
            vec![0.3, 0.6],
        )]);
        let c = corners(&detail, "Cd", &mut (0..2)).unwrap();
        assert!(c.iter().all(|&v| v == [0.3, 0.6, 0.0]));
    }

    #[test]
    fn missing_empty_or_inconsistent_attributes_give_none() {
        let cd = attribute(AttributeDomain::Point, 3, vec![1.0; 12]);
        assert!(corners(&mesh(vec![cd.clone()]), "", &mut (0..2)).is_none());
        assert!(corners(&mesh(vec![cd]), "tone", &mut (0..2)).is_none());
        let short = attribute(AttributeDomain::Point, 3, vec![1.0; 9]);
        assert!(corners(&mesh(vec![short]), "Cd", &mut (0..2)).is_none());
    }
}
