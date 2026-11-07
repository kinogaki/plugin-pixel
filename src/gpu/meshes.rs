//! The meshes side by side in the scene's buffers ([`Meshes`]): each mesh
//! keeps its runs of them, as Spectre's packed buffers give prims theirs,
//! so a mesh made again that fits its runs (a deform's) is written there
//! alone ([`Meshes::rewrite`]), and only a mesh added, gone or outgrowing
//! its runs lays them all out again.
use super::budget::Part;
use super::records::Vertex;
use crate::render::Mesh;
use crate::render::bvh::{Node, Triangle};
use std::ops::Range;
use std::sync::Arc;

/// Where a mesh lies in the meshes' buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    /// Its triangles' run of indices.
    pub indices: Range<u32>,
    /// Its first vertex, which its indices count from.
    pub base: u32,
    /// Its BVH's root node.
    pub root: u32,
}

/// The meshes side by side: their vertices and triangles, and their BVHs in
/// one array of nodes and one of triangles, each mesh's indices moved past
/// the meshes before it. Each mesh holds a run of each array as long as it
/// was laid out with ([`Room`]): what it uses of them is its [`Span`].
pub struct Meshes {
    /// Each mesh's vertices, in a run.
    pub vertices: Vec<Vertex>,
    /// Three vertices a triangle, each mesh's counted from its first.
    pub indices: Vec<u32>,
    /// Where each mesh lies.
    pub meshes: Vec<Span>,
    /// The BVHs' nodes.
    pub nodes: Vec<Node>,
    /// Their triangles, each with its corners' mean tint in its `w`s.
    pub triangles: Vec<Triangle>,
    /// Each mesh's runs.
    room: Vec<Room>,
    /// The meshes laid out, by index.
    held: Vec<Arc<Mesh>>,
}

/// Where a mesh's runs start in each array, and how long they are.
#[derive(Clone, Copy, Debug)]
struct Room {
    /// Its first vertex, index, node and triangle.
    at: [usize; 4],
    /// The vertices, indices, nodes and triangles its runs hold.
    len: [usize; 4],
}

/// A mesh's vertices, indices, nodes and triangles.
fn lengths(mesh: &Mesh) -> [usize; 4] {
    let vertices = mesh.positions.len();
    [
        vertices,
        mesh.triangles.len() * 3,
        mesh.bvh.nodes.len(),
        mesh.bvh.triangles.len(),
    ]
}

/// What one array's runs written in place cover, in its elements.
pub type Written = [Vec<Range<usize>>; 4];

impl Meshes {
    /// `meshes` side by side.
    pub fn build(meshes: &[Arc<Mesh>]) -> Meshes {
        let mut made = Meshes {
            vertices: Vec::new(),
            indices: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            triangles: Vec::new(),
            room: Vec::new(),
            held: meshes.to_vec(),
        };
        for mesh in meshes {
            let len = lengths(mesh);
            let at = [
                made.vertices.len(),
                made.indices.len(),
                made.nodes.len(),
                made.triangles.len(),
            ];
            made.vertices.resize(at[0] + len[0], Vertex::default());
            made.indices.resize(at[1] + len[1], 0);
            made.nodes.resize(at[2] + len[2], Node::default());
            made.triangles.resize(at[3] + len[3], Triangle::default());
            let room = Room { at, len };
            let span = made.write(mesh, room);
            made.meshes.push(span);
            made.room.push(room);
        }
        made
    }

    /// `meshes` in place of the meshes laid out, when they are as many and
    /// each is the one laid out at its index or fits its runs: those made
    /// again written there, and the runs written. None (nothing changed)
    /// when they aren't: they are laid out again.
    pub fn rewrite(&mut self, meshes: &[Arc<Mesh>]) -> Option<Written> {
        let fits = |(k, mesh): (usize, &Arc<Mesh>)| {
            Arc::ptr_eq(mesh, &self.held[k])
                || (lengths(mesh).iter().zip(self.room[k].len)).all(|(n, room)| *n <= room)
        };
        if meshes.len() != self.held.len() || !meshes.iter().enumerate().all(fits) {
            return None;
        }
        let mut written: Written = Default::default();
        for (k, mesh) in meshes.iter().enumerate() {
            if Arc::ptr_eq(mesh, &self.held[k]) {
                continue;
            }
            let room = self.room[k];
            self.meshes[k] = self.write(mesh, room);
            for (array, (at, n)) in written.iter_mut().zip(room.at.iter().zip(lengths(mesh))) {
                array.push(*at..at + n);
            }
            self.held[k] = mesh.clone();
        }
        Some(written)
    }

    /// `mesh` written at the start of its runs `room`: where it lies.
    fn write(&mut self, mesh: &Mesh, room: Room) -> Span {
        let tint = |v: u32| mesh.tints.as_ref().map_or([1.0; 4], |c| c[v as usize]);
        let [vertices, indices, nodes, triangles] = room.at;
        let vertex = (0..mesh.positions.len()).map(|v| Vertex {
            position: mesh.positions[v],
            normal: mesh.normals[v],
            tint: tint(v as u32),
            uv: mesh.uvs.as_ref().map_or([0.0; 2], |c| c[v]),
        });
        for (to, v) in self.vertices[vertices..].iter_mut().zip(vertex) {
            *to = v;
        }
        let corners = mesh.triangles.iter().flatten();
        for (to, &i) in self.indices[indices..].iter_mut().zip(corners) {
            *to = i;
        }
        let node = mesh.bvh.nodes.iter().map(|n| Node {
            a: n.a + if n.count == 0 { nodes } else { triangles } as u32,
            ..*n
        });
        for (to, n) in self.nodes[nodes..].iter_mut().zip(node) {
            *to = n;
        }
        let tinted = (mesh.bvh.triangles.iter())
            .zip(&mesh.bvh.order)
            .map(|(tri, &t)| {
                let corners = mesh.triangles[t as usize].map(tint);
                let [r, g, b] =
                    std::array::from_fn(|k| corners.iter().map(|c| c[k]).sum::<f32>() / 3.0);
                let mut tri = *tri;
                (tri.v0[3], tri.e1[3], tri.e2[3]) = (r, g, b);
                tri
            });
        for (to, t) in self.triangles[triangles..].iter_mut().zip(tinted) {
            *to = t;
        }
        let first = indices as u32;
        Span {
            indices: first..first + (mesh.triangles.len() * 3) as u32,
            base: vertices as u32,
            root: nodes as u32,
        }
    }

    /// What was written in place never reached the GPU: the next meshes
    /// are laid out whole.
    pub fn unsent(&mut self) {
        self.held.clear();
    }

    /// What they put on the device.
    pub fn parts(&self) -> [Part; 4] {
        [
            Part::of("its vertices", &self.vertices, false),
            Part::of("its vertex indices", &self.indices, false),
            Part::of("its hierarchy's nodes", &self.nodes, true),
            Part::of("its traced triangles", &self.triangles, true),
        ]
    }
}
