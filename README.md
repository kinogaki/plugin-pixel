# Pixel plugin (frozen)

Pixel is a realtime pixel-art render engine for [Kinogaki Studio](https://github.com/kinogaki). It renders on the host's own wgpu device (wgpu 30.0.1) at a coarse film resolution, lights it with ray-traced occlusion, shadows, one bounce and volumes, quantizes and dithers the light through a shade ramp, and upscales the result with nearest sampling to the view.

Frozen in the Kinogaki Studio tree: outside the build, never built, edited or checked there. Pixel is Studio's second render engine, public and open source; it comes back written fresh on the current SDK as its own plugin, and this code stays only as source to take from (the root's `docs/PLUGINS.md`, Frozen plugins). What follows describes the code as it was frozen.

It is written only against the render interface, `kinogaki-render-api` from [studio-sdk](https://github.com/kinogaki/studio-sdk) (with its `wgpu` feature for the display contract), and depends on nothing else from Kinogaki. Its other dependencies are `wgpu` and `bytemuck` from crates.io.

A frame renders at `ceil(view / pixel_size)` and runs seven GPU passes: G-buffer (raster; albedo is the Pixel Surface's Color times the mesh's color attribute where it has one, and the transmittance its Translucency gives), ambient light (the domes', a dome's ground color below the horizon; no dome, none) × ray-traced AO shaped by AO Strength and Power, direct light from area, distant and spot lights with ray-traced hard shadows, optional gobos and light through translucent back faces (shadow rays keep the product of the transmittances of the thin translucent surfaces they cross, so light dapples through foliage; a mesh's shading normals, authored ones included, set the cosine; a shadow ray first steps a little toward its light, so a thin card shaded by a normal leaning past its own plane, such as Crown Normal's foliage, does not shadow itself) (darkened by AO as AO on Direct asks), one bounce of diffuse light (GI: a fixed pattern of rays a pixel, each hit lit as the direct pass lights, × its albedo), all ray traced against a BVH built on the CPU when geometry changes, volume (each camera ray marched through Pixel Volumes, closed meshes found by their crossings in a small BVH of their own and never drawn, and through the scene's Fog prim when In-scatter lights it: a fixed jitter a pixel, each step lit by the lights with shadow rays, so surfaces cut light shafts, gobos and the ambient light × one sky ray; written as light and opacity, premultiplied), shade (rim lights, quantize and dither through the Shade Ramp, the volumes over it, `L_surface · T + L` with the opacity and the media's light (through the ramp) on Volume Levels, dithered at Volume Dither so only the bands' edges dither, then unlit height fog on the same bands), and a nearest upscale to the view. Cameras are perspective or orthographic. Inputs and outputs are linear ACEScg.

| Crate | Role |
|---|---|
| `kinogaki-pixel-render` (`crates/pixel`) | No device: the engine description (`description()`: settings, the prim types it declares, the Pixel Surface and Pixel Volume vocabulary, presets, outputs), the scene store the render interface's prims make, film, camera, BVH, geometry and dither |
| `kinogaki-pixel-render-wgpu` (`crates/wgpu`) | The passes (`Renderer`, a `Viewport` a view), the `RenderEngine` on its render thread, its `Display` and `Pixel` (a `kinogaki_render_api::display::Engine`) |

| Module (`kinogaki-pixel-render`) | Role |
|---|---|
| `description`, `material`, `settings` | What hosts read: the engine, Pixel Surface, Pixel Volume and Pixel Render Settings (Pixels, Ambient Occlusion, Global Illumination, Volumes; the Shade Ramp). Light comes from the scene: the render interface's standard lights and fog, which it declares, and its own Rim Light type (`kinogaki.pixel.rim_light`) |
| `volume`, `media` | The Pixel Volume's medium (density, scatter color, anisotropy, emission, noise, edge falloff, shadows), its density field and phase as the GPU mirrors them; the volumes' records and BVH |
| `store` | Prims to the scene (emitters' blackbodies and CIE illuminants as ACEScg colors), saying what changed (only geometry rebuilds the BVH; triangles naming a Pixel Volume become a volume); domes add up into the ambient light, rim lights into the rim by side, the first Fog prim is the fog |
| `tint` | A mesh's color attribute (`Cd`, or the name a Pixel Surface's Color Attribute gives) as corner tints its Color is multiplied by: point and vertex values blend across a triangle, primitive values are flat |
| `scene` | Meshes, volumes, Pixel Surface materials, lights, the dome; the camera is the render interface's |
| `lights`, `gobo`, `rim` | Lights packed for the direct and bounce passes (each kind's falloff and brightness); the gobo (`kinogaki.pixel.gobo`, applied to distant lights and spots) and its noise; the rim light type |
| `hemisphere` | The fixed cosine-weighted ray pattern AO and GI share |
| `shading` | The lighting's and shade pass's CPU definitions: AO's shaping, the Shade Ramp's levels, rim, volumes over surfaces and fog |
| `film`, `camera` | Film sizing, the view ↔ film mapping, the frustum or box that locks the grid to the view |
| `math`, `bvh`, `geometry` | The vectors and matrices it computes with (plain IEEE arithmetic in a fixed order); the binned-SAH BVH in the GPU's layout; meshes flattened to world space, and each BVH triangle's surface (object, mean tint) for bounce hits |
| `dither`, `aov` | Bayer matrices and the shade function the GPU mirrors; the outputs |

| Module (`kinogaki-pixel-render-wgpu`) | Role |
|---|---|
| `gpu` | The renderer: buffers, a view's targets and images, the passes, read-back and timestamps |
| `engine`, `worker` | The engine's handle and its render thread |
| `ring`, `display` | A view's three display images, pinned while a host shows one, each with film-size copies of its frame's beauty and the outputs a compositing host asked for (`Display::acquire_film`) |
| `outputs` | Final frames' outputs at the view's size |

## Building

A plain clone builds with a stable Rust toolchain (1.98 or later); Cargo fetches studio-sdk at the revision `Cargo.toml` pins and the rest from crates.io.

```sh
cargo build --workspace
```

## Checks

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo test --workspace --locked -- --ignored     # GPU tests; PNGs go to PIXEL_RENDER_OUT (default: ../out beside the checkout)
cargo test --release --test timing -- --ignored --nocapture
```

The GPU tests need an adapter (Metal on macOS); they render small scenes and check them against CPU references.
