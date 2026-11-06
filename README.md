# Pixel

Pixel is a realtime pixel-art render engine for [Kinogaki Studio](https://github.com/kinogaki), Studio's second render engine. It renders on the host's own wgpu device at a coarse film resolution, lights the film with ray-traced ambient occlusion, hard shadows, one bounce of diffuse light and volumes, quantizes and dithers the light through a Shade Ramp, and repeats each film pixel over its block of the view. Inputs and outputs are linear ACEScg.

It is a plugin on Studio's public SDK (`kinogaki-studio-api`, `kinogaki-render-api` with its `wgpu` feature): the library crate `kinogaki-pixel` holds the engine, and `plugin/` (`kinogaki-plugin-pixel`) is the dylib Studio loads, `renders/pixel.dylib`.

## A frame

A view of `w × h` pixels renders a film of `ceil(w / s) × ceil(h / s)` pixels, `s` the Pixel Size; film pixel `(i, j)` covers view pixels `[i·s, (i+1)·s) × [j·s, (j+1)·s)` from the view's top-left corner, so the grid stays locked to the view. Seven passes:

1. **G-buffer** (raster): albedo (the Pixel Surface's Color × the attribute its Color Attribute names), normal, depth, object id and the surface's transmittance.
2. **Ambient**: the domes' light from above and below the horizon (a dome's ground color), blended by how far the surface faces up, × ambient occlusion from a fixed pattern of hemisphere rays, shaped by AO Strength and AO Power.
3. **Direct**: each light with one hard shadow ray: area lights and spots as points at their centres, distant lights as parallel light; spots fade over their cone; a gobo dapples distant lights and spots. Light reaching a translucent surface's back comes through × its transmittance, and a shadow ray keeps what the thin surfaces it crosses let through. AO on Direct darkens it by the AO.
4. **Bounce**: one diffuse bounce, GI Samples rays a pixel in the same fixed pattern up to GI Distance, each hit lit as the direct pass lights it, × its albedo, × GI.
5. **Volume**: each camera ray marched through the Pixel Volumes it crosses (closed meshes found by their crossings) and, when In-scatter is on, the fog, at a fixed jitter a pixel; each step lit by the lights with shadow rays (surfaces cut light shafts), the gobos and the domes × one sky ray.
6. **Shade**: the light (direct + ambient + bounce + the rim lights) quantized by its luminance to Shade Levels with an ordered Bayer dither between adjacent levels, each level its Shade Ramp color in the light's chromaticity, × albedo; the volumes over it on their own levels (Volume Levels, Volume Dither); then flat height fog.
7. **Upscale**: each film pixel repeated over its block of the view.

Every pass is deterministic: a frame of the same scene and view is the same image. A view renders once when something it shows changes, then rests.

## Materials, lights and fog

- **Pixel Surface**: Color, Color Attribute (`Cd`), Translucency, Translucency Color. **Pixel Volume**: Density, Scatter Color, Anisotropy, Edge Falloff, Shadows, Emission and Noise. Pixel has no node graph: each node is a material on its own. The shaded viewport shows a surface's color (× its attribute) and a volume as a ghost of its bounds; another engine's material renders as a Pixel Surface of its color.
- **Lights** (`kinogaki.pixel.light`): UsdLux's area, dome, distant and spot lights (distant lights and spots with a Gobo section), and the **Rim Light**, a Fresnel-like edge, `(1 − n·v)⁴` × its emission, where direct light reaches (Lit) or all round (All).
- **Fog** (`kinogaki.pixel.fog`): exponential height fog, flat, or lit and marched with the volumes when In-scatter is on.
- Starting materials: Grass, Stone, Clay, Fog Volume, Mist.

## Outputs

Beauty, Albedo, Direct Light, Ambient Light, Indirect Light, Volume, Ambient Occlusion, Depth, Normal and Object ID, each a film-size texture; a final render reads them at the view's size. A compositing host takes the film (`Display::acquire_film`): one Image of the beauty and the outputs it asks for (kept from a later frame on), each layer held as its texture, its metadata saying it is pixel art.

## Code

| Module | Role |
|---|---|
| `render` | The renderer, knowing no SDK: the scene (meshes in their own space and the instances placing them), `film`, `camera`, `bvh` (binned SAH in two levels: a BVH a mesh, built when it is made, and a top level over the instances, built when one moves), `dither`, `settings` |
| `gpu` | The passes on the host's device (the G-buffer draws each mesh once for its instances; rays walk the top level, then each instance's mesh in its own space), their records and shaders (`gpu/shaders`) |
| `description`, `settings`, `outputs` | What a host reads: capabilities, settings, outputs |
| `nodes`, `presets` | Pixel Surface and Pixel Volume, their display approximation and translation; the starting materials |
| `lights` | The light and fog types and kinds; emissions as colors |
| `sync` | The host's prims as the scene: each shape's meshes by material, kept while its dirty bits say only its placement, visibility or name changed; every placement an instance; volumes, lights |
| `engine` | The render thread, the display images, final reads and previews |

## Building

The engine builds from this repository and the public [studio-sdk](https://github.com/kinogaki/studio-sdk) alone, checked out side by side as Studio's tree lays them out (`sdk/` and `plugins/pixel/`), with the compiler the SDK pins (`sdk/rust-toolchain.toml`):

```sh
git clone https://github.com/kinogaki/studio-sdk sdk
git clone https://github.com/kinogaki/plugin-pixel plugins/pixel
cd plugins/pixel
cargo test                           # the scene sync
cargo test --test gpu -- --ignored --skip hand_run_   # the GPU tests (a native graphics adapter)
```

`plugin/` is the library Studio loads. It links Studio's shared library (`kinogaki_shared`), so it is built in Studio's tree, or against a Studio release's plugin kit as the SDK's guides describe; Studio's tree also runs Pixel's example render through the editor (`workflows/`); Pixel's own GPU tests run the SDK's engine conformance.

## License

Pixel is dual-licensed under the MIT license ([LICENSE-MIT](LICENSE-MIT)) and the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), at your option.
