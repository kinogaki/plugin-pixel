# Pixel

Pixel is a realtime pixel-art render engine for [Kinogaki Studio](https://github.com/kinogaki), Studio's second render engine. It renders on the host's own wgpu device at a coarse film resolution, lights the film with ray-traced ambient occlusion, hard shadows, one bounce of diffuse light and volumes, quantizes and dithers the light through a Shade Ramp, and repeats each film pixel over its block of the view. Inputs and outputs are linear ACEScg.

It is a plugin on Studio's public SDK (`kinogaki-studio-api`, `kinogaki-render-api` with its `wgpu` feature): the library crate `kinogaki-pixel` holds the engine, and `plugin/` (`kinogaki-plugin-pixel`) is the dylib Studio loads, `renders/pixel.dylib`.

## A frame

A view of `w × h` pixels renders a film of `ceil(w / s) × ceil(h / s)` pixels, `s` the Pixel Size; film pixel `(i, j)` covers view pixels `[i·s, (i+1)·s) × [j·s, (j+1)·s)` from the view's top-left corner, so the grid stays locked to the view. Seven passes:

1. **G-buffer** (raster): albedo (the Pixel Surface's Color × the attribute its Color Attribute names × its texture), normal, depth, object id and the surface's transmittance. A texture reads its picture at each fragment's texture coordinates through its placement, each axis wrapped (repeat, clamp, mirror, black), its nearest texel (`closest`) or four blended, × its factor; where a cutout is set and the texel's alpha (× the vertex alpha a glTF color image reads, the mesh's `Alpha`) is below it, nothing is drawn. Texture prims arrive as the host holds the files, with their colors' decoding into ACEScg: Pixel decodes each texel (its transfer function, then its matrix) and makes premultiplied colors straight as it makes the texture's picture; data comes straight.
2. **Ambient**: the domes' light from above and below the horizon (a dome's ground color), blended by how far the surface faces up, × ambient occlusion from a fixed pattern of hemisphere rays, shaped by AO Strength and AO Power.
3. **Direct**: each light with one hard shadow ray: area lights and spots as points at their centres, distant lights as parallel light; spots fade over their cone; a gobo dapples distant lights and spots. Light reaching a translucent surface's back comes through × its transmittance, and a shadow ray keeps what the thin surfaces it crosses let through. AO on Direct darkens it by the AO.
4. **Bounce**: one diffuse bounce, GI Samples rays a pixel in the same fixed pattern up to GI Distance, each hit lit as the direct pass lights it, × its albedo (a texture's mean for its texels), × GI.
5. **Volume**: each camera ray marched through the Pixel Volumes it crosses (closed meshes found by their crossings) and, when In-scatter is on, the fog, at a fixed jitter a pixel; each step lit by the lights with shadow rays (surfaces cut light shafts), the gobos and the domes × one sky ray.
6. **Shade**: the light (direct + ambient + bounce + the rim lights) quantized by its luminance to Shade Levels with an ordered Bayer dither between adjacent levels, each level its Shade Ramp color in the light's chromaticity, × albedo (an unlit surface's albedo alone); the volumes over it on their own levels (Volume Levels, Volume Dither); then flat height fog.
7. **Upscale**: each film pixel repeated over its block of the view.

The rays of passes 2 to 5 (AO, shadows, bounces, the volumes' shadow and sky rays) pass where a cutout cuts a surface away: at the texture coordinates and alpha the corners of the triangle a ray meets blend to there, the G-buffer's own test.

Every pass is deterministic: a frame of the same scene and view is the same image. A view renders once when something it shows changes, then rests.

## Materials, lights and fog

- **Pixel Surface**: Color (a texture node may drive it), Color Attribute (`Cd`), Translucency, Translucency Color, Cutout (cut out where the color's texture is less opaque than this). **Pixel Volume**: Density, Scatter Color, Anisotropy, Edge Falloff, Shadows, Emission and Noise. Pixel has no node graph: each node is a material on its own. The shaded viewport shows a surface's color (× its attribute) and a volume as a ghost of its bounds.
- **MaterialX**: Pixel renders Surface Unlit (and Kinogaki's Surface Unlit Cutout, cut out at its cutoff in its mask mode) as it is, an unlit surface: Emission × Emission Color whatever lights it (a bounce ray meeting it brings that light back), Transmission × Transmission Color letting light through. Any other material arrives as its standard form, an OpenPBR Surface, as a Pixel Surface: light alone (no base, no specular) a Surface Unlit of its emission; else a Pixel Surface of its base color, of the texture node driving it (cut out at the cutoff a glTF mask compares its alpha with), or of white times the attribute a geometric color or property drives it with. What else drives an input it doesn't render, and says so once. Another engine renders a Pixel Surface as its standard form, an OpenPBR Surface of its color, its texture driving the base color (a Pixel Volume has none).
- **Textures**: Pixel samples MaterialX's `image` and glTF's texture nodes (`gltf_image`, `gltf_colorimage`, and Kinogaki's with glTF's minification filter) itself, reading their coordinate set (`texcoord`, `geompropvalue`), placement (`place2d`; a glTF texture's offset, rotation and scale, by the graphs MaterialX defines them with), each axis's address mode, filter (`closest` the nearest texel, else bilinear) and factors, and glTF's vertex colors (`geomcolor`) as the tint. The host sends each image as a Texture prim; Pixel keeps it as a picture, its own form (straight floats of linear ACEScg, its longer side at most 2048 texels: a larger image reduced by the least power of two that fits, each texel the mean of those it covers, made a row at a time on the process's workers from the image's tiles) packed with the others into one half-float atlas 2048 texels wide and at most 8192 tall (the largest halved until they fit: four 2048 × 2048 pictures fit). The minification filter is kept but not read: the film is coarse and samples the picture at its own texels.
- **Lights** (`kinogaki.pixel.light`): UsdLux's area, dome, distant and spot lights (distant lights and spots with a Gobo section), and the **Rim Light**, a Fresnel-like edge, `(1 − n·v)⁴` × its emission, where direct light reaches (Lit) or all round (All).
- **Fog** (`kinogaki.pixel.fog`): exponential height fog, flat, or lit and marched with the volumes when In-scatter is on.
- Starting materials: Grass, Stone, Clay, Fog Volume, Mist.

## Outputs

Beauty, Albedo, Direct Light, Ambient Light, Indirect Light, Volume, Ambient Occlusion, Depth, Normal and Object ID, each a film-size texture; a final render reads them at the view's size. A compositing host takes the film (`Display::acquire_film`): one Image of the beauty and the outputs it asks for (kept from a later frame on), each layer held as its texture, its metadata saying it is pixel art.

## Code

| Module | Role |
|---|---|
| `render` | The renderer, knowing no SDK: the scene (meshes in their own space, a vertex each distinct corner, and the instances placing them), `film`, `camera`, `bvh` (binned SAH in two levels: a BVH a mesh, built when it is made, and a top level over the instances, built when one moves), `dither`, `settings` |
| `gpu` | The passes on the host's device (the G-buffer draws each mesh once for its instances; rays walk the top level, then each instance's mesh in its own space), the atlas of pictures the G-buffer and the rays sample, their records and shaders (`gpu/shaders`). A mesh its material cuts out also keeps its traced triangles' corners' texture coordinates and alpha for the rays (36 bytes a triangle; other meshes none). Each mesh keeps its runs of the meshes' buffers: a mesh made again that fits them (a deform's) is written there alone, and only a mesh added, gone or outgrowing its runs lays them out again (a 128-triangle shape's points moved beside 1.6 million triangles: median 2.4–5.0 ms to the GPU, 58–59 ms when every mesh was laid out again; `tests/gpu/moves.rs`, by hand). A scene or a film larger than the device's buffers and textures hold (wgpu's limits as the host asked for them) is never made: every view it touches fails saying why, and renders again once it fits; what the device refuses besides is caught the same way |
| `description`, `settings`, `outputs` | What a host reads: capabilities, settings, outputs |
| `nodes`, `presets` | Pixel Surface and Pixel Volume, what a network renders as (Surface Unlit too), their display approximation, standard form and translation; the starting materials |
| `sampled` | The texture nodes Pixel samples, read: coordinates, placement, wraps, filter, factors |
| `lights` | The light and fog types and kinds; emissions as colors |
| `sync` | The host's prims as the scene: each shape's meshes by material (an instance's own material or material map giving its own, once each), kept while its dirty bits say only its placement, visibility or name changed, while it is hidden, and while no material it reads changes kind, tint, alpha, coordinates or whether it cuts out; every placement an instance, its own primvars giving its object id and the color and alpha its material reads; volumes, lights |
| `pictures` | Texture prims as the pictures surfaces sample |
| `engine` | The render thread, the display images, final reads and previews; an update renders again the views whose scene or camera it changed |

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
