// The upscale pass: the view-size image, each film pixel repeated over its
// pixel_size × pixel_size block from the view's top-left (film::Film).

@group(0) @binding(1) var beauty_tex: texture_2d<f32>;
@group(0) @binding(2) var view_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if p.x >= frame.view.x || p.y >= frame.view.y {
        return;
    }
    textureStore(view_out, p, textureLoad(beauty_tex, p / frame.film.z, 0));
}
