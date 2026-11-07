//! What the host's device holds, which the renderer checks a scene and a
//! view against before it makes their buffers and textures: wgpu's limits
//! are the host's to choose (Studio asks for wgpu's defaults: 256 MiB a
//! buffer, 128 MiB a storage binding, 8192 texels a texture's side), and a
//! buffer or texture past them is an error wgpu raises as a panic. What the
//! device refuses besides (running out of memory) is caught, not raised
//! ([`Caught`]).
use kinogaki_render_api::display::wgpu;
use std::future::Future;
use std::task::{Context, Poll, Waker};

/// One part of what a scene puts on the device: what it is, its bytes, and
/// whether it is bound whole as storage.
pub struct Part {
    /// What it is, for the message (a plural: "its triangles").
    pub name: &'static str,
    /// Its bytes.
    pub bytes: u64,
    /// Bound whole as storage (else a vertex or index buffer).
    pub bound: bool,
}

impl Part {
    /// `items` as part `name`, bound whole as storage or not.
    pub fn of<T>(name: &'static str, items: &[T], bound: bool) -> Part {
        Part {
            name,
            bytes: std::mem::size_of_val(items) as u64,
            bound,
        }
    }
}

/// What one buffer and one texture of a device hold.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Bytes a buffer holds.
    pub buffer: u64,
    /// Bytes a buffer bound whole as storage holds.
    pub bound: u64,
    /// Texels a texture's side holds.
    pub side: u32,
}

impl Budget {
    /// `device`'s.
    pub fn of(device: &wgpu::Device) -> Budget {
        let limits = device.limits();
        Budget {
            buffer: limits.max_buffer_size,
            bound: limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
            side: limits.max_texture_dimension_2d,
        }
    }

    /// Whether each of `parts` fits a buffer; else why the scene doesn't,
    /// naming the first that doesn't.
    pub fn holds(&self, parts: &[Part]) -> Result<(), String> {
        for part in parts {
            let most = if part.bound { self.bound } else { self.buffer };
            if part.bytes > most {
                return Err(format!(
                    "the scene is too large for Pixel on this GPU: {} take {}, and a buffer holds {}",
                    part.name,
                    mib(part.bytes),
                    mib(most)
                ));
            }
        }
        Ok(())
    }

    /// Whether a view of `size` texels fits a texture; else why not.
    pub fn view(&self, [width, height]: [u32; 2]) -> Result<(), String> {
        match width.max(height) <= self.side {
            true => Ok(()),
            false => Err(format!(
                "the view's {width} × {height} pixels are more than this GPU's textures hold ({} a side)",
                self.side
            )),
        }
    }
}

/// `bytes` in MiB, rounded up.
fn mib(bytes: u64) -> String {
    format!("{} MiB", bytes.div_ceil(1 << 20))
}

/// The device's refusals on this thread from when it is made until it ends:
/// out of memory, or past what wgpu validates (errors it would otherwise
/// raise as a panic). Dropped early, it lets them go.
pub struct Caught {
    // Popped in reverse: the scope pushed last first.
    valid: wgpu::ErrorScopeGuard,
    memory: wgpu::ErrorScopeGuard,
}

impl Caught {
    /// Catching `device`'s refusals from now on.
    pub fn on(device: &wgpu::Device) -> Caught {
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let valid = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Caught { valid, memory }
    }

    /// What the device refused since, if anything.
    pub fn end(self) -> Result<(), String> {
        let refused = [now(self.valid.pop()), now(self.memory.pop())];
        match refused.into_iter().flatten().next() {
            Some(error) => Err(format!("the GPU refused Pixel's work: {error}")),
            None => Ok(()),
        }
    }
}

/// `future`'s output: wgpu's native futures are ready once made.
fn now<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_part_past_its_buffer_says_which_and_by_how_much() {
        let budget = Budget {
            buffer: 256 << 20,
            bound: 128 << 20,
            side: 8192,
        };
        let vertices = Part {
            name: "its vertices",
            bytes: 200 << 20,
            bound: false,
        };
        let triangles = Part {
            name: "its triangles",
            bytes: (200 << 20) + 1,
            bound: true,
        };
        assert_eq!(budget.holds(&[vertices]), Ok(()));
        assert_eq!(
            budget.holds(&[triangles]),
            Err("the scene is too large for Pixel on this GPU: its triangles take 201 MiB, and a buffer holds 128 MiB".into())
        );
        assert_eq!(budget.view([8192, 100]), Ok(()));
        assert!(
            budget
                .view([100, 8193])
                .is_err_and(|why| why.contains("8192 a side"))
        );
    }
}
