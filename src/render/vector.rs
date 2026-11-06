//! The single-precision vector the renderer computes with on the CPU, as the
//! GPU does: plain IEEE arithmetic in a fixed order.
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

/// Three floats: a point, a direction or a color.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    /// X.
    pub x: f32,
    /// Y.
    pub y: f32,
    /// Z.
    pub z: f32,
}

impl V3 {
    /// All zeros.
    pub const ZERO: V3 = V3::splat(0.0);
    /// All ones.
    pub const ONE: V3 = V3::splat(1.0);

    /// (`x`, `y`, `z`).
    pub const fn new(x: f32, y: f32, z: f32) -> V3 {
        V3 { x, y, z }
    }

    /// `v` in every component.
    pub const fn splat(v: f32) -> V3 {
        V3::new(v, v, v)
    }

    /// From an array.
    pub const fn of([x, y, z]: [f32; 3]) -> V3 {
        V3::new(x, y, z)
    }

    /// As an array.
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// As four floats, `w` last: what a GPU record holds.
    pub const fn with_w(self, w: f32) -> [f32; 4] {
        [self.x, self.y, self.z, w]
    }

    /// The dot product.
    pub fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// The cross product.
    pub fn cross(self, o: V3) -> V3 {
        V3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Its length.
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// It at unit length; `fallback` when it has none.
    pub fn unit_or(self, fallback: V3) -> V3 {
        let l = self.length();
        if l > 1e-12 && l.is_finite() {
            self * (1.0 / l)
        } else {
            fallback
        }
    }

    /// The smaller of each component.
    pub fn min(self, o: V3) -> V3 {
        V3::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    /// The larger of each component.
    pub fn max(self, o: V3) -> V3 {
        V3::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    /// Its smallest component.
    pub fn min_element(self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    /// Its largest component.
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul for V3 {
    type Output = V3;
    fn mul(self, o: V3) -> V3 {
        V3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}

impl Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        V3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Div for V3 {
    type Output = V3;
    fn div(self, o: V3) -> V3 {
        V3::new(self.x / o.x, self.y / o.y, self.z / o.z)
    }
}

impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        V3::new(-self.x, -self.y, -self.z)
    }
}

impl Index<usize> for V3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}
