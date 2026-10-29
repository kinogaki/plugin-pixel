//! The vectors and matrices Pixel computes with: [`Vec3`], [`Vec4`] and
//! [`Mat4`] in f64, and [`Vec3f`] in f32 for the GPU boundary.
//!
//! Every operation is plain IEEE arithmetic in a fixed order (no fused
//! multiply-add, no reassociation, no SIMD that would change rounding), so
//! the same inputs give the same bits on every OS and CPU. Matrices are
//! column-major and `m * v` treats `v` as a column vector, so `a * b`
//! applies `b` first, then `a`.

use core::ops::{Add, Div, Index, Mul, Neg, Sub};

#[inline(always)]
fn min_of<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

#[inline(always)]
fn max_of<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// A 3-component vector of f64.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    /// The x component.
    pub x: f64,
    /// The y component.
    pub y: f64,
    /// The z component.
    pub z: f64,
}

impl Vec3 {
    /// Every component zero.
    pub const ZERO: Self = Self::splat(0.0);

    /// The unit x axis.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);

    /// The unit y axis.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);

    /// The unit z axis.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    /// The vector of these components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Every component `v`.
    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// The dot product, summed left to right.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    /// The Euclidean length `sqrt(v · v)`.
    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// `self × (1 / sqrt(self · self))`, or `fallback` when that
    /// reciprocal is zero or not finite: a zero, non-finite, or so tiny or
    /// huge a vector that its square underflows or overflows.
    #[inline]
    pub fn normalize_or(self, fallback: Self) -> Self {
        let rcp = 1.0 / self.dot(self).sqrt();
        if rcp.is_finite() && rcp > 0.0 {
            self * rcp
        } else {
            fallback
        }
    }

    /// The cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - rhs.y * self.z,
            y: self.z * rhs.x - rhs.z * self.x,
            z: self.x * rhs.y - rhs.x * self.y,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    #[inline]
    #[track_caller]
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}

/// A 4-component vector of f64.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    /// The x component.
    pub x: f64,
    /// The y component.
    pub y: f64,
    /// The z component.
    pub z: f64,
    /// The w component.
    pub w: f64,
}

impl Vec4 {
    /// The unit x axis.
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);

    /// The unit y axis.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);

    /// The unit w axis.
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// The vector of these components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// The first three components.
    #[inline]
    pub const fn truncate(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl Mul for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
            w: self.w * rhs.w,
        }
    }
}

impl Mul<f64> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl Index<usize> for Vec4 {
    type Output = f64;
    #[inline]
    #[track_caller]
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index {i} out of range"),
        }
    }
}

/// A 3-component vector of f32.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3f {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3f {
    /// Every component one.
    pub const ONE: Self = Self::splat(1.0);

    /// The vector of these components.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Every component `v`.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// The vector of an array's components, in order.
    #[inline]
    pub const fn from_array(a: [f32; 3]) -> Self {
        let [x, y, z] = a;
        Self { x, y, z }
    }

    /// The components as an array, in order.
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// The dot product, summed left to right.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    /// Per component `self < rhs ? self : rhs`: `rhs` when they compare
    /// equal (so `min(0, -0)` is `-0`) or either is NaN.
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: min_of(self.x, rhs.x),
            y: min_of(self.y, rhs.y),
            z: min_of(self.z, rhs.z),
        }
    }

    /// Per component `self > rhs ? self : rhs`: `rhs` when they compare
    /// equal or either is NaN.
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: max_of(self.x, rhs.x),
            y: max_of(self.y, rhs.y),
            z: max_of(self.z, rhs.z),
        }
    }

    /// The smallest component (`a < b ? a : b`, folded from the right).
    #[inline]
    pub fn min_element(self) -> f32 {
        min_of(self.x, min_of(self.y, self.z))
    }

    /// The largest component (`a > b ? a : b`, folded from the right).
    #[inline]
    pub fn max_element(self) -> f32 {
        max_of(self.x, max_of(self.y, self.z))
    }

    /// The cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - rhs.y * self.z,
            y: self.z * rhs.x - rhs.z * self.x,
            z: self.x * rhs.y - rhs.x * self.y,
        }
    }
}

impl Add for Vec3f {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3f {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul for Vec3f {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl Mul<f32> for Vec3f {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div for Vec3f {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl Index<usize> for Vec3f {
    type Output = f32;
    #[inline]
    #[track_caller]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3f index {i} out of range"),
        }
    }
}

/// A 4×4 matrix of f64, stored as four columns.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    /// The first column.
    pub x_axis: Vec4,
    /// The second column.
    pub y_axis: Vec4,
    /// The third column.
    pub z_axis: Vec4,
    /// The fourth column (the translation, for an affine matrix).
    pub w_axis: Vec4,
}

impl Mat4 {
    /// The matrix of these columns.
    #[inline]
    pub const fn from_cols(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    /// The rotation by `angle` radians about x, counter-clockwise.
    #[inline]
    pub fn from_rotation_x(angle: f64) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            Vec4::X,
            Vec4::new(0.0, cosa, sina, 0.0),
            Vec4::new(0.0, -sina, cosa, 0.0),
            Vec4::W,
        )
    }

    /// The rotation by `angle` radians about y, counter-clockwise.
    #[inline]
    pub fn from_rotation_y(angle: f64) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::from_cols(
            Vec4::new(cosa, 0.0, -sina, 0.0),
            Vec4::Y,
            Vec4::new(sina, 0.0, cosa, 0.0),
            Vec4::W,
        )
    }

    /// Column `index` (0 to 3). Panics on any other index.
    #[inline]
    #[track_caller]
    pub fn col(&self, index: usize) -> Vec4 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            3 => self.w_axis,
            _ => panic!("Mat4 column {index} out of range"),
        }
    }

    /// Row `index` (0 to 3). Panics on any other index.
    #[inline]
    #[track_caller]
    fn row(&self, index: usize) -> Vec4 {
        Vec4::new(
            self.x_axis[index],
            self.y_axis[index],
            self.z_axis[index],
            self.w_axis[index],
        )
    }

    /// The transpose.
    #[inline]
    pub fn transpose(&self) -> Self {
        Self::from_cols(self.row(0), self.row(1), self.row(2), self.row(3))
    }

    /// The inverse by cofactors (GLM's layout) times one reciprocal of the
    /// determinant: infinite or NaN entries for a singular matrix (no check).
    pub fn inverse(&self) -> Self {
        let Vec4 {
            x: m00,
            y: m01,
            z: m02,
            w: m03,
        } = self.x_axis;
        let Vec4 {
            x: m10,
            y: m11,
            z: m12,
            w: m13,
        } = self.y_axis;
        let Vec4 {
            x: m20,
            y: m21,
            z: m22,
            w: m23,
        } = self.z_axis;
        let Vec4 {
            x: m30,
            y: m31,
            z: m32,
            w: m33,
        } = self.w_axis;

        let coef00 = m22 * m33 - m32 * m23;
        let coef02 = m12 * m33 - m32 * m13;
        let coef03 = m12 * m23 - m22 * m13;

        let coef04 = m21 * m33 - m31 * m23;
        let coef06 = m11 * m33 - m31 * m13;
        let coef07 = m11 * m23 - m21 * m13;

        let coef08 = m21 * m32 - m31 * m22;
        let coef10 = m11 * m32 - m31 * m12;
        let coef11 = m11 * m22 - m21 * m12;

        let coef12 = m20 * m33 - m30 * m23;
        let coef14 = m10 * m33 - m30 * m13;
        let coef15 = m10 * m23 - m20 * m13;

        let coef16 = m20 * m32 - m30 * m22;
        let coef18 = m10 * m32 - m30 * m12;
        let coef19 = m10 * m22 - m20 * m12;

        let coef20 = m20 * m31 - m30 * m21;
        let coef22 = m10 * m31 - m30 * m11;
        let coef23 = m10 * m21 - m20 * m11;

        let fac0 = Vec4::new(coef00, coef00, coef02, coef03);
        let fac1 = Vec4::new(coef04, coef04, coef06, coef07);
        let fac2 = Vec4::new(coef08, coef08, coef10, coef11);
        let fac3 = Vec4::new(coef12, coef12, coef14, coef15);
        let fac4 = Vec4::new(coef16, coef16, coef18, coef19);
        let fac5 = Vec4::new(coef20, coef20, coef22, coef23);

        let vec0 = Vec4::new(m10, m00, m00, m00);
        let vec1 = Vec4::new(m11, m01, m01, m01);
        let vec2 = Vec4::new(m12, m02, m02, m02);
        let vec3 = Vec4::new(m13, m03, m03, m03);

        let inv0 = vec1 * fac0 - vec2 * fac1 + vec3 * fac2;
        let inv1 = vec0 * fac0 - vec2 * fac3 + vec3 * fac4;
        let inv2 = vec0 * fac1 - vec1 * fac3 + vec3 * fac5;
        let inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

        let sign_a = Vec4::new(1.0, -1.0, 1.0, -1.0);
        let sign_b = Vec4::new(-1.0, 1.0, -1.0, 1.0);

        let inverse = Self::from_cols(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);

        let col0 = Vec4::new(
            inverse.x_axis.x,
            inverse.y_axis.x,
            inverse.z_axis.x,
            inverse.w_axis.x,
        );
        let dot0 = self.x_axis * col0;
        let dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
        inverse * (1.0 / dot1)
    }

    /// The point `v` (w = 1) through an affine matrix (last row 0 0 0 1).
    #[inline]
    pub fn transform_point3(&self, v: Vec3) -> Vec3 {
        let mut res = self.x_axis * v.x;
        res = self.y_axis * v.y + res;
        res = self.z_axis * v.z + res;
        res = self.w_axis + res;
        res.truncate()
    }

    /// The direction `v` (w = 0) through an affine matrix: no translation.
    #[inline]
    pub fn transform_vector3(&self, v: Vec3) -> Vec3 {
        let mut res = self.x_axis * v.x;
        res = self.y_axis * v.y + res;
        res = self.z_axis * v.z + res;
        res.truncate()
    }

    /// `x_axis · v.x + y_axis · v.y + z_axis · v.z + w_axis · v.w`, summed
    /// left to right.
    #[inline]
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z + self.w_axis * v.w
    }
}

impl Mul<f64> for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::from_cols(
            self.x_axis * rhs,
            self.y_axis * rhs,
            self.z_axis * rhs,
            self.w_axis * rhs,
        )
    }
}

/// `self * rhs` applies `rhs` first, then `self`.
impl Mul for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(
            self.mul_vec4(rhs.x_axis),
            self.mul_vec4(rhs.y_axis),
            self.mul_vec4(rhs.z_axis),
            self.mul_vec4(rhs.w_axis),
        )
    }
}
