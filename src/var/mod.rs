mod matrix;
mod scalar;
mod vector;

pub use matrix::MatrixVar;
pub use scalar::ScalarVar;
pub use vector::VectorVar;

use crate::{FloatMatrix, Shape};

pub trait Var {
    fn index(&self) -> usize;
    fn shape(&self) -> Shape;
}

pub struct Grad {
    derivs: Vec<FloatMatrix>,
}

impl Grad {
    pub fn wrt<V>(&self, var: &V) -> &FloatMatrix
    where
        V: Var,
    {
        &self.derivs[var.index()]
    }
}
