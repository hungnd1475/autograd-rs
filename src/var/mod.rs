mod matrix;
mod scalar;
mod vector;

pub use matrix::MatrixVar;
pub use scalar::ScalarVar;
pub use vector::VectorVar;

use crate::Matrix;

pub trait Var {
    fn index(&self) -> usize;
}

pub struct Grad {
    derivs: Vec<Matrix>,
}

impl Grad {
    pub fn wrt<V>(&self, var: &V) -> &Matrix
    where
        V: Var,
    {
        &self.derivs[var.index()]
    }
}
