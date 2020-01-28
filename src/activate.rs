use crate::op::BinaryOp;
use crate::var::{Binary, Var, Vector};

pub fn sigmoid<S>(x: &Var<Vector, S>) -> Var<Vector, Binary> {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax<S>(x: &Var<Vector, S>) -> Var<Vector, Binary> {
    let exp_x = x.exp();
    &exp_x / exp_x.sum()
}

pub fn relu<S>(x: &Var<Vector, S>) -> Var<Vector, Binary> {
    let zeroes = x.constant_like(0.0);
    zeroes.binary(x, BinaryOp::Max)
}
