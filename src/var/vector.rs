use super::{ScalarVar, Var};
use crate::op::{BinaryOp, UnaryOp};
use crate::{FloatVector, Node, Shape, ShapeExt, Tape};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy)]
/// Represents a real-valued variable.
pub struct VectorVar<'t> {
    tape: &'t Tape,
    shape: Shape,
    index: usize,
}

impl<'t> Var for VectorVar<'t> {
    fn index(&self) -> usize {
        self.index
    }

    fn shape(&self) -> Shape {
        self.shape
    }
}

impl<'t> VectorVar<'t> {
    pub(crate) fn new(tape: &'t Tape, shape: Shape, index: usize) -> Self {
        Self { tape, shape, index }
    }

    fn length(&self) -> usize {
        let (nrow, ncol) = self.shape;
        if nrow == 1 {
            ncol
        } else {
            nrow
        }
    }

    fn unary_vector(&self, op: UnaryOp) -> VectorVar<'t> {
        let shape = op.eval_shape(self.shape);
        assert!(
            shape.is_vector(),
            "The result of the unary operation {:?} on a vector is not a vector.",
            op
        );
        VectorVar {
            tape: self.tape,
            index: self.tape.push_unary(shape, self.index, op),
            shape,
        }
    }

    fn unary_scalar(&self, op: UnaryOp) -> ScalarVar<'t> {
        let shape = op.eval_shape(self.shape);
        assert!(
            shape.is_scalar(),
            "The result of the unary operation {:?} on a vector is not a scalar.",
            op
        );
        ScalarVar::new(self.tape, self.tape.push_unary((1, 1), self.index, op))
    }

    fn binary_vector(&self, op: BinaryOp, other_vector: &VectorVar<'t>) -> VectorVar<'t> {
        assert_eq!(self.tape as *const Tape, other_vector.tape as *const Tape);
        let shape = op.eval_shape(self.shape, other_vector.shape);
        assert!(
            shape.is_vector(),
            "The result of the binary operation {:?} on a vector and a vector is not a vector.",
            op
        );
        VectorVar {
            tape: self.tape,
            index: self
                .tape
                .push_binary(shape, self.index, other_vector.index, op),
            shape,
        }
    }

    fn binary_scalar(&self, op: BinaryOp, other_vector: &VectorVar<'t>) -> ScalarVar<'t> {
        assert_eq!(self.tape as *const Tape, other_vector.tape as *const Tape);
        let shape = op.eval_shape(self.shape, other_vector.shape);
        assert!(
            shape.is_scalar(),
            "The result of the binary operation {:?} on a vector and a vector is not a vector.",
            op
        );
        ScalarVar::new(
            self.tape,
            self.tape
                .push_binary(shape, self.index, other_vector.index, op),
        )
    }

    /// Sets the value of the variable.
    pub fn set(&mut self, new_value: FloatVector) {
        let new_value = new_value.into_shape(self.shape).unwrap();
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => *value = Some(new_value),
            _ => panic!("Cannot set value for non-input variable."),
        }
        self.tape.is_evaluated.set(false);
    }
}

impl<'t> VectorVar<'t> {
    pub fn t(&self) -> Self {
        self.unary_vector(UnaryOp::T)
    }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Self {
        self.unary_vector(UnaryOp::Sin)
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Self {
        self.unary_vector(UnaryOp::Cos)
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Self {
        self.unary_vector(UnaryOp::Tan)
    }

    /// Takes this variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Self {
        let const_var = self
            .tape
            .vector_const(FloatVector::from_elem(self.length(), p));
        self.binary_vector(BinaryOp::Pow, &const_var)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: VectorVar<'t>) -> Self {
        self.binary_vector(BinaryOp::Pow, &other)
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Self {
        self.unary_vector(UnaryOp::Ln)
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Self {
        self.unary_vector(UnaryOp::Exp)
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Self {
        let const_var = self
            .tape
            .vector_const(FloatVector::from_elem(self.length(), base));
        self.binary_vector(BinaryOp::Log, &const_var)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: VectorVar<'t>) -> Self {
        self.binary_vector(BinaryOp::Log, &other)
    }

    pub fn dot(&self, other: VectorVar<'t>) -> ScalarVar<'t> {
        self.binary_scalar(BinaryOp::Dot, &other)
    }

    pub fn sqrt(&self) -> VectorVar<'t> {
        self.pow_const(0.5)
    }

    pub fn l2norm(&self) -> ScalarVar<'t> {
        self.t().dot(*self).sqrt()
    }

    pub fn sum(&self) -> ScalarVar<'t> {
        self.unary_scalar(UnaryOp::Sum)
    }
}

impl<'t> Add<VectorVar<'t>> for VectorVar<'t> {
    type Output = Self;

    fn add(self, other: VectorVar<'t>) -> Self::Output {
        self.binary_vector(BinaryOp::Add, &other)
    }
}

impl<'t> Add<f64> for VectorVar<'t> {
    type Output = Self;

    fn add(self, constant: f64) -> Self::Output {
        let const_var = self
            .tape
            .vector_const(FloatVector::from_elem(self.length(), constant));
        self.binary_vector(BinaryOp::Add, &const_var)
    }
}

impl<'t> Add<VectorVar<'t>> for f64 {
    type Output = VectorVar<'t>;

    fn add(self, var: VectorVar<'t>) -> Self::Output {
        var + self
    }
}

impl<'t> Mul<VectorVar<'t>> for VectorVar<'t> {
    type Output = Self;

    fn mul(self, other: VectorVar<'t>) -> Self::Output {
        self.binary_vector(BinaryOp::Mul, &other)
    }
}

impl<'t> Mul<f64> for VectorVar<'t> {
    type Output = Self;

    fn mul(self, constant: f64) -> Self::Output {
        let const_var = self
            .tape
            .vector_const(FloatVector::from_elem(self.length(), constant));
        self.binary_vector(BinaryOp::Mul, &const_var)
    }
}

impl<'t> Mul<VectorVar<'t>> for f64 {
    type Output = VectorVar<'t>;

    fn mul(self, var: VectorVar<'t>) -> Self::Output {
        var * self
    }
}

impl<'t> Sub<VectorVar<'t>> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn sub(self, other: VectorVar<'t>) -> Self::Output {
        self.binary_vector(BinaryOp::Sub, &other)
    }
}

impl<'t> Sub<f64> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn sub(self, constant: f64) -> Self::Output {
        let const_var = self
            .tape
            .vector_const(FloatVector::from_elem(self.length(), constant));
        self.binary_vector(BinaryOp::Sub, &const_var)
    }
}

impl<'t> Sub<VectorVar<'t>> for f64 {
    type Output = VectorVar<'t>;

    fn sub(self, var: VectorVar<'t>) -> Self::Output {
        let const_var = var
            .tape
            .vector_const(FloatVector::from_elem(var.length(), self));
        const_var.binary_vector(BinaryOp::Sub, &var)
    }
}

impl<'t> Div<VectorVar<'t>> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn div(self, other: VectorVar<'t>) -> Self::Output {
        self.binary_vector(BinaryOp::Div, &other)
    }
}

impl<'t> Div<f64> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn div(self, constant: f64) -> Self::Output {
        let const_var = self
            .tape
            .vector_const(FloatVector::from_elem(self.length(), constant));
        self.binary_vector(BinaryOp::Div, &const_var)
    }
}

impl<'t> Div<VectorVar<'t>> for f64 {
    type Output = VectorVar<'t>;

    fn div(self, var: VectorVar<'t>) -> Self::Output {
        let const_var = var
            .tape
            .vector_const(FloatVector::from_elem(var.length(), self));
        const_var.binary_vector(BinaryOp::Div, &var)
    }
}

impl<'t> Neg for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn neg(self) -> Self::Output {
        self.unary_vector(UnaryOp::Neg)
    }
}
