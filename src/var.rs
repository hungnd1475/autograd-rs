use crate::alg::{FloatMatrix, FloatVector};
use crate::op::{BinaryOp, UnaryOp};
use crate::tape::{Grad, Tape};
use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape(pub [usize; 2]);

impl Shape {
    pub fn dim(&self) -> (usize, usize) {
        let Shape(shape) = self;
        (shape[0], shape[1])
    }

    pub fn is_vector(&self) -> bool {
        let (nrow, ncol) = self.dim();
        nrow == 1 || ncol == 1
    }

    pub fn is_row_vector(&self) -> bool {
        let (nrow, ncol) = self.dim();
        nrow == 1 && ncol != 1
    }

    pub fn is_col_vector(&self) -> bool {
        let (nrow, ncol) = self.dim();
        nrow != 1 && ncol == 1
    }

    pub fn is_scalar(&self) -> bool {
        let (nrow, ncol) = self.dim();
        nrow == 1 && ncol == 1
    }
}

impl From<(usize, usize)> for Shape {
    fn from(dim: (usize, usize)) -> Self {
        let (nrow, ncol) = dim;
        Shape([nrow, ncol])
    }
}

pub struct ShapeError(Shape, &'static str);

impl fmt::Debug for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape {:?} is not for a {:?}.", self.0, self.1)
    }
}

pub trait VarKind: Copy + TryFrom<Shape, Error = ShapeError> {
    fn shape(&self) -> Shape;
}

#[derive(Copy, Clone)]
pub struct Scalar;

impl Scalar {
    pub fn shape() -> Shape {
        Shape([1, 1])
    }
}

impl VarKind for Scalar {
    fn shape(&self) -> Shape {
        Scalar::shape()
    }
}

impl TryFrom<Shape> for Scalar {
    type Error = ShapeError;

    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        if value.is_scalar() {
            Ok(Self)
        } else {
            Err(ShapeError(value, "scalar"))
        }
    }
}

#[derive(Copy, Clone)]
pub struct Vector {
    length: usize,
    is_row: bool,
}

impl VarKind for Vector {
    fn shape(&self) -> Shape {
        if self.is_row {
            Shape([1, self.length])
        } else {
            Shape([self.length, 1])
        }
    }
}

impl TryFrom<Shape> for Vector {
    type Error = ShapeError;

    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        if value.is_vector() {
            let (nrow, ncol) = value.dim();
            Ok(Vector {
                length: nrow * ncol,
                is_row: value.is_row_vector(),
            })
        } else {
            Err(ShapeError(value, "vector"))
        }
    }
}

#[derive(Copy, Clone)]
pub struct Matrix {
    nrow: usize,
    ncol: usize,
}

impl VarKind for Matrix {
    fn shape(&self) -> Shape {
        Shape([self.nrow, self.ncol])
    }
}

impl TryFrom<Shape> for Matrix {
    type Error = ShapeError;

    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        let (nrow, ncol) = value.dim();
        Ok(Matrix { nrow, ncol })
    }
}

pub struct Nullary;
pub struct Constant;
pub struct Unary;
pub struct Binary;

pub struct Var<K, S>
where
    K: VarKind,
{
    tape: Rc<Tape>,
    pub(crate) index: usize,
    kind: K,
    _source: PhantomData<S>,
}

impl<K, S> Var<K, S>
where
    K: VarKind,
{
    /// Initializes a new variable with the given kind.
    fn new(tape: &Rc<Tape>, index: usize, kind: K) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind,
            _source: PhantomData,
        }
    }

    fn constant_like(&self, value: f64) -> Var<K, Constant> {
        let value = FloatMatrix::from_elem(self.kind.shape().dim(), value);
        let index = self.tape.push_constant(value);
        Var::new(&self.tape, index, self.kind)
    }

    /// Creates a new variable resulting from an unary operation applied on this variable.
    fn unary<KResult>(&self, op: UnaryOp) -> Var<KResult, Unary>
    where
        KResult: VarKind,
    {
        let shape = op.eval_shape(self.kind.shape());
        let index = self.tape.push_unary(shape, self.index, op);
        Var::new(&self.tape, index, KResult::try_from(shape).unwrap())
    }

    /// Creates a new variable resulting from a binary operation applied on this variable and another.
    fn binary<KOther, SOther, KResult>(
        &self,
        other: &Var<KOther, SOther>,
        op: BinaryOp,
    ) -> Var<KResult, Binary>
    where
        KOther: VarKind,
        KResult: VarKind,
    {
        assert_eq!(&*self.tape as *const Tape, &*other.tape as *const Tape);
        let shape = op.eval_shape(self.kind.shape(), other.kind.shape());
        let index = self.tape.push_binary(shape, self.index, other.index, op);
        Var::new(&self.tape, index, KResult::try_from(shape).unwrap())
    }

    /// Takes the tranpose of this variable.
    pub fn t(&self) -> Var<K, Unary> {
        self.unary(UnaryOp::T)
    }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Var<K, Unary> {
        self.unary(UnaryOp::Sin)
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Var<K, Unary> {
        self.unary(UnaryOp::Cos)
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Var<K, Unary> {
        self.unary(UnaryOp::Tan)
    }

    /// Takes this variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Var<K, Binary> {
        self.binary(&self.constant_like(p), BinaryOp::Pow)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: &Self) -> Var<K, Binary> {
        self.binary(other, BinaryOp::Pow)
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Var<K, Unary> {
        self.unary(UnaryOp::Ln)
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Var<K, Unary> {
        self.unary(UnaryOp::Exp)
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Var<K, Binary> {
        self.binary(&self.constant_like(base), BinaryOp::Log)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: &Self) -> Var<K, Binary> {
        self.binary(other, BinaryOp::Log)
    }

    /// Takes the squared root of this variable.
    pub fn sqrt(&self) -> Var<K, Binary> {
        self.pow_const(0.5)
    }
}

impl<S> Var<Scalar, S> {
    /// Initializes a new scalar variable.
    pub(crate) fn scalar(tape: &Rc<Tape>, index: usize) -> Self {
        Var::new(tape, index, Scalar)
    }

    /// Broadcasts this scalar to a vector of the given shape.
    pub fn broadcast(&self, shape: Shape) -> Var<Vector, Unary> {
        assert!(shape.is_vector(), "Broadcasting shape must be a vector.");
        self.unary(UnaryOp::Broadcast(shape))
    }
}

impl Var<Scalar, Nullary> {
    /// Sets the value for this variable.
    pub fn set(&mut self, value: f64) {
        let value = FloatMatrix::from_elem(self.kind.shape().dim(), value);
        self.tape.set_value(self.index, value);
    }
}

impl<S> Var<Scalar, S> {
    /// Evaluates the variable and those that it depends on.
    pub fn eval(&self) -> FloatMatrix {
        self.tape.eval(self.index)
    }

    /// Computes the gradients of the variable with respects to all of its parameters.
    pub fn grad(&self) -> Grad {
        self.tape.grad(self.index)
    }
}

impl<S> Var<Vector, S> {
    /// Initializes a new vector variable.
    pub(crate) fn vector(tape: &Rc<Tape>, index: usize, shape: Shape) -> Self {
        Var::new(tape, index, Vector::try_from(shape).unwrap())
    }

    /// Takes the L2 norm of this variable.
    pub fn l2norm(&self) -> Var<Scalar, Binary> {
        self.t().dot(self).sqrt()
    }

    /// Takes the sum of this variable's elements.
    pub fn sum(&self) -> Var<Scalar, Unary> {
        let shape = self.kind.shape();
        let axis = if shape.is_col_vector() { 0 } else { 1 };
        self.unary(UnaryOp::Sum(axis))
    }

    /// Broadcasts this vector to a matrix of the given shape.
    pub fn broadcast(&self, shape: Shape) -> Var<Matrix, Unary> {
        self.unary(UnaryOp::Broadcast(shape))
    }
}

impl Var<Vector, Nullary> {
    /// Sets the value of the variable.
    pub fn set(&mut self, value: FloatVector) {
        let value = value.into_shape(self.kind.shape().dim()).unwrap();
        self.tape.set_value(self.index, value);
    }
}

impl<S> Var<Matrix, S> {
    /// Initializes a new matrix variable.
    pub(crate) fn matrix(tape: &Rc<Tape>, index: usize, nrow: usize, ncol: usize) -> Self {
        Var::new(tape, index, Matrix { nrow, ncol })
    }

    /// Takes the sum of this variable.
    pub fn sum(&self, axis: usize) -> Var<Vector, Unary> {
        self.unary(UnaryOp::Sum(axis))
    }
}

impl Var<Matrix, Nullary> {
    /// Sets the value of the variable.
    pub fn set(&mut self, value: FloatMatrix) {
        assert_eq!(
            value.dim(),
            self.kind.shape().dim(),
            "The shape of the new value does not match {:?} != {:?}.",
            value.dim(),
            self.kind.shape().dim()
        );
        self.tape.set_value(self.index, value);
    }
}

impl<K, SL, SR> Add<Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<K, SL, SR> Add<&Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(other, BinaryOp::Add)
    }
}

impl<K, SL, SR> Add<Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<K, SL, SR> Add<&Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<K, S> Add<f64> for Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Add)
    }
}

impl<K, S> Add<f64> for &Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Add)
    }
}

impl<K, S> Add<Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, other: Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Add)
    }
}

impl<K, S> Add<&Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn add(self, other: &Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Add)
    }
}

impl<SL, SR> Add<&Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn add(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn add(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<&Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn add(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn add(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<&Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn add(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn add(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<&Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn add(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<SL, SR> Add<Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn add(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Add)
    }
}

impl<K, SL, SR> Sub<Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<K, SL, SR> Sub<&Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(other, BinaryOp::Sub)
    }
}

impl<K, SL, SR> Sub<Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<K, SL, SR> Sub<&Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<K, S> Sub<f64> for Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Sub)
    }
}

impl<K, S> Sub<f64> for &Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Sub)
    }
}

impl<K, S> Sub<Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, other: Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Sub)
    }
}

impl<K, S> Sub<&Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn sub(self, other: &Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<&Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn sub(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn sub(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<&Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn sub(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn sub(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<&Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn sub(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn sub(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<&Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn sub(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<SL, SR> Sub<Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn sub(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Sub)
    }
}

impl<K, SL, SR> Mul<Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<K, SL, SR> Mul<&Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(other, BinaryOp::Mul)
    }
}

impl<K, SL, SR> Mul<Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<K, SL, SR> Mul<&Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<K, S> Mul<f64> for Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Mul)
    }
}

impl<K, S> Mul<f64> for &Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Mul)
    }
}

impl<K, S> Mul<Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, other: Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Mul)
    }
}

impl<K, S> Mul<&Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn mul(self, other: &Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<&Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn mul(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn mul(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<&Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn mul(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn mul(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<&Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn mul(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn mul(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<&Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn mul(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<SL, SR> Mul<Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn mul(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Mul)
    }
}

impl<K, SL, SR> Div<Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<K, SL, SR> Div<&Var<K, SR>> for Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(other, BinaryOp::Div)
    }
}

impl<K, SL, SR> Div<Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, other: Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<K, SL, SR> Div<&Var<K, SR>> for &Var<K, SL>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, other: &Var<K, SR>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<K, S> Div<f64> for Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Div)
    }
}

impl<K, S> Div<f64> for &Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Div)
    }
}

impl<K, S> Div<Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, other: Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Div)
    }
}

impl<K, S> Div<&Var<K, S>> for f64
where
    K: VarKind,
{
    type Output = Var<K, Binary>;

    fn div(self, other: &Var<K, S>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<K, S> Neg for Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Unary>;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

impl<K, S> Neg for &Var<K, S>
where
    K: VarKind,
{
    type Output = Var<K, Unary>;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

/// Represents the dot product operation.
pub trait DotProduct<K, S>
where
    K: VarKind,
{
    type KResult: VarKind;
    fn dot(&self, other: &Var<K, S>) -> Var<Self::KResult, Binary>;
}

impl<SL, SR> DotProduct<Vector, SR> for Var<Vector, SL> {
    type KResult = Scalar;

    fn dot(&self, other: &Var<Vector, SR>) -> Var<Self::KResult, Binary> {
        self.binary(other, BinaryOp::Dot)
    }
}

impl<SL, SR> DotProduct<Matrix, SR> for Var<Vector, SL> {
    type KResult = Vector;

    fn dot(&self, other: &Var<Matrix, SR>) -> Var<Self::KResult, Binary> {
        self.binary(other, BinaryOp::Dot)
    }
}

impl<SL, SR> DotProduct<Vector, SR> for Var<Matrix, SL> {
    type KResult = Vector;

    fn dot(&self, other: &Var<Vector, SR>) -> Var<Self::KResult, Binary> {
        self.binary(other, BinaryOp::Dot)
    }
}

impl<SL, SR> DotProduct<Matrix, SR> for Var<Matrix, SL> {
    type KResult = Matrix;

    fn dot(&self, other: &Var<Matrix, SR>) -> Var<Self::KResult, Binary> {
        self.binary(other, BinaryOp::Dot)
    }
}
