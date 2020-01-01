use crate::alg::{FloatMatrix, FloatVector};
use crate::op::{BinaryOp, UnaryOp};
use crate::tape::{Node, Tape};
use std::convert::TryFrom;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub struct Shape(pub usize, pub usize);

impl Shape {
    pub fn dim(&self) -> (usize, usize) {
        (self.0, self.1)
    }

    pub fn is_vector(&self) -> bool {
        let Shape(nrow, ncol) = *self;
        nrow == 1 || ncol == 1
    }

    pub fn is_row_vector(&self) -> bool {
        let Shape(nrow, ncol) = *self;
        nrow == 1 && ncol != 1
    }

    pub fn is_col_vector(&self) -> bool {
        let Shape(nrow, ncol) = *self;
        nrow != 1 && ncol == 1
    }

    pub fn is_scalar(&self) -> bool {
        let Shape(nrow, ncol) = *self;
        nrow == 1 && ncol == 1
    }
}

impl From<(usize, usize)> for Shape {
    fn from(dim: (usize, usize)) -> Self {
        let (nrow, ncol) = dim;
        Shape(nrow, ncol)
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
pub struct ScalarKind;

impl ScalarKind {
    pub fn shape() -> Shape {
        Shape(1, 1)
    }
}

impl VarKind for ScalarKind {
    fn shape(&self) -> Shape {
        ScalarKind::shape()
    }
}

impl TryFrom<Shape> for ScalarKind {
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
pub struct VectorKind {
    length: usize,
    is_row: bool,
}

impl VarKind for VectorKind {
    fn shape(&self) -> Shape {
        if self.is_row {
            Shape(1, self.length)
        } else {
            Shape(self.length, 1)
        }
    }
}

impl TryFrom<Shape> for VectorKind {
    type Error = ShapeError;

    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        if value.is_vector() {
            Ok(VectorKind {
                length: value.0 * value.1,
                is_row: value.is_row_vector(),
            })
        } else {
            Err(ShapeError(value, "vector"))
        }
    }
}

#[derive(Copy, Clone)]
pub struct MatrixKind {
    nrow: usize,
    ncol: usize,
}

impl VarKind for MatrixKind {
    fn shape(&self) -> Shape {
        Shape(self.nrow, self.ncol)
    }
}

impl TryFrom<Shape> for MatrixKind {
    type Error = ShapeError;

    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        Ok(MatrixKind {
            nrow: value.0,
            ncol: value.1,
        })
    }
}

pub struct Var<K>
where
    K: VarKind,
{
    tape: Rc<Tape>,
    index: usize,
    kind: K,
}

impl<K> Var<K>
where
    K: VarKind,
{
    /// Initializes a new variable with the given kind.
    fn new(tape: &Rc<Tape>, index: usize, kind: K) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind,
        }
    }

    fn constant_like(&self, value: f64) -> Var<K> {
        let value = FloatMatrix::from_elem(self.kind.shape().dim(), value);
        let index = self.tape.push_constant(value);
        Var::new(&self.tape, index, self.kind)
    }

    /// Creates a new variable resulting from an unary operation applied on this variable.
    fn unary<KResult>(&self, op: UnaryOp) -> Var<KResult>
    where
        KResult: VarKind,
    {
        let shape = op.eval_shape(self.kind.shape());
        let index = self.tape.push_unary(shape, self.index, op);
        Var::new(&self.tape, index, KResult::try_from(shape).unwrap())
    }

    /// Creates a new variable resulting from a binary operation applied on this variable and another.
    fn binary<KOther, KResult>(&self, other: &Var<KOther>, op: BinaryOp) -> Var<KResult>
    where
        KOther: VarKind,
        KResult: VarKind,
    {
        assert_eq!(&*self.tape as *const Tape, &*other.tape as *const Tape);
        let shape = op.eval_shape(self.kind.shape(), other.kind.shape());
        let index = self.tape.push_binary(shape, self.index, other.index, op);
        Var::new(&self.tape, index, KResult::try_from(shape).unwrap())
    }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Self {
        self.unary(UnaryOp::Sin)
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Self {
        self.unary(UnaryOp::Cos)
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Self {
        self.unary(UnaryOp::Tan)
    }

    /// Takes this variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Self {
        self.binary(&self.constant_like(p), BinaryOp::Pow)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: &Self) -> Self {
        self.binary(other, BinaryOp::Pow)
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Self {
        self.unary(UnaryOp::Ln)
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Self {
        self.unary(UnaryOp::Exp)
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Self {
        self.binary(&self.constant_like(base), BinaryOp::Log)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: &Self) -> Self {
        self.binary(other, BinaryOp::Log)
    }

    /// Takes the squared root of this variable.
    pub fn sqrt(&self) -> Self {
        self.pow_const(0.5)
    }
}

impl<K> Add<Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<K> Add<&Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, other: &Var<K>) -> Self::Output {
        self.binary(other, BinaryOp::Add)
    }
}

impl<K> Add<Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<K> Add<&Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, other: &Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<K> Add<f64> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Add)
    }
}

impl<K> Add<f64> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Add)
    }
}

impl<K> Add<Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, other: Var<K>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Add)
    }
}

impl<K> Add<&Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn add(self, other: &Var<K>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Add)
    }
}

impl<K> Sub<Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<K> Sub<&Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, other: &Var<K>) -> Self::Output {
        self.binary(other, BinaryOp::Sub)
    }
}

impl<K> Sub<Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<K> Sub<&Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, other: &Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<K> Sub<f64> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Sub)
    }
}

impl<K> Sub<f64> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Sub)
    }
}

impl<K> Sub<Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, other: Var<K>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Sub)
    }
}

impl<K> Sub<&Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn sub(self, other: &Var<K>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Sub)
    }
}

impl<K> Mul<Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<K> Mul<&Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, other: &Var<K>) -> Self::Output {
        self.binary(other, BinaryOp::Mul)
    }
}

impl<K> Mul<Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<K> Mul<&Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, other: &Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<K> Mul<f64> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Mul)
    }
}

impl<K> Mul<f64> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Mul)
    }
}

impl<K> Mul<Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, other: Var<K>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Mul)
    }
}

impl<K> Mul<&Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn mul(self, other: &Var<K>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Mul)
    }
}

impl<K> Div<Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<K> Div<&Var<K>> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, other: &Var<K>) -> Self::Output {
        self.binary(other, BinaryOp::Div)
    }
}

impl<K> Div<Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, other: Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<K> Div<&Var<K>> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, other: &Var<K>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<K> Div<f64> for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Div)
    }
}

impl<K> Div<f64> for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, c: f64) -> Self::Output {
        self.binary(&self.constant_like(c), BinaryOp::Div)
    }
}

impl<K> Div<Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, other: Var<K>) -> Self::Output {
        other.constant_like(self).binary(&other, BinaryOp::Div)
    }
}

impl<K> Div<&Var<K>> for f64
where
    K: VarKind,
{
    type Output = Var<K>;

    fn div(self, other: &Var<K>) -> Self::Output {
        other.constant_like(self).binary(other, BinaryOp::Div)
    }
}

impl<K> Neg for Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

impl<K> Neg for &Var<K>
where
    K: VarKind,
{
    type Output = Var<K>;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

/// Represents the dot product operation.
pub trait DotProduct<KOther>
where
    KOther: VarKind,
{
    type KResult: VarKind;
    fn dot(&self, other: &Var<KOther>) -> Var<Self::KResult>;
}

pub type ScalarVar = Var<ScalarKind>;
pub type VectorVar = Var<VectorKind>;
pub type MatrixVar = Var<MatrixKind>;

impl ScalarVar {
    /// Initializes a new scalar variable.
    pub(crate) fn scalar(tape: &Rc<Tape>, index: usize) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind: ScalarKind,
        }
    }

    /// Sets the value for this variable.
    pub fn set(&mut self, new_value: f64) {
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => {
                *value = Some(FloatMatrix::from_elem(self.kind.shape().dim(), new_value));
            }
            _ => panic!("Cannot set value for non-input variable."),
        }
        self.tape.is_evaluated.set(false);
    }
}

pub struct Grad {
    derivs: Vec<FloatMatrix>,
}

impl Grad {
    pub fn wrt<K>(&self, var: &Var<K>) -> &FloatMatrix
    where
        K: VarKind,
    {
        &self.derivs[var.index]
    }
}

impl ScalarVar {
    /// Sorts the expression graph in togological order starting from this variable.
    fn topological_sort(&self) -> Vec<usize> {
        let nodes = self.tape.nodes.borrow();
        let mut visited = vec![false; nodes.len()]; // flag visited nodes
        let mut visit_stack = Vec::with_capacity(nodes.len()); // used to store the visited nodes
        let mut nodes_stack = Vec::with_capacity(nodes.len()); // used to store the traversal result
        let mut root = Some(self.index);

        loop {
            while let Some(root_index) = root {
                let root_node = &nodes[root_index];
                match root_node {
                    Node::Constant(_) => {
                        visit_stack.push(root_index);
                        root = None;
                    }
                    Node::Nullary { .. } => {
                        visit_stack.push(root_index);
                        root = None;
                    }
                    Node::Unary { dep, .. } => {
                        visit_stack.push(root_index);
                        root = Some(*dep);
                    }
                    Node::Binary { deps, .. } => {
                        visit_stack.push(deps[1]);
                        visit_stack.push(root_index);
                        root = Some(deps[0]);
                    }
                }
            }

            if let Some(root_index) = visit_stack.pop() {
                let root_node = &nodes[root_index];
                let mut right_index = None;
                match root_node {
                    Node::Binary { deps, .. } => {
                        if let Some(top_index) = visit_stack.last() {
                            if *top_index == deps[1] {
                                right_index = Some(deps[1]);
                            }
                        }
                    }
                    _ => {}
                }
                if let Some(right_index) = right_index {
                    visit_stack.pop();
                    visit_stack.push(root_index);
                    root = Some(right_index);
                } else {
                    if !visited[root_index] {
                        nodes_stack.push(root_index);
                        visited[root_index] = true;
                    }
                }
            }

            if visit_stack.is_empty() {
                break;
            }
        }

        nodes_stack
    }

    /// Evaluates the variable and those that it depends on.
    pub fn eval(&self) -> FloatMatrix {
        let nodes_order = self.topological_sort();
        let mut nodes = self.tape.nodes.borrow_mut();

        // applying the operators on the traversal results from left to right
        for &var_index in &nodes_order {
            let result = {
                let node = &nodes[var_index];
                match node {
                    Node::Constant(_) | Node::Nullary { .. } => None,
                    Node::Unary { dep, op, .. } => Some(op.eval(nodes[*dep].value())),
                    Node::Binary { deps, op, .. } => {
                        Some(op.eval(nodes[deps[0]].value(), nodes[deps[1]].value()))
                    }
                }
            };
            if let Some(result) = result {
                let node = &mut nodes[var_index];
                match node {
                    Node::Constant(_) | Node::Nullary { .. } => {}
                    Node::Unary { ref mut value, .. } => *value = Some(result),
                    Node::Binary { ref mut value, .. } => *value = Some(result),
                }
            }
        }

        self.tape.is_evaluated.set(true);
        nodes[self.index].value().clone()
    }

    /// Computes the gradients of the variable with respects to all of its parameters.
    pub fn grad(&self) -> Grad {
        if !self.tape.is_evaluated.get() {
            panic!("Graph has not been evaluated.");
        }

        let nodes_order = self.topological_sort();
        let nodes = self.tape.nodes.borrow();
        let mut derivs: Vec<FloatMatrix> = nodes
            .iter()
            .map(|x| FloatMatrix::zeros(x.shape().dim()))
            .collect();
        derivs[self.index] = FloatMatrix::ones(derivs[self.index].dim());

        for &var_index in nodes_order.iter().rev() {
            let node = &nodes[var_index];
            match node {
                Node::Constant(_) | Node::Nullary { .. } => {}
                Node::Unary { dep, op, .. } => {
                    let grad = op.grad(&nodes[*dep], node.value(), &derivs[var_index]);
                    derivs[*dep] = &derivs[*dep] + &grad;
                }
                Node::Binary { deps, op, .. } => {
                    let grads = op.grad(
                        &nodes[deps[0]],
                        &nodes[deps[1]],
                        node.value(),
                        &derivs[var_index],
                    );
                    derivs[deps[0]] = &derivs[deps[0]] + &grads[0];
                    derivs[deps[1]] = &derivs[deps[1]] + &grads[1];
                }
            }
        }

        Grad { derivs }
    }
}

impl VectorVar {
    /// Initializes a new vector variable.
    pub(crate) fn vector(tape: &Rc<Tape>, index: usize, shape: Shape) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind: VectorKind::try_from(shape).unwrap(),
        }
    }

    /// Sets the value of the variable.
    pub fn set(&mut self, new_value: FloatVector) {
        let new_value = new_value.into_shape(self.kind.shape().dim()).unwrap();
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => *value = Some(new_value),
            _ => panic!("Cannot set value for dependent variable."),
        }
        self.tape.is_evaluated.set(false);
    }

    /// Takes the tranpose of this variable.
    pub fn t(&self) -> Self {
        self.unary(UnaryOp::T)
    }

    /// Takes the L2 norm of this variable.
    pub fn l2norm(&self) -> ScalarVar {
        self.t().dot(self).sqrt()
    }

    /// Takes the sum of this variable's elements.
    pub fn sum(&self) -> ScalarVar {
        self.unary(UnaryOp::Sum)
    }
}

impl DotProduct<VectorKind> for VectorVar {
    type KResult = ScalarKind;

    fn dot(&self, other: &Var<VectorKind>) -> Var<Self::KResult> {
        self.binary(other, BinaryOp::Dot)
    }
}

impl DotProduct<MatrixKind> for VectorVar {
    type KResult = VectorKind;

    fn dot(&self, other: &Var<MatrixKind>) -> Var<Self::KResult> {
        self.binary(other, BinaryOp::Dot)
    }
}

impl MatrixVar {
    /// Initializes a new matrix variable.
    pub(crate) fn matrix(tape: &Rc<Tape>, index: usize, nrow: usize, ncol: usize) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind: MatrixKind { nrow, ncol },
        }
    }

    /// Takes the transpose of this variable.
    pub fn t(&self) -> Self {
        self.unary(UnaryOp::T)
    }

    /// Sets the value of the variable.
    pub fn set(&mut self, new_value: FloatMatrix) {
        assert_eq!(
            new_value.dim(),
            self.kind.shape().dim(),
            "The shape of the new value does not match {:?} != {:?}.",
            new_value.dim(),
            self.kind.shape().dim()
        );
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => *value = Some(new_value),
            _ => panic!("Cannot set value for non-input variable."),
        }
        self.tape.is_evaluated.set(false);
    }
}

impl DotProduct<VectorKind> for MatrixVar {
    type KResult = VectorKind;

    fn dot(&self, other: &Var<VectorKind>) -> Var<Self::KResult> {
        self.binary(other, BinaryOp::Dot)
    }
}

impl DotProduct<MatrixKind> for MatrixVar {
    type KResult = MatrixKind;

    fn dot(&self, other: &Var<MatrixKind>) -> Var<Self::KResult> {
        self.binary(other, BinaryOp::Dot)
    }
}
