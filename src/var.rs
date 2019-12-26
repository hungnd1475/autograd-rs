use crate::op::{BinaryOp, UnaryOp};
use crate::{FloatMatrix, FloatVector, LinearAlgebra, Node, Shape, ShapeExt, Tape};
use auto_ops::impl_op_ex;
use std::convert::From;
use std::ops::Neg;
use std::rc::Rc;

pub trait VarKind: From<Shape> {
    fn shape(&self) -> Shape;
}

pub struct ScalarKind;

impl ScalarKind {
    fn shape() -> Shape {
        (1, 1)
    }
}

impl VarKind for ScalarKind {
    fn shape(&self) -> Shape {
        ScalarKind::shape()
    }
}

impl From<Shape> for ScalarKind {
    fn from(shape: Shape) -> Self {
        assert!(shape.is_scalar(), "Shape {:?} is not for a scalar.", shape);
        Self
    }
}

pub struct VectorKind {
    length: usize,
    is_row: bool,
}

impl VarKind for VectorKind {
    fn shape(&self) -> Shape {
        if self.is_row {
            (1, self.length)
        } else {
            (self.length, 1)
        }
    }
}

impl From<Shape> for VectorKind {
    fn from(shape: Shape) -> Self {
        assert!(shape.is_vector(), "Shape {:?} is not for a vector.", shape);
        Self {
            length: shape.0 * shape.1,
            is_row: shape.is_row_vector(),
        }
    }
}

pub struct MatrixKind {
    nrow: usize,
    ncol: usize,
}

impl VarKind for MatrixKind {
    fn shape(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }
}

impl From<Shape> for MatrixKind {
    fn from(shape: Shape) -> Self {
        Self {
            nrow: shape.0,
            ncol: shape.1,
        }
    }
}

pub struct Var<K>
where
    K: VarKind,
{
    tape: Rc<Tape>,
    pub(crate) index: usize,
    pub(crate) kind: K,
}

impl<K> Var<K>
where
    K: VarKind,
{
    fn new(tape: &Rc<Tape>, index: usize, kind: K) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind,
        }
    }
}

fn unary<KIn, KOut>(input: &Var<KIn>, op: UnaryOp) -> Var<KOut>
where
    KIn: VarKind,
    KOut: VarKind,
{
    let shape = op.eval_shape(input.kind.shape());
    let index = input.tape.push_unary(shape, input.index, op);
    Var::new(&input.tape, index, KOut::from(shape))
}

fn binary<KLeft, KRight, KOut>(left: &Var<KLeft>, right: &Var<KRight>, op: BinaryOp) -> Var<KOut>
where
    KLeft: VarKind,
    KRight: VarKind,
    KOut: VarKind,
{
    assert_eq!(&*left.tape as *const Tape, &*right.tape as *const Tape);
    let shape = op.eval_shape(left.kind.shape(), right.kind.shape());
    let index = left.tape.push_binary(shape, left.index, right.index, op);
    Var::new(&left.tape, index, KOut::from(shape))
}

pub trait DotProduct<K>
where
    K: VarKind,
{
    type Result: VarKind;
    fn dot(&self, other: &Var<K>) -> Var<Self::Result>;
}

pub type ScalarVar = Var<ScalarKind>;
pub type VectorVar = Var<VectorKind>;
pub type MatrixVar = Var<MatrixKind>;

impl ScalarVar {
    pub(crate) fn scalar(tape: &Rc<Tape>, index: usize) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind: ScalarKind,
        }
    }

    pub(crate) fn constant(tape: &Rc<Tape>, value: f64) -> Self {
        let value = FloatMatrix::from_elem(ScalarKind::shape(), value);
        let index = tape.push_constant(value);
        Var::scalar(tape, index)
    }

    /// Sets the value for this variable.
    pub fn set(&mut self, new_value: f64) {
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => {
                *value = Some(FloatMatrix::from_elem(self.kind.shape(), new_value));
            }
            _ => panic!("Cannot set value for non-input variable."),
        }
        self.tape.is_evaluated.set(false);
    }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Self {
        unary(self, UnaryOp::Sin)
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Self {
        unary(self, UnaryOp::Cos)
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Self {
        unary(self, UnaryOp::Tan)
    }

    /// Takes this variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Self {
        let const_var = Self::constant(&self.tape, p);
        binary(self, &const_var, BinaryOp::Pow)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: Self) -> Self {
        binary(self, &other, BinaryOp::Pow)
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Self {
        unary(self, UnaryOp::Ln)
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Self {
        unary(self, UnaryOp::Exp)
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Self {
        let const_var = Self::constant(&self.tape, base);
        binary(self, &const_var, BinaryOp::Log)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: Self) -> Self {
        binary(self, &other, BinaryOp::Log)
    }

    /// Takes the squared root of this variable.
    pub fn sqrt(&self) -> Self {
        self.pow_const(0.5)
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
            .map(|x| FloatMatrix::zeros(x.shape()))
            .collect();
        derivs[self.index] = derivs[self.index].ones_like();

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

impl_op_ex!(+|x: &ScalarVar, y: &ScalarVar| -> ScalarVar { binary(x, y, BinaryOp::Add) });
impl_op_ex!(+|x: &ScalarVar, y: f64| -> ScalarVar { 
    let y = ScalarVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Add)
});
impl_op_ex!(+|x: f64, y: &ScalarVar| -> ScalarVar {
    let x = ScalarVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Add)
});

impl_op_ex!(*|x: &ScalarVar, y: &ScalarVar| -> ScalarVar { binary(x, y, BinaryOp::Mul) });
impl_op_ex!(*|x: &ScalarVar, y: f64| -> ScalarVar {
    let y = ScalarVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Mul)
});
impl_op_ex!(*|x: f64, y: &ScalarVar| -> ScalarVar {
    let x = ScalarVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Add)
});

impl_op_ex!(-|x: &ScalarVar, y: &ScalarVar| -> ScalarVar { binary(x, y, BinaryOp::Sub) });
impl_op_ex!(-|x: &ScalarVar, y: f64| -> ScalarVar {
    let y = ScalarVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Sub)
});
impl_op_ex!(-|x: f64, y: &ScalarVar| -> ScalarVar {
    let x = ScalarVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Sub)
});

impl_op_ex!(/|x: &ScalarVar, y: &ScalarVar| -> ScalarVar { binary(x, y, BinaryOp::Div) });
impl_op_ex!(/|x: &ScalarVar, y: f64| -> ScalarVar {
    let y = ScalarVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Div)
});
impl_op_ex!(/|x: f64, y: &ScalarVar| -> ScalarVar {
    let x = ScalarVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Div)
});

impl Neg for ScalarVar {
    type Output = ScalarVar;

    fn neg(self) -> Self::Output {
        unary(&self, UnaryOp::Neg)
    }
}

impl Neg for &ScalarVar {
    type Output = ScalarVar;

    fn neg(self) -> Self::Output {
        unary(self, UnaryOp::Neg)
    }
}

impl VectorVar {
    pub(crate) fn vector(tape: &Rc<Tape>, index: usize, shape: Shape) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind: VectorKind::from(shape),
        }
    }

    pub(crate) fn constant(tape: &Rc<Tape>, value: FloatVector) -> Self {
        let shape = (value.len(), 1);
        let value = value.into_shape(shape).unwrap();
        let index = tape.push_constant(value);
        Var::vector(tape, index, shape)
    }

    /// Sets the value of the variable.
    pub fn set(&mut self, new_value: FloatVector) {
        let new_value = new_value.into_shape(self.kind.shape()).unwrap();
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => *value = Some(new_value),
            _ => panic!("Cannot set value for dependent variable."),
        }
        self.tape.is_evaluated.set(false);
    }

    /// Takes the tranpose of this variable.
    pub fn t(&self) -> Self {
        unary(self, UnaryOp::T)
    }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Self {
        unary(self, UnaryOp::Sin)
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Self {
        unary(self, UnaryOp::Cos)
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Self {
        unary(self, UnaryOp::Tan)
    }

    /// Takes this variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Self {
        let const_var = Self::constant(&self.tape, FloatVector::from_elem(self.kind.length, p));
        binary(self, &const_var, BinaryOp::Pow)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: &Self) -> Self {
        binary(self, other, BinaryOp::Pow)
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Self {
        unary(self, UnaryOp::Ln)
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Self {
        unary(self, UnaryOp::Exp)
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Self {
        let const_var = Self::constant(&self.tape, FloatVector::from_elem(self.kind.length, base));
        binary(self, &const_var, BinaryOp::Log)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: &Self) -> Self {
        binary(self, other, BinaryOp::Log)
    }

    /// Takes the squared root of this variable.
    pub fn sqrt(&self) -> Self {
        self.pow_const(0.5)
    }

    /// Takes the L2 norm of this variable.
    pub fn l2norm(&self) -> ScalarVar {
        self.t().dot(self).sqrt()
    }

    /// Takes the sum of this variable's elements.
    pub fn sum(&self) -> ScalarVar {
        unary(self, UnaryOp::Sum)
    }
}

impl DotProduct<VectorKind> for VectorVar {
    type Result = ScalarKind;

    fn dot(&self, other: &Var<VectorKind>) -> Var<Self::Result> {
        binary(self, other, BinaryOp::Dot)
    }
}

impl DotProduct<MatrixKind> for VectorVar {
    type Result = MatrixKind;

    fn dot(&self, other: &Var<MatrixKind>) -> Var<Self::Result> {
        binary(self, other, BinaryOp::Dot)
    }
}

impl_op_ex!(+|x: &VectorVar, y: &VectorVar| -> VectorVar { binary(x, y, BinaryOp::Add) });
impl_op_ex!(+|x: &VectorVar, y: f64| -> VectorVar {
    let y = FloatVector::from_elem(x.kind.length, y);
    let y = VectorVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Add)
});
impl_op_ex!(+|x: f64, y: &VectorVar| -> VectorVar {
    let x = FloatVector::from_elem(y.kind.length, x);
    let x = VectorVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Add)
});

impl_op_ex!(*|x: &VectorVar, y: &VectorVar| -> VectorVar { binary(x, y, BinaryOp::Mul) });
impl_op_ex!(*|x: &VectorVar, y: f64| -> VectorVar {
    let y = FloatVector::from_elem(x.kind.length, y);
    let y = VectorVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Mul)
});
impl_op_ex!(*|x: f64, y: &VectorVar| -> VectorVar {
    let x = FloatVector::from_elem(y.kind.length, x);
    let x = VectorVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Add)
});

impl_op_ex!(-|x: &VectorVar, y: &VectorVar| -> VectorVar { binary(x, y, BinaryOp::Sub) });
impl_op_ex!(-|x: &VectorVar, y: f64| -> VectorVar {
    let y = FloatVector::from_elem(x.kind.length, y);
    let y = VectorVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Sub)
});
impl_op_ex!(-|x: f64, y: &VectorVar| -> VectorVar {
    let x = FloatVector::from_elem(y.kind.length, x);
    let x = VectorVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Sub)
});

impl_op_ex!(/|x: &VectorVar, y: &VectorVar| -> VectorVar { binary(x, y, BinaryOp::Div) });
impl_op_ex!(/|x: &VectorVar, y: f64| -> VectorVar {
    let y = FloatVector::from_elem(x.kind.length, y);
    let y = VectorVar::constant(&x.tape, y);
    binary(x, &y, BinaryOp::Div)
});
impl_op_ex!(/|x: f64, y: &VectorVar| -> VectorVar {
    let x = FloatVector::from_elem(y.kind.length, x);
    let x = VectorVar::constant(&y.tape, x);
    binary(&x, y, BinaryOp::Div)
});

impl Neg for VectorVar {
    type Output = VectorVar;

    fn neg(self) -> Self::Output {
        unary(&self, UnaryOp::Neg)
    }
}

impl Neg for &VectorVar {
    type Output = VectorVar;

    fn neg(self) -> Self::Output {
        unary(self, UnaryOp::Neg)
    }
}

impl MatrixVar {
    pub(crate) fn matrix(tape: &Rc<Tape>, index: usize, nrow: usize, ncol: usize) -> Self {
        Self {
            tape: Rc::clone(tape),
            index,
            kind: MatrixKind { nrow, ncol },
        }
    }

    /// Sets the value of the variable.
    pub fn set(&mut self, new_value: FloatMatrix) {
        assert_eq!(
            new_value.dim(),
            self.kind.shape(),
            "The shape of the new value does not match {:?} != {:?}.",
            new_value.dim(),
            self.kind.shape()
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
    type Result = VectorKind;

    fn dot(&self, other: &Var<VectorKind>) -> Var<Self::Result> {
        binary(self, other, BinaryOp::Dot)
    }
}

impl DotProduct<MatrixKind> for MatrixVar {
    type Result = MatrixKind;

    fn dot(&self, other: &Var<MatrixKind>) -> Var<Self::Result> {
        binary(self, other, BinaryOp::Dot)
    }
}
