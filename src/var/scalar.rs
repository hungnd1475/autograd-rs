use super::{Grad, Var};
use crate::op::{BinaryOp, UnaryOp};
use crate::{FloatMatrix, MatrixExt, Node, Shape, Tape};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub struct ScalarVar<'t> {
    tape: &'t Tape,
    index: usize,
}

impl<'t> ScalarVar<'t> {
    fn unary(&self, op: UnaryOp) -> ScalarVar<'t> {
        ScalarVar::new(self.tape, self.tape.push_unary((1, 1), self.index, op))
    }

    fn binary(&self, op: BinaryOp, other: &ScalarVar<'t>) -> ScalarVar<'t> {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        ScalarVar::new(
            self.tape,
            self.tape.push_binary((1, 1), self.index, other.index, op),
        )
    }

    pub(crate) fn new(tape: &'t Tape, index: usize) -> Self {
        Self { tape, index }
    }

    pub fn set(&mut self, new_value: f64) {
        let mut nodes = self.tape.nodes.borrow_mut();
        match &mut nodes[self.index] {
            Node::Nullary { ref mut value, .. } => {
                *value = Some(FloatMatrix::from_elem((1, 1), new_value));
            }
            _ => panic!("Cannot set value for non-input variable."),
        }
        self.tape.is_evaluated.set(false);
    }

    pub fn index(&self) -> usize {
        self.index
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
        let const_var = self.tape.scalar_const(p);
        self.binary(BinaryOp::Pow, &const_var)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: ScalarVar<'t>) -> Self {
        self.binary(BinaryOp::Pow, &other)
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
        let const_var = self.tape.scalar_const(base);
        self.binary(BinaryOp::Log, &const_var)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: ScalarVar<'t>) -> Self {
        self.binary(BinaryOp::Log, &other)
    }

    pub fn sqrt(&self) -> Self {
        self.pow_const(0.5)
    }
}

impl<'t> ScalarVar<'t> {
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

        // println!("{:?}", vars_order);
        // println!("{:?}", vals);

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
                    let grad = op.grad(&nodes[*dep], nodes[var_index].value(), &derivs[var_index]);
                    derivs[*dep] = &derivs[*dep] + &grad;
                }
                Node::Binary { deps, op, .. } => {
                    let grads = op.grad(
                        &nodes[deps[0]],
                        &nodes[deps[1]],
                        nodes[var_index].value(),
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

impl<'t> Var for ScalarVar<'t> {
    fn index(&self) -> usize {
        self.index
    }

    fn shape(&self) -> Shape {
        (1, 1)
    }
}

impl<'t> Add<ScalarVar<'t>> for ScalarVar<'t> {
    type Output = Self;

    fn add(self, other: ScalarVar<'t>) -> Self::Output {
        self.binary(BinaryOp::Add, &other)
    }
}

impl<'t> Add<f64> for ScalarVar<'t> {
    type Output = Self;

    fn add(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant);
        self.binary(BinaryOp::Add, &const_var)
    }
}

impl<'t> Add<ScalarVar<'t>> for f64 {
    type Output = ScalarVar<'t>;

    fn add(self, var: ScalarVar<'t>) -> Self::Output {
        var + self
    }
}

impl<'t> Mul<ScalarVar<'t>> for ScalarVar<'t> {
    type Output = Self;

    fn mul(self, other: ScalarVar<'t>) -> Self::Output {
        self.binary(BinaryOp::Mul, &other)
    }
}

impl<'t> Mul<f64> for ScalarVar<'t> {
    type Output = Self;

    fn mul(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant);
        self.binary(BinaryOp::Mul, &const_var)
    }
}

impl<'t> Mul<ScalarVar<'t>> for f64 {
    type Output = ScalarVar<'t>;

    fn mul(self, var: ScalarVar<'t>) -> Self::Output {
        var * self
    }
}

impl<'t> Sub<ScalarVar<'t>> for ScalarVar<'t> {
    type Output = ScalarVar<'t>;

    fn sub(self, other: ScalarVar<'t>) -> Self::Output {
        self.binary(BinaryOp::Sub, &other)
    }
}

impl<'t> Sub<f64> for ScalarVar<'t> {
    type Output = ScalarVar<'t>;

    fn sub(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant);
        self.binary(BinaryOp::Sub, &const_var)
    }
}

impl<'t> Sub<ScalarVar<'t>> for f64 {
    type Output = ScalarVar<'t>;

    fn sub(self, var: ScalarVar<'t>) -> Self::Output {
        let const_var = var.tape.scalar_const(self);
        const_var.binary(BinaryOp::Sub, &var)
    }
}

impl<'t> Div<ScalarVar<'t>> for ScalarVar<'t> {
    type Output = ScalarVar<'t>;

    fn div(self, other: ScalarVar<'t>) -> Self::Output {
        self.binary(BinaryOp::Div, &other)
    }
}

impl<'t> Div<f64> for ScalarVar<'t> {
    type Output = ScalarVar<'t>;

    fn div(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant);
        self.binary(BinaryOp::Div, &const_var)
    }
}

impl<'t> Div<ScalarVar<'t>> for f64 {
    type Output = ScalarVar<'t>;

    fn div(self, var: ScalarVar<'t>) -> Self::Output {
        let const_var = var.tape.scalar_const(self);
        const_var.binary(BinaryOp::Div, &var)
    }
}

impl<'t> Neg for ScalarVar<'t> {
    type Output = ScalarVar<'t>;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}
