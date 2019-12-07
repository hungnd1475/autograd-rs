use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
/// Represents the supported unary operations
enum UnaryOp {
    Sin,
    Cos,
    Ln,
    Exp,
}

impl UnaryOp {
    /// Evaluate the operation with the given parameter
    fn eval(&self, value: f64) -> f64 {
        match *self {
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Exp => value.exp(),
        }
    }

    /// Compute the gradient of the operation with respect to the given parameter
    fn grad(&self, value: f64) -> f64 {
        match *self {
            UnaryOp::Sin => value.cos(),
            UnaryOp::Cos => -value.sin(),
            UnaryOp::Ln => 1.0 / value,
            UnaryOp::Exp => value.exp(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Represents the supported binary operations
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl BinaryOp {
    /// Evaluate the operation with the given parameters
    fn eval(&self, left: f64, right: f64) -> f64 {
        match *self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
            BinaryOp::Pow => left.powf(right),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the left parameter
    fn left_grad(&self, left: f64, right: f64) -> f64 {
        match *self {
            BinaryOp::Add => 1.0,
            BinaryOp::Sub => 1.0,
            BinaryOp::Mul => right,
            BinaryOp::Div => 1.0 / right,
            BinaryOp::Pow => right * left.powf(right - 1.0),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the right parameter
    fn right_grad(&self, left: f64, right: f64) -> f64 {
        match *self {
            BinaryOp::Add => 1.0,
            BinaryOp::Sub => -1.0,
            BinaryOp::Mul => left,
            BinaryOp::Div => -left / right.powi(2),
            BinaryOp::Pow => left.ln() * left.powf(right),
        }
    }

    /// Compute the full gradient of the operation with repsect to the given parameters
    fn grad(&self, left: &VarNode, right: &VarNode) -> [f64; 2] {
        let lg = {
            match *left {
                VarNode::Constant { .. } => 0.0,
                _ => self.left_grad(left.value(), right.value()),
            }
        };

        let rg = {
            match *right {
                VarNode::Constant { .. } => 0.0,
                _ => self.right_grad(left.value(), right.value()),
            }
        };

        [lg, rg]
    }
}

#[derive(Debug)]
/// Presents the operand in the graph
enum VarNode {
    Nullary {
        value: f64,
    },
    Unary {
        value: Option<f64>,
        op: UnaryOp,
        dep: usize,
    },
    Binary {
        value: Option<f64>,
        op: BinaryOp,
        deps: [usize; 2],
    },
    Constant {
        value: f64,
    },
}

impl VarNode {
    fn value(&self) -> f64 {
        match *self {
            VarNode::Constant { value } => value,
            VarNode::Nullary { value } => value,
            VarNode::Unary { value, .. } => value.unwrap(),
            VarNode::Binary { value, .. } => value.unwrap(),
        }
    }
}

#[derive(Debug)]
/// Represents the adjoint in the graph
enum GradNode {
    Nullary,
    Unary {
        grad: Option<f64>,
        dep: usize,
        op: UnaryOp,
    },
    Binary {
        grads: Option<[f64; 2]>,
        deps: [usize; 2],
        op: BinaryOp,
    },
}

/// Represents a tape used to construct the adjoint graph
pub struct Tape {
    grad_nodes: RefCell<Vec<GradNode>>,
    var_nodes: RefCell<Vec<VarNode>>,
}

impl Tape {
    /// Initialize a new tape
    pub fn new() -> Self {
        Self {
            grad_nodes: RefCell::new(Vec::new()),
            var_nodes: RefCell::new(Vec::new()),
        }
    }

    /// Create a node representing an input variable
    pub fn var<'t>(&'t self, value: f64) -> Var {
        let index = self.push_nullary(value);
        Var { tape: self, index }
    }

    /// Create a node representing a constant
    fn constant(&self, value: f64) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Constant { value });

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Get the length of the tape
    pub fn len(&self) -> usize {
        self.grad_nodes.borrow().len()
    }

    /// Push a node representing an input variable onto the graph
    fn push_nullary(&self, value: f64) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Nullary { value });

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Push a node representing the result of an unary operator onto the graph
    fn push_unary(&self, parent: usize, op: UnaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Unary {
            value: None,
            dep: parent,
            op,
        });

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Unary {
            grad: None,
            dep: parent,
            op,
        });
        len
    }

    /// Push a node representing the result of a binary operator onto the graph
    fn push_binary(&self, parent0: usize, parent1: usize, op: BinaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Binary {
            value: None,
            deps: [parent0, parent1],
            op,
        });

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Binary {
            grads: None,
            deps: [parent0, parent1],
            op,
        });
        len
    }
}

#[derive(Clone, Copy)]
/// Representing a real variable
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
}

impl<'t> Var<'t> {
    /// Evaluate the variable and those that it is depends on
    pub fn eval(&self) -> f64 {
        // The basic idea is that we traverse the expression graph in post order,
        // then apply the operators on the traversal result from left to right

        let mut vars = self.tape.var_nodes.borrow_mut();
        let mut var_stack = Vec::new();
        let mut exp_stack = Vec::new();
        let mut val_stack = Vec::new();
        let mut root = Some(self.index);

        loop {
            while let Some(root_index) = root {
                let root_node = &vars[root_index];
                match *root_node {
                    VarNode::Constant { .. } => {
                        var_stack.push(root_index);
                        root = None;
                    }
                    VarNode::Nullary { .. } => {
                        var_stack.push(root_index);
                        root = None;
                    }
                    VarNode::Unary { dep, .. } => {
                        var_stack.push(root_index);
                        root = Some(dep);
                    }
                    VarNode::Binary { deps, .. } => {
                        var_stack.push(deps[1]);
                        var_stack.push(root_index);
                        root = Some(deps[0]);
                    }
                }
            }

            if let Some(root_index) = var_stack.pop() {
                let root_node = &vars[root_index];
                let mut right_index = None;
                match *root_node {
                    VarNode::Binary { deps, .. } => {
                        if let Some(top_index) = var_stack.last() {
                            if *top_index == deps[1] {
                                right_index = Some(deps[1]);
                            }
                        }
                    }
                    _ => {}
                }
                if let Some(right_index) = right_index {
                    var_stack.pop();
                    var_stack.push(root_index);
                    root = Some(right_index);
                } else {
                    exp_stack.push(root_index);
                }
            }

            if var_stack.is_empty() {
                break;
            }
        }

        for index in exp_stack {
            let node = &mut vars[index];
            match *node {
                VarNode::Constant { value, .. } => val_stack.push(value),
                VarNode::Nullary { value } => val_stack.push(value),
                VarNode::Unary {
                    op, ref mut value, ..
                } => {
                    let val = val_stack.pop().unwrap();
                    let res = op.eval(val);
                    val_stack.push(res);
                    *value = Some(res);
                }
                VarNode::Binary {
                    op, ref mut value, ..
                } => {
                    let right = val_stack.pop().unwrap();
                    let left = val_stack.pop().unwrap();
                    let res = op.eval(left, right);
                    val_stack.push(res);
                    *value = Some(res);
                }
            }
        }

        val_stack.pop().unwrap()
    }

    /// Set the value of the variable
    pub fn set(&self, new_value: f64) {
        let mut vars = self.tape.var_nodes.borrow_mut();
        let current_var = &mut vars[self.index];
        match current_var {
            VarNode::Constant { .. } => panic!("Cannot set value for constant node."),
            VarNode::Binary { .. } => panic!("Cannot set value for computed node."),
            VarNode::Unary { .. } => panic!("Cannot set value for computed node."),
            VarNode::Nullary { ref mut value } => {
                *value = new_value;
            }
        }
    }

    /// Compute the gradients of the variable with respects to all of its parameters
    pub fn grad(&self) -> Grad {
        let len = self.tape.len();
        let mut nodes = self.tape.grad_nodes.borrow_mut();
        let vars = self.tape.var_nodes.borrow();
        let mut derivs = vec![0.0; len];
        derivs[self.index] = 1.0;

        for i in (0..len).rev() {
            let node = &mut nodes[i];
            let deriv = derivs[i];
            match *node {
                GradNode::Nullary => {}
                GradNode::Unary {
                    ref mut grad,
                    dep,
                    op,
                } => {
                    let grad_value = op.grad(vars[dep].value());
                    derivs[dep] += grad_value * deriv;
                    *grad = Some(grad_value);
                }
                GradNode::Binary {
                    ref mut grads,
                    deps,
                    op,
                } => {
                    let grad_values = op.grad(&vars[deps[0]], &vars[deps[1]]);
                    derivs[deps[0]] += grad_values[0] * deriv;
                    derivs[deps[1]] += grad_values[1] * deriv;
                    *grads = Some(grad_values);
                }
            }
        }

        Grad { derivs }
    }

    pub fn sin(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Sin),
        }
    }

    pub fn cos(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Cos),
        }
    }

    pub fn pow_const(&self, p: f64) -> Self {
        let const_index = self.tape.constant(p);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Pow),
        }
    }

    pub fn pow(&self, other: Var<'t>) -> Self {
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Pow),
        }
    }

    pub fn ln(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Ln),
        }
    }

    pub fn exp(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Exp),
        }
    }
}

impl<'t> Add<Var<'t>> for Var<'t> {
    type Output = Self;

    fn add(self, other: Var<'t>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Add),
        }
    }
}

impl<'t> Add<f64> for Var<'t> {
    type Output = Self;

    fn add(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant(constant);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Add),
        }
    }
}

impl<'t> Add<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn add(self, var: Var<'t>) -> Self::Output {
        var + self
    }
}

impl<'t> Mul<Var<'t>> for Var<'t> {
    type Output = Self;

    fn mul(self, other: Var<'t>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Mul),
        }
    }
}

impl<'t> Mul<f64> for Var<'t> {
    type Output = Self;

    fn mul(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant(constant);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Mul),
        }
    }
}

impl<'t> Mul<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn mul(self, var: Var<'t>) -> Self::Output {
        var * self
    }
}

impl<'t> Sub<Var<'t>> for Var<'t> {
    type Output = Var<'t>;

    fn sub(self, other: Var<'t>) -> Self::Output {
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Sub),
        }
    }
}

impl<'t> Sub<f64> for Var<'t> {
    type Output = Var<'t>;

    fn sub(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant(constant);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Sub),
        }
    }
}

impl<'t> Sub<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn sub(self, var: Var<'t>) -> Self::Output {
        let const_index = var.tape.constant(self);
        Var {
            tape: var.tape,
            index: var.tape.push_binary(const_index, var.index, BinaryOp::Sub),
        }
    }
}

impl<'t> Div<Var<'t>> for Var<'t> {
    type Output = Var<'t>;

    fn div(self, other: Var<'t>) -> Self::Output {
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Div),
        }
    }
}

impl<'t> Div<f64> for Var<'t> {
    type Output = Var<'t>;

    fn div(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant(constant);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Div),
        }
    }
}

impl<'t> Div<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn div(self, var: Var<'t>) -> Self::Output {
        let const_index = var.tape.constant(self);
        Var {
            tape: var.tape,
            index: var.tape.push_binary(const_index, var.index, BinaryOp::Div),
        }
    }
}

pub struct Grad {
    derivs: Vec<f64>,
}

impl Grad {
    pub fn wrt<'t>(&self, var: Var<'t>) -> f64 {
        self.derivs[var.index]
    }
}
