use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
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
    fn grad(&self, left: (&VarNode, f64), right: (&VarNode, f64)) -> [f64; 2] {
        let lg = {
            let (left_node, left_value) = left;
            let (_, right_value) = right;
            match *left_node {
                VarNode::Constant { .. } => 0.0,
                _ => self.left_grad(left_value, right_value),
            }
        };

        let rg = {
            let (_, left_value) = left;
            let (right_node, right_value) = right;
            match *right_node {
                VarNode::Constant { .. } => 0.0,
                _ => self.right_grad(left_value, right_value),
            }
        };

        [lg, rg]
    }
}

/// Presents the operand in the graph
enum VarNode {
    Nullary,
    Unary { op: UnaryOp, dep: usize },
    Binary { op: BinaryOp, deps: [usize; 2] },
    Constant { value: f64 },
}

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
    var_values: RefCell<Vec<Option<f64>>>,
}

impl Tape {
    /// Initialize a new tape
    pub fn new() -> Self {
        Self {
            grad_nodes: RefCell::new(Vec::new()),
            var_nodes: RefCell::new(Vec::new()),
            var_values: RefCell::new(Vec::new()),
        }
    }

    /// Create an input variable
    pub fn var<'t>(&'t self) -> Var {
        let index = self.push_nullary();
        Var { tape: self, index }
    }

    /// Create a constant
    fn constant(&self, value: f64) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let mut vals = self.var_values.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Constant { value });
        vals.push(Some(value));

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    pub fn len(&self) -> usize {
        self.grad_nodes.borrow().len()
    }

    fn push_nullary(&self) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let mut vals = self.var_values.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Nullary);
        vals.push(None);

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    fn push_unary(&self, parent: usize, op: UnaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let mut vals = self.var_values.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Unary { dep: parent, op });
        vals.push(None);

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Unary {
            grad: None,
            dep: parent,
            op,
        });
        len
    }

    fn push_binary(&self, parent0: usize, parent1: usize, op: BinaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let mut vals = self.var_values.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Binary {
            deps: [parent0, parent1],
            op,
        });
        vals.push(None);

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
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
}

impl<'t> Var<'t> {
    pub fn eval(&self) -> f64 {
        let vars = self.tape.var_nodes.borrow();
        let current_var = &vars[self.index];
        let value = match current_var {
            VarNode::Constant { value, .. } => *value,
            VarNode::Nullary { .. } => {
                let vals = self.tape.var_values.borrow();
                vals[self.index].expect("Input variable is not initialized.")
            }
            VarNode::Unary { dep, op, .. } => {
                let parent_var = Var {
                    tape: self.tape,
                    index: *dep,
                };
                op.eval(parent_var.eval())
            }
            VarNode::Binary { deps, op, .. } => {
                let left_var = Var {
                    tape: self.tape,
                    index: deps[0],
                };
                let right_var = Var {
                    tape: self.tape,
                    index: deps[1],
                };
                op.eval(left_var.eval(), right_var.eval())
            }
        };

        let mut vals = self.tape.var_values.borrow_mut();
        vals[self.index] = Some(value);
        value
    }

    pub fn set_value(&self, value: f64) {
        let vars = self.tape.var_nodes.borrow();
        let current_var = &vars[self.index];
        match current_var {
            VarNode::Constant { .. } => panic!("Cannot set value for constant node."),
            VarNode::Binary { .. } => panic!("Cannot set value for computed node."),
            VarNode::Unary { .. } => panic!("Cannot set value for computed node."),
            VarNode::Nullary { .. } => {
                let mut vals = self.tape.var_values.borrow_mut();
                vals[self.index] = Some(value);
            }
        }
    }

    pub fn grad(&self) -> Grad {
        let len = self.tape.len();
        let mut nodes = self.tape.grad_nodes.borrow_mut();
        let vars = self.tape.var_nodes.borrow();
        let vals = self.tape.var_values.borrow();
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
                    let grad_value = op.grad(vals[dep].unwrap());
                    derivs[dep] += grad_value * deriv;
                    *grad = Some(grad_value);
                }
                GradNode::Binary {
                    ref mut grads,
                    deps,
                    op,
                } => {
                    let grad_values = op.grad(
                        (&vars[deps[0]], vals[deps[0]].unwrap()),
                        (&vars[deps[1]], vals[deps[1]].unwrap()),
                    );
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
