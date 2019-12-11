use std::cell::{Cell, RefCell};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
/// Represents the supported unary operations.
enum UnaryOp {
    Neg,
    Sin,
    Cos,
    Tan,
    Ln,
    Exp,
}

impl UnaryOp {
    /// Evaluates the operation with the given parameter.
    fn eval(&self, value: f64) -> f64 {
        match self {
            UnaryOp::Neg => -value,
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Exp => value.exp(),
        }
    }

    /// Computes the gradient of the operation with respect to the given parameter.
    fn grad(&self, var: &VarNode) -> f64 {
        match var {
            VarNode::Constant(_) => 0.0,
            _ => {
                let value = var.value();
                match self {
                    UnaryOp::Neg => -1.0,
                    UnaryOp::Sin => value.cos(),
                    UnaryOp::Cos => -value.sin(),
                    UnaryOp::Tan => 2.0 / ((2.0 * value).cos() + 1.0),
                    UnaryOp::Ln => 1.0 / value,
                    UnaryOp::Exp => value.exp(),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Represents the supported binary operations.
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
}

impl BinaryOp {
    /// Evaluates the operation with the given parameters.
    fn eval(&self, left: f64, right: f64) -> f64 {
        match self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
            BinaryOp::Pow => left.powf(right),
            BinaryOp::Log => left.log(right),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the left parameter.
    fn left_grad(&self, left: f64, right: f64) -> f64 {
        match self {
            BinaryOp::Add => 1.0,
            BinaryOp::Sub => 1.0,
            BinaryOp::Mul => right,
            BinaryOp::Div => 1.0 / right,
            BinaryOp::Pow => right * left.powf(right - 1.0),
            BinaryOp::Log => 1.0 / (left * right.ln()),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the right parameter.
    fn right_grad(&self, left: f64, right: f64) -> f64 {
        match self {
            BinaryOp::Add => 1.0,
            BinaryOp::Sub => -1.0,
            BinaryOp::Mul => left,
            BinaryOp::Div => -left / right.powi(2),
            BinaryOp::Pow => left.ln() * left.powf(right),
            BinaryOp::Log => -left.ln() / (right.ln().powi(2) * right),
        }
    }

    /// Computes the full gradient of the operation with repsect to the given parameters.
    fn grad(&self, left: &VarNode, right: &VarNode) -> [f64; 2] {
        let lg = {
            match left {
                VarNode::Constant(_) => 0.0,
                _ => self.left_grad(left.value(), right.value()),
            }
        };

        let rg = {
            match right {
                VarNode::Constant(_) => 0.0,
                _ => self.right_grad(left.value(), right.value()),
            }
        };

        [lg, rg]
    }
}

#[derive(Debug)]
/// Presents the operand in the graph.
enum VarNode {
    /// Represents a constant.
    Constant(f64),
    /// Represents an input variable.
    Nullary(Option<f64>),
    /// Represents the result of an unary operation.
    Unary {
        value: Option<f64>, // the lazily computed value
        op: UnaryOp,        // the operation resulting in the node
        dep: usize,         // the operand for this node's operation
    },
    /// Represents the result of an binary operation.
    Binary {
        value: Option<f64>, // the lazily computed value
        op: BinaryOp,       // the operation resulting in the node
        deps: [usize; 2],   // the operands for this node's operation
    },
}

impl VarNode {
    fn value(&self) -> f64 {
        match self {
            VarNode::Constant(value) => *value,
            VarNode::Nullary(value) => value.expect("Input variable has not been initialized."),
            VarNode::Unary { value, .. } => {
                value.expect("Intermediate variable has not been evaluated.")
            }
            VarNode::Binary { value, .. } => {
                value.expect("Intermediate variable has not been evaluated.")
            }
        }
    }
}

#[derive(Debug)]
/// Represents the adjoint in the graph.
enum GradNode {
    /// Represents the gradident of an input.
    Nullary,
    /// Represents the gradient of an unary operation.
    Unary {
        value: Option<f64>, // the lazily computed gradient
        dep: usize,         // the operands for this node's operation
        op: UnaryOp,        // the operation resulting in this node
    },
    /// Represents the gradients of a binary operation.
    Binary {
        values: Option<[f64; 2]>, // the lazily computed gradients
        deps: [usize; 2],         // the operands for this node's operation
        op: BinaryOp,             // the operation resulting in this node
    },
}

/// Represents a tape used to construct the adjoint graph.
pub struct Tape {
    grad_nodes: RefCell<Vec<GradNode>>,
    var_nodes: RefCell<Vec<VarNode>>,
    is_evaluated: Cell<bool>,
}

impl Tape {
    /// Initializes a new tape.
    pub fn new() -> Self {
        Self {
            grad_nodes: RefCell::new(Vec::new()),
            var_nodes: RefCell::new(Vec::new()),
            is_evaluated: Cell::new(false),
        }
    }

    /// Creates a node representing an input variable.
    pub fn var<'t>(&'t self, value: f64) -> Var {
        let index = self.push_nullary(Some(value));
        Var { tape: self, index }
    }

    /// Creates a node representing an unintialized input variable.
    pub fn var_unintialized<'t>(&'t self) -> Var {
        let index = self.push_nullary(None);
        Var { tape: self, index }
    }

    /// Creates a node representing a constant.
    fn constant(&self, value: f64) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Constant(value));

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.grad_nodes.borrow().len()
    }

    /// Pushes a node representing an input variable onto the graph.
    fn push_nullary(&self, value: Option<f64>) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Nullary(value));

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Pushes a node representing the result of an unary operator onto the graph.
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
            value: None,
            dep: parent,
            op,
        });
        len
    }

    /// Pushes a node representing the result of a binary operator onto the graph.
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
            values: None,
            deps: [parent0, parent1],
            op,
        });
        len
    }
}

#[derive(Clone, Copy)]
/// Represents a real-valued variable.
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
}

impl<'t> Var<'t> {
    /// Sorts the expression graph in togological order starting from this variable.
    fn topological_sort(&self) -> Vec<usize> {
        let vars = self.tape.var_nodes.borrow();
        let mut visited = vec![false; vars.len()]; // flag visited nodes
        let mut visit_stack = Vec::with_capacity(vars.len()); // used to store the visited nodes
        let mut vars_stack = Vec::with_capacity(vars.len()); // used to store the traversal result
        let mut root = Some(self.index);

        loop {
            while let Some(root_index) = root {
                let root_node = &vars[root_index];
                match root_node {
                    VarNode::Constant(_) => {
                        visit_stack.push(root_index);
                        root = None;
                    }
                    VarNode::Nullary(_) => {
                        visit_stack.push(root_index);
                        root = None;
                    }
                    VarNode::Unary { dep, .. } => {
                        visit_stack.push(root_index);
                        root = Some(*dep);
                    }
                    VarNode::Binary { deps, .. } => {
                        visit_stack.push(deps[1]);
                        visit_stack.push(root_index);
                        root = Some(deps[0]);
                    }
                }
            }

            if let Some(root_index) = visit_stack.pop() {
                let root_node = &vars[root_index];
                let mut right_index = None;
                match root_node {
                    VarNode::Binary { deps, .. } => {
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
                        vars_stack.push(root_index);
                        visited[root_index] = true;
                    }
                }
            }

            if visit_stack.is_empty() {
                break;
            }
        }

        vars_stack
    }

    /// Evaluates the variable and those that it depends on.
    pub fn eval(&self) -> f64 {
        let vars_order = self.topological_sort();
        let mut vars = self.tape.var_nodes.borrow_mut();
        let mut vals = vec![0.0; self.tape.len()]; // used to store the computed value

        // applying the operators on the traversal results from left to right
        for &var_index in &vars_order {
            let node = &mut vars[var_index];
            match node {
                VarNode::Constant(value) => vals[var_index] = *value,
                VarNode::Nullary(_) => vals[var_index] = node.value(),
                VarNode::Unary { value, dep, op } => {
                    let val = vals[*dep];
                    let res = op.eval(val);
                    vals[var_index] = res;
                    *value = Some(res);
                }
                VarNode::Binary { value, deps, op } => {
                    let left_val = vals[deps[0]];
                    let right_val = vals[deps[1]];
                    let res = op.eval(left_val, right_val);
                    vals[var_index] = res;
                    *value = Some(res);
                }
            }
        }

        println!("{:?}", vars_order);
        println!("{:?}", vals);

        self.tape.is_evaluated.set(true);
        vals[self.index]
    }

    /// Sets the value of the variable.
    pub fn set(&self, new_value: f64) {
        // sets the value
        let mut vars = self.tape.var_nodes.borrow_mut();
        let current_var = &mut vars[self.index];
        match current_var {
            VarNode::Nullary(value) => {
                *value = Some(new_value);
            }
            _ => panic!("Cannot set value for non-input variable."),
        }
        // invalidate the tape
        self.tape.is_evaluated.set(false);
    }

    /// Computes the gradients of the variable with respects to all of its parameters.
    pub fn grad(&self) -> Grad {
        if !self.tape.is_evaluated.get() {
            panic!("Graph has not been evaluated.");
        }

        let vars_order = self.topological_sort();
        let vars = self.tape.var_nodes.borrow();
        let mut grads = self.tape.grad_nodes.borrow_mut();
        let mut derivs = vec![0.0; vars.len()];
        derivs[self.index] = 1.0;

        for &var_index in vars_order.iter().rev() {
            let node = &mut grads[var_index];
            let deriv = derivs[var_index];
            match node {
                GradNode::Nullary => {}
                GradNode::Unary { value, dep, op } => {
                    let grad_value = op.grad(&vars[*dep]);
                    derivs[*dep] += grad_value * deriv;
                    *value = Some(grad_value);
                }
                GradNode::Binary { values, deps, op } => {
                    let grad_values = op.grad(&vars[deps[0]], &vars[deps[1]]);
                    derivs[deps[0]] += grad_values[0] * deriv;
                    derivs[deps[1]] += grad_values[1] * deriv;
                    *values = Some(grad_values);
                }
            }
        }

        Grad { derivs }
    }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Sin),
        }
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Cos),
        }
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Tan),
        }
    }

    /// Takesthis variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Self {
        let const_index = self.tape.constant(p);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Pow),
        }
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: Var<'t>) -> Self {
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Pow),
        }
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Ln),
        }
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Self {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Exp),
        }
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Self {
        let const_index = self.tape.constant(base);
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Log),
        }
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: Var<'t>) -> Self {
        Var {
            tape: self.tape,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Log),
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

impl<'t> Neg for Var<'t> {
    type Output = Var<'t>;

    fn neg(self) -> Self::Output {
        Var {
            tape: self.tape,
            index: self.tape.push_unary(self.index, UnaryOp::Neg),
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
