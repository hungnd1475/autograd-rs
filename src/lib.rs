use ndarray::{Array1, Array2};
use std::cell::{Cell, RefCell};
use std::ops::{Add, Div, Mul, Neg, Sub};

type Vector = Array1<f64>;
type Matrix = Array2<f64>;

pub trait VectorExt {
    fn zeros_like(&self) -> Vector;
    fn ones_like(&self) -> Vector;

    fn as_column(self) -> Matrix;
    fn as_row(self) -> Matrix;

    fn sin(&self) -> Vector;
    fn cos(&self) -> Vector;
    fn tan(&self) -> Vector;
    fn ln(&self) -> Vector;
    fn exp(&self) -> Vector;

    fn pow(&self, power: &Vector) -> Vector;
    fn pow_scalar(&self, p: f64) -> Vector;
    fn log(&self, base: &Vector) -> Vector;
}

pub trait MatrixExt {
    fn to_vector(self) -> Result<Vector, &'static str>;
}

impl VectorExt for Vector {
    fn zeros_like(&self) -> Vector {
        Vector::zeros(self.dim())
    }

    fn ones_like(&self) -> Vector {
        Vector::ones(self.dim())
    }

    fn as_column(self) -> Matrix {
        unimplemented!()
    }

    fn as_row(self) -> Matrix {
        unimplemented!()
    }

    fn sin(&self) -> Vector {
        self.mapv(|x| x.sin())
    }

    fn cos(&self) -> Vector {
        self.mapv(|x| x.cos())
    }

    fn tan(&self) -> Vector {
        self.mapv(|x| x.tan())
    }

    fn ln(&self) -> Vector {
        self.mapv(|x| x.ln())
    }

    fn exp(&self) -> Vector {
        self.mapv(|x| x.exp())
    }

    fn pow(&self, power: &Vector) -> Vector {
        let mut result = self.clone();
        result.zip_mut_with(power, |x, y| *x = x.powf(*y));
        result
    }

    fn pow_scalar(&self, p: f64) -> Vector {
        let power = Vector::from_elem(self.dim(), p);
        self.pow(&power)
    }

    fn log(&self, base: &Vector) -> Vector {
        let mut result = self.clone();
        result.zip_mut_with(base, |x, y| *x = x.log(*y));
        result
    }
}

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
    fn compute_shape(&self, size: usize) -> usize {
        size
    }

    /// Evaluates the operation with the given parameter.
    fn eval(&self, value: &Vector) -> Vector {
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
    fn grad(&self, var: (&VarNode, &Vector)) -> Vector {
        let (node, val) = var;
        match node {
            VarNode::Constant(size) => Vector::zeros(*size),
            _ => match self {
                UnaryOp::Neg => -val.ones_like(),
                UnaryOp::Sin => val.cos(),
                UnaryOp::Cos => -val.sin(),
                UnaryOp::Tan => 2.0 / ((2.0 * val).cos() + 1.0),
                UnaryOp::Ln => 1.0 / val,
                UnaryOp::Exp => val.exp(),
            },
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
    fn compute_shape(&self, left_size: usize, right_size: usize) -> usize {
        left_size
    }

    /// Evaluates the operation with the given parameters.
    fn eval(&self, left: &Vector, right: &Vector) -> Vector {
        match self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
            BinaryOp::Pow => left.pow(right),
            BinaryOp::Log => left.log(right),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the left parameter.
    fn left_grad(&self, left: &Vector, right: &Vector) -> Vector {
        match self {
            BinaryOp::Add => left.ones_like(),
            BinaryOp::Sub => left.ones_like(),
            BinaryOp::Mul => right.clone(),
            BinaryOp::Div => 1.0 / right,
            BinaryOp::Pow => right * &left.pow(&(right - 1.0)),
            BinaryOp::Log => 1.0 / (left * &right.ln()),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the right parameter.
    fn right_grad(&self, left: &Vector, right: &Vector) -> Vector {
        match self {
            BinaryOp::Add => right.ones_like(),
            BinaryOp::Sub => -right.ones_like(),
            BinaryOp::Mul => left.clone(),
            BinaryOp::Div => -left / right.pow_scalar(2.0),
            BinaryOp::Pow => left.ln() * left.pow(right),
            BinaryOp::Log => -left.ln() / (right.ln().pow_scalar(2.0) * right),
        }
    }

    /// Computes the full gradient of the operation with repsect to the given parameters.
    fn grad(&self, left_var: (&VarNode, &Vector), right_var: (&VarNode, &Vector)) -> [Vector; 2] {
        let (left_node, left_val) = left_var;
        let (right_node, right_val) = right_var;

        let lg = {
            match left_node {
                VarNode::Constant(size) => Vector::zeros(*size),
                _ => self.left_grad(left_val, right_val),
            }
        };

        let rg = {
            match right_node {
                VarNode::Constant(size) => Vector::zeros(*size),
                _ => self.right_grad(left_val, right_val),
            }
        };

        [lg, rg]
    }
}

#[derive(Debug)]
/// Presents the operand in the graph.
enum VarNode {
    /// Represents a constant.
    Constant(usize),
    /// Represents an input variable.
    Nullary(usize),
    /// Represents the result of an unary operation.
    Unary {
        size: usize, // the lazily computed value
        op: UnaryOp, // the operation resulting in the node
        dep: usize,  // the operand for this node's operation
    },
    /// Represents the result of an binary operation.
    Binary {
        size: usize,      // the lazily computed value
        op: BinaryOp,     // the operation resulting in the node
        deps: [usize; 2], // the operands for this node's operation
    },
}

impl VarNode {
    fn size(&self) -> usize {
        match self {
            VarNode::Constant(size) => *size,
            VarNode::Nullary(size) => *size,
            VarNode::Unary { size, .. } | VarNode::Binary { size, .. } => *size,
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
        value: Option<Vector>, // the lazily computed gradient
        dep: usize,            // the operands for this node's operation
        op: UnaryOp,           // the operation resulting in this node
    },
    /// Represents the gradients of a binary operation.
    Binary {
        values: Option<[Vector; 2]>, // the lazily computed gradients
        deps: [usize; 2],            // the operands for this node's operation
        op: BinaryOp,                // the operation resulting in this node
    },
}

/// Represents a tape used to construct the adjoint graph.
pub struct Tape {
    grad_nodes: RefCell<Vec<GradNode>>,
    var_nodes: RefCell<Vec<VarNode>>,
    var_values: RefCell<Vec<Option<Vector>>>,
    is_evaluated: Cell<bool>,
}

impl Tape {
    /// Initializes a new tape.
    pub fn new() -> Self {
        Self {
            grad_nodes: RefCell::new(Vec::new()),
            var_nodes: RefCell::new(Vec::new()),
            var_values: RefCell::new(Vec::new()),
            is_evaluated: Cell::new(false),
        }
    }

    /// Creates a node representing an input variable.
    pub fn var<'t>(&'t self, value: Vec<f64>) -> Var {
        let value = Vector::from_shape_vec(value.len(), value).unwrap();
        let size = value.dim();
        let index = self.push_nullary(Some(value), size);
        Var {
            tape: self,
            index,
            size,
        }
    }

    /// Creates a node representing an unintialized input variable.
    pub fn var_uninitialized<'t>(&'t self, size: usize) -> Var {
        let index = self.push_nullary(None, size);
        Var {
            tape: self,
            size,
            index,
        }
    }

    /// Creates a node representing a constant.
    fn constant_scalar(&self, value: f64, size: usize) -> Var {
        let value = Vector::from_elem(size, value);
        Var {
            tape: self,
            size,
            index: self.push_constant(value),
        }
    }

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.grad_nodes.borrow().len()
    }

    fn push_constant(&self, value: Vector) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Constant(value.dim()));

        let mut vals = self.var_values.borrow_mut();
        vals.push(Some(value));

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Pushes a node representing an input variable onto the graph.
    fn push_nullary(&self, value: Option<Vector>, size: usize) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Nullary(size));

        let mut vals = self.var_values.borrow_mut();
        vals.push(value);

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Pushes a node representing the result of an unary operator onto the graph.
    fn push_unary(&self, size: usize, dep: usize, op: UnaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Unary { size, dep, op });

        let mut vals = self.var_values.borrow_mut();
        vals.push(None);

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Unary {
            value: None,
            dep,
            op,
        });
        len
    }

    /// Pushes a node representing the result of a binary operator onto the graph.
    fn push_binary(&self, size: usize, dep0: usize, dep1: usize, op: BinaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Binary {
            size,
            deps: [dep0, dep1],
            op,
        });

        let mut vals = self.var_values.borrow_mut();
        vals.push(None);

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Binary {
            values: None,
            deps: [dep0, dep1],
            op,
        });
        len
    }
}

#[derive(Clone, Copy)]
/// Represents a real-valued variable.
pub struct Var<'t> {
    tape: &'t Tape,
    size: usize,
    index: usize,
}

impl<'t> Var<'t> {
    fn unary(&self, op: UnaryOp) -> Self {
        let size = op.compute_shape(self.size);
        Var {
            tape: self.tape,
            size,
            index: self.tape.push_unary(size, self.index, op),
        }
    }

    fn binary(&self, other: &Var<'t>, op: BinaryOp) -> Self {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        let size = op.compute_shape(self.size, other.size);
        Var {
            tape: self.tape,
            size,
            index: self.tape.push_binary(size, self.index, other.index, op),
        }
    }
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
    pub fn eval(&self) -> Vector {
        let vars_order = self.topological_sort();
        let vars = self.tape.var_nodes.borrow();
        let mut vals = self.tape.var_values.borrow_mut();

        // applying the operators on the traversal results from left to right
        for &var_index in &vars_order {
            let node = &vars[var_index];
            match node {
                VarNode::Constant(_) | VarNode::Nullary(_) => {}
                VarNode::Unary { dep, op, .. } => {
                    let res = op.eval(vals[*dep].as_ref().unwrap());
                    vals[var_index] = Some(res);
                }
                VarNode::Binary { deps, op, .. } => {
                    let res = op.eval(
                        vals[deps[0]].as_ref().unwrap(),
                        vals[deps[1]].as_ref().unwrap(),
                    );
                    vals[var_index] = Some(res);
                }
            }
        }

        // println!("{:?}", vars_order);
        // println!("{:?}", vals);

        self.tape.is_evaluated.set(true);
        vals[self.index].as_ref().unwrap().clone()
    }

    /// Sets the value of the variable.
    pub fn set(&self, new_value: Vector) {
        // sets the value
        let vars = self.tape.var_nodes.borrow();
        let mut vals = self.tape.var_values.borrow_mut();
        match &vars[self.index] {
            VarNode::Nullary(_) => vals[self.index] = Some(new_value),
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
        let vals = self.tape.var_values.borrow();
        let mut grads = self.tape.grad_nodes.borrow_mut();
        let mut derivs: Vec<Vector> = vars.iter().map(|x| Vector::zeros(x.size())).collect();
        derivs[self.index] = derivs[self.index].ones_like();

        for &var_index in vars_order.iter().rev() {
            let node = &mut grads[var_index];
            let deriv = derivs[var_index].clone();
            match node {
                GradNode::Nullary => {}
                GradNode::Unary { value, dep, op } => {
                    let dep_node = &vars[*dep];
                    let dep_val = vals[*dep].as_ref().unwrap();
                    let grad_value = op.grad((dep_node, dep_val));
                    derivs[*dep] = &derivs[*dep] + &(&grad_value * &deriv);
                    *value = Some(grad_value);
                }
                GradNode::Binary { values, deps, op } => {
                    let left = (&vars[deps[0]], vals[deps[0]].as_ref().unwrap());
                    let right = (&vars[deps[1]], vals[deps[1]].as_ref().unwrap());
                    let grad_values = op.grad(left, right);
                    derivs[deps[0]] = &derivs[deps[0]] + &(&grad_values[0] * &deriv);
                    derivs[deps[1]] = &derivs[deps[1]] + &(&grad_values[1] * &deriv);
                    *values = Some(grad_values);
                }
            }
        }

        Grad { derivs }
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
        let const_var = self.tape.constant_scalar(p, self.size);
        self.binary(&const_var, BinaryOp::Pow)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: Var<'t>) -> Self {
        self.binary(&other, BinaryOp::Pow)
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
        let const_var = self.tape.constant_scalar(base, self.size);
        self.binary(&const_var, BinaryOp::Log)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: Var<'t>) -> Self {
        self.binary(&other, BinaryOp::Log)
    }
}

impl<'t> Add<Var<'t>> for Var<'t> {
    type Output = Self;

    fn add(self, other: Var<'t>) -> Self::Output {
        self.binary(&other, BinaryOp::Add)
    }
}

impl<'t> Add<f64> for Var<'t> {
    type Output = Self;

    fn add(self, constant: f64) -> Self::Output {
        let const_var = self.tape.constant_scalar(constant, self.size);
        self.binary(&const_var, BinaryOp::Add)
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
        self.binary(&other, BinaryOp::Mul)
    }
}

impl<'t> Mul<f64> for Var<'t> {
    type Output = Self;

    fn mul(self, constant: f64) -> Self::Output {
        let const_var = self.tape.constant_scalar(constant, self.size);
        self.binary(&const_var, BinaryOp::Mul)
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
        self.binary(&other, BinaryOp::Sub)
    }
}

impl<'t> Sub<f64> for Var<'t> {
    type Output = Var<'t>;

    fn sub(self, constant: f64) -> Self::Output {
        let const_var = self.tape.constant_scalar(constant, self.size);
        self.binary(&const_var, BinaryOp::Sub)
    }
}

impl<'t> Sub<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn sub(self, var: Var<'t>) -> Self::Output {
        let const_var = var.tape.constant_scalar(self, var.size);
        const_var.binary(&var, BinaryOp::Sub)
    }
}

impl<'t> Div<Var<'t>> for Var<'t> {
    type Output = Var<'t>;

    fn div(self, other: Var<'t>) -> Self::Output {
        self.binary(&other, BinaryOp::Div)
    }
}

impl<'t> Div<f64> for Var<'t> {
    type Output = Var<'t>;

    fn div(self, constant: f64) -> Self::Output {
        let const_var = self.tape.constant_scalar(constant, self.size);
        self.binary(&const_var, BinaryOp::Div)
    }
}

impl<'t> Div<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn div(self, var: Var<'t>) -> Self::Output {
        let const_var = var.tape.constant_scalar(self, var.size);
        const_var.binary(&var, BinaryOp::Div)
    }
}

impl<'t> Neg for Var<'t> {
    type Output = Var<'t>;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

pub struct Grad {
    derivs: Vec<Vector>,
}

impl Grad {
    pub fn wrt<'t>(&self, var: Var<'t>) -> &Vector {
        &self.derivs[var.index]
    }
}
