use ndarray::{Array1, Array2, Ix1};
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
        unimplemented!()
    }

    fn cos(&self) -> Vector {
        unimplemented!()
    }

    fn tan(&self) -> Vector {
        unimplemented!()
    }

    fn ln(&self) -> Vector {
        unimplemented!()
    }

    fn exp(&self) -> Vector {
        unimplemented!()
    }

    fn pow(&self, power: &Vector) -> Vector {
        unimplemented!()
    }

    fn pow_scalar(&self, p: f64) -> Vector {
        unimplemented!()
    }

    fn log(&self, base: &Vector) -> Vector {
        unimplemented!()
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
    fn result_size(&self, size: usize) -> usize {
        size
    }

    /// Evaluates the operation with the given parameter.
    fn eval(&self, value: &Vector) -> Vector {
        match *self {
            UnaryOp::Neg => -value,
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Exp => value.exp(),
        }
    }

    /// Computes the gradient of the operation with respect to the given parameter.
    fn grad(&self, var: &VarNode) -> Vector {
        match var {
            VarNode::Constant(x) => x.zeros_like(),
            _ => {
                let value = var.value();
                match *self {
                    UnaryOp::Neg => -value.ones_like(),
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
    fn result_size(&self, left_size: usize, right_size: usize) -> usize {
        left_size
    }

    /// Evaluates the operation with the given parameters.
    fn eval(&self, left: &Vector, right: &Vector) -> Vector {
        match *self {
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
        match *self {
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
        match *self {
            BinaryOp::Add => right.ones_like(),
            BinaryOp::Sub => -right.ones_like(),
            BinaryOp::Mul => left.clone(),
            BinaryOp::Div => -left / right.pow_scalar(2.0),
            BinaryOp::Pow => left.ln() * left.pow(right),
            BinaryOp::Log => -left.ln() / (right.ln().pow_scalar(2.0) * right),
        }
    }

    /// Computes the full gradient of the operation with repsect to the given parameters.
    fn grad(&self, left: &VarNode, right: &VarNode) -> [Vector; 2] {
        let lg = {
            match left {
                VarNode::Constant(x) => x.zeros_like(),
                _ => self.left_grad(&left.value(), &right.value()),
            }
        };

        let rg = {
            match right {
                VarNode::Constant(x) => x.zeros_like(),
                _ => self.right_grad(&left.value(), &right.value()),
            }
        };

        [lg, rg]
    }
}

#[derive(Debug)]
/// Presents the operand in the graph.
enum VarNode {
    /// Represents a constant.
    Constant(Vector),
    /// Represents an input variable.
    Nullary(Option<Vector>),
    /// Represents the result of an unary operation.
    Unary {
        value: Option<Vector>, // the lazily computed value
        op: UnaryOp,           // the operation resulting in the node
        dep: usize,            // the operand for this node's operation
    },
    /// Represents the result of an binary operation.
    Binary {
        value: Option<Vector>, // the lazily computed value
        op: BinaryOp,          // the operation resulting in the node
        deps: [usize; 2],      // the operands for this node's operation
    },
}

impl VarNode {
    fn value(&self) -> &Vector {
        match self {
            VarNode::Constant(value) => value,
            VarNode::Nullary(value) => value
                .as_ref()
                .expect("Input variable has not been initialized."),
            VarNode::Unary { value, .. } => value
                .as_ref()
                .expect("Intermediate variable has not been evaluated."),
            VarNode::Binary { value, .. } => value
                .as_ref()
                .expect("Intermediate variable has not been evaluated."),
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
    pub fn var<'t>(&'t self, value: Vector) -> Var {
        let size = value.dim();
        let index = self.push_nullary(Some(value));
        Var {
            tape: self,
            index,
            size,
        }
    }

    /// Creates a node representing an unintialized input variable.
    pub fn var_unintialized<'t>(&'t self, size: usize) -> Var {
        let index = self.push_nullary(None);
        Var {
            tape: self,
            index,
            size,
        }
    }

    /// Creates a node representing a constant.
    fn constant_scalar(&self, value: f64, size: usize) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Constant(Vector::from_elem(size, value)));

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.grad_nodes.borrow().len()
    }

    /// Pushes a node representing an input variable onto the graph.
    fn push_nullary(&self, value: Option<Vector>) -> usize {
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
    size: usize,
    index: usize,
}

impl<'t> Var<'t> {
    /// Evaluates the variable and those that it is depends on.
    // pub fn eval(&self) -> f64 {
    //     // The basic idea is that we traverse the expression graph in postorder,
    //     // then apply the operators on the traversal result from left to right

    //     let mut vars = self.tape.var_nodes.borrow_mut();
    //     let mut visit_stack = Vec::new(); // used to store the visited nodes
    //     let mut var_stack = Vec::new(); // used to store the traversal result
    //     let mut val_stack = Vec::new(); // used to store the computed value
    //     let mut root = Some(self.index);

    //     // constructing the traversal result
    //     loop {
    //         while let Some(root_index) = root {
    //             let root_node = &vars[root_index];
    //             match *root_node {
    //                 VarNode::Constant(_) => {
    //                     visit_stack.push(root_index);
    //                     root = None;
    //                 }
    //                 VarNode::Nullary(_) => {
    //                     visit_stack.push(root_index);
    //                     root = None;
    //                 }
    //                 VarNode::Unary { dep, .. } => {
    //                     visit_stack.push(root_index);
    //                     root = Some(dep);
    //                 }
    //                 VarNode::Binary { deps, .. } => {
    //                     visit_stack.push(deps[1]);
    //                     visit_stack.push(root_index);
    //                     root = Some(deps[0]);
    //                 }
    //             }
    //         }

    //         if let Some(root_index) = visit_stack.pop() {
    //             let root_node = &vars[root_index];
    //             let mut right_index = None;
    //             match *root_node {
    //                 VarNode::Binary { deps, .. } => {
    //                     if let Some(top_index) = visit_stack.last() {
    //                         if *top_index == deps[1] {
    //                             right_index = Some(deps[1]);
    //                         }
    //                     }
    //                 }
    //                 _ => {}
    //             }
    //             if let Some(right_index) = right_index {
    //                 visit_stack.pop();
    //                 visit_stack.push(root_index);
    //                 root = Some(right_index);
    //             } else {
    //                 var_stack.push(root_index);
    //             }
    //         }

    //         if visit_stack.is_empty() {
    //             break;
    //         }
    //     }

    //     // applying the operators on the traversal results from left to right
    //     for var_index in &var_stack {
    //         let node = &mut vars[*var_index];
    //         match *node {
    //             VarNode::Constant(value) => val_stack.push(value),
    //             VarNode::Nullary(_) => val_stack.push(node.value()),
    //             VarNode::Unary {
    //                 op, ref mut value, ..
    //             } => {
    //                 let val = val_stack.pop().unwrap();
    //                 let res = op.eval(val);
    //                 val_stack.push(res);
    //                 *value = Some(res);
    //             }
    //             VarNode::Binary {
    //                 op, ref mut value, ..
    //             } => {
    //                 let right = val_stack.pop().unwrap();
    //                 let left = val_stack.pop().unwrap();
    //                 let res = op.eval(left, right);
    //                 val_stack.push(res);
    //                 *value = Some(res);
    //             }
    //         }
    //     }

    //     // println!("{:?}", exp_stack);
    //     self.tape.is_evaluated.set(true);
    //     val_stack.pop().unwrap()
    // }

    // /// Sets the value of the variable.
    // pub fn set(&self, new_value: f64) {
    //     // sets the value
    //     let mut vars = self.tape.var_nodes.borrow_mut();
    //     let current_var = &mut vars[self.index];
    //     match *current_var {
    //         VarNode::Nullary(ref mut value) => {
    //             *value = Some(new_value);
    //         }
    //         _ => panic!("Cannot set value for non-input variable."),
    //     }
    //     self.tape.is_evaluated.set(false);
    // }

    // /// Computes the gradients of the variable with respects to all of its parameters.
    // pub fn grad(&self) -> Grad {
    //     if !self.tape.is_evaluated.get() {
    //         panic!("Graph has not been evaluated");
    //     }

    //     let len = self.tape.len();
    //     let mut grads = self.tape.grad_nodes.borrow_mut();
    //     let vars = self.tape.var_nodes.borrow();
    //     let mut derivs = vec![0.0; len];
    //     derivs[self.index] = 1.0;

    //     for i in (0..len).rev() {
    //         let node = &mut grads[i];
    //         let deriv = derivs[i];
    //         match *node {
    //             GradNode::Nullary => {}
    //             GradNode::Unary {
    //                 ref mut value,
    //                 dep,
    //                 op,
    //             } => {
    //                 let grad_value = op.grad(&vars[dep]);
    //                 derivs[dep] += grad_value * deriv;
    //                 *value = Some(grad_value);
    //             }
    //             GradNode::Binary {
    //                 ref mut values,
    //                 deps,
    //                 op,
    //             } => {
    //                 let grad_values = op.grad(&vars[deps[0]], &vars[deps[1]]);
    //                 derivs[deps[0]] += grad_values[0] * deriv;
    //                 derivs[deps[1]] += grad_values[1] * deriv;
    //                 *values = Some(grad_values);
    //             }
    //         }
    //     }

    //     Grad { derivs }
    // }

    /// Takes the sine of this variable.
    pub fn sin(&self) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
            index: self.tape.push_unary(self.index, UnaryOp::Sin),
        }
    }

    /// Takes the cosine of this variable.
    pub fn cos(&self) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
            index: self.tape.push_unary(self.index, UnaryOp::Cos),
        }
    }

    /// Takes the tangent of this variable.
    pub fn tan(&self) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
            index: self.tape.push_unary(self.index, UnaryOp::Tan),
        }
    }

    /// Takes this variable raised to a given constant power.
    pub fn pow_const(&self, p: f64) -> Self {
        let const_index = self.tape.constant_scalar(p, self.size);
        Var {
            tape: self.tape,
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Pow),
        }
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: Var<'t>) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Pow),
        }
    }

    /// Takes the natural logarithm of this variable.
    pub fn ln(&self) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
            index: self.tape.push_unary(self.index, UnaryOp::Ln),
        }
    }

    /// Takes the natural exponential of this variable.
    pub fn exp(&self) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
            index: self.tape.push_unary(self.index, UnaryOp::Exp),
        }
    }

    /// Takes the log of this variable with a constant base.
    pub fn log_const(&self, base: f64) -> Self {
        let const_index = self.tape.constant_scalar(base, self.size);
        Var {
            tape: self.tape,
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Log),
        }
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: Var<'t>) -> Self {
        Var {
            tape: self.tape,
            size: self.size,
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
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Add),
        }
    }
}

impl<'t> Add<f64> for Var<'t> {
    type Output = Self;

    fn add(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant_scalar(constant, self.size);
        Var {
            tape: self.tape,
            size: self.size,
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
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Mul),
        }
    }
}

impl<'t> Mul<f64> for Var<'t> {
    type Output = Self;

    fn mul(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant_scalar(constant, self.size);
        Var {
            tape: self.tape,
            size: self.size,
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
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Sub),
        }
    }
}

impl<'t> Sub<f64> for Var<'t> {
    type Output = Var<'t>;

    fn sub(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant_scalar(constant, self.size);
        Var {
            tape: self.tape,
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Sub),
        }
    }
}

impl<'t> Sub<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn sub(self, var: Var<'t>) -> Self::Output {
        let const_index = var.tape.constant_scalar(self, var.size);
        Var {
            tape: var.tape,
            size: var.size,
            index: var.tape.push_binary(const_index, var.index, BinaryOp::Sub),
        }
    }
}

impl<'t> Div<Var<'t>> for Var<'t> {
    type Output = Var<'t>;

    fn div(self, other: Var<'t>) -> Self::Output {
        Var {
            tape: self.tape,
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, other.index, BinaryOp::Div),
        }
    }
}

impl<'t> Div<f64> for Var<'t> {
    type Output = Var<'t>;

    fn div(self, constant: f64) -> Self::Output {
        let const_index = self.tape.constant_scalar(constant, self.size);
        Var {
            tape: self.tape,
            size: self.size,
            index: self
                .tape
                .push_binary(self.index, const_index, BinaryOp::Div),
        }
    }
}

impl<'t> Div<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn div(self, var: Var<'t>) -> Self::Output {
        let const_index = var.tape.constant_scalar(self, var.size);
        Var {
            tape: var.tape,
            size: var.size,
            index: var.tape.push_binary(const_index, var.index, BinaryOp::Div),
        }
    }
}

impl<'t> Neg for Var<'t> {
    type Output = Var<'t>;

    fn neg(self) -> Self::Output {
        Var {
            tape: self.tape,
            size: self.size,
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
