use ndarray::Array2;
use std::cell::{Cell, RefCell};
use std::ops::{Add, Div, Mul, Neg, Sub};

type Matrix = Array2<f64>;
type Shape = (usize, usize);

pub trait MatrixExt {
    fn zeros_like(&self) -> Matrix;
    fn ones_like(&self) -> Matrix;

    fn sin(&self) -> Matrix;
    fn cos(&self) -> Matrix;
    fn tan(&self) -> Matrix;
    fn ln(&self) -> Matrix;
    fn exp(&self) -> Matrix;

    fn pow(&self, power: &Matrix) -> Matrix;
    fn pow_scalar(&self, p: f64) -> Matrix;
    fn log(&self, base: &Matrix) -> Matrix;
}

impl MatrixExt for Matrix {
    fn zeros_like(&self) -> Matrix {
        Matrix::zeros(self.dim())
    }

    fn ones_like(&self) -> Matrix {
        Matrix::ones(self.dim())
    }

    fn sin(&self) -> Matrix {
        self.mapv(|x| x.sin())
    }

    fn cos(&self) -> Matrix {
        self.mapv(|x| x.cos())
    }

    fn tan(&self) -> Matrix {
        self.mapv(|x| x.tan())
    }

    fn ln(&self) -> Matrix {
        self.mapv(|x| x.ln())
    }

    fn exp(&self) -> Matrix {
        self.mapv(|x| x.exp())
    }

    fn pow(&self, power: &Matrix) -> Matrix {
        let mut result = self.clone();
        result.zip_mut_with(power, |x, y| *x = x.powf(*y));
        result
    }

    fn pow_scalar(&self, p: f64) -> Matrix {
        let power = Matrix::from_elem(self.dim(), p);
        self.pow(&power)
    }

    fn log(&self, base: &Matrix) -> Matrix {
        let mut result = self.clone();
        result.zip_mut_with(base, |x, y| *x = x.log(*y));
        result
    }
}

#[derive(Clone, Copy, Debug)]
/// Represents the supported unary operations.
enum UnaryOp {
    T,
    Neg,
    Sin,
    Cos,
    Tan,
    Ln,
    Exp,
}

impl UnaryOp {
    fn eval_shape(&self, input_shape: Shape) -> Shape {
        input_shape
    }

    fn grad_shape(&self, input_shape: Shape) -> Shape {
        input_shape
    }

    /// Evaluates the operation with the given parameter.
    fn eval(&self, value: &Matrix) -> Matrix {
        match self {
            UnaryOp::T => value.t().to_owned(),
            UnaryOp::Neg => -value,
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Exp => value.exp(),
        }
    }

    /// Computes the gradient of the operation with respect to the given parameter.
    fn grad(&self, var: (&VarNode, &Matrix), ans: &Matrix, g: &Matrix) -> Matrix {
        let (node, val) = var;
        match node {
            VarNode::Constant(size) => Matrix::zeros(*size),
            _ => match self {
                UnaryOp::T => g.t().to_owned(),
                UnaryOp::Neg => -g,
                UnaryOp::Sin => val.cos() * g,
                UnaryOp::Cos => -val.sin() * g,
                UnaryOp::Tan => 2.0 * g / ((2.0 * val).cos() + 1.0),
                UnaryOp::Ln => g / val,
                UnaryOp::Exp => ans * g,
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
    Dot,
}

impl BinaryOp {
    fn eval_shape(&self, left_shape: Shape, right_shape: Shape) -> Shape {
        match self {
            BinaryOp::Dot => (left_shape.0, right_shape.1),
            _ => left_shape,
        }
    }

    fn grad_shape(&self, left_shape: Shape, right_shape: Shape) -> (Shape, Shape) {
        match self {
            BinaryOp::Dot => (right_shape, left_shape),
            _ => (left_shape, right_shape),
        }
    }

    /// Evaluates the operation with the given parameters.
    fn eval(&self, left: &Matrix, right: &Matrix) -> Matrix {
        match self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
            BinaryOp::Pow => left.pow(right),
            BinaryOp::Log => left.log(right),
            BinaryOp::Dot => left.dot(right),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the left parameter.
    fn left_grad(&self, left: &Matrix, right: &Matrix, ans: &Matrix, g: &Matrix) -> Matrix {
        match self {
            BinaryOp::Add => g.clone(),
            BinaryOp::Sub => g.clone(),
            BinaryOp::Mul => right * g,
            BinaryOp::Div => g / right,
            BinaryOp::Pow => right * &left.pow(&(right - 1.0)) * g,
            BinaryOp::Log => g / &(left * &right.ln()),
            BinaryOp::Dot => g.dot(&right.t()),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the right parameter.
    fn right_grad(&self, left: &Matrix, right: &Matrix, ans: &Matrix, g: &Matrix) -> Matrix {
        match self {
            BinaryOp::Add => g.clone(),
            BinaryOp::Sub => -g,
            BinaryOp::Mul => left * g,
            BinaryOp::Div => -left * g / right.pow_scalar(2.0),
            BinaryOp::Pow => left.ln() * left.pow(right) * g,
            BinaryOp::Log => -left.ln() * g / (right.ln().pow_scalar(2.0) * right),
            BinaryOp::Dot => left.t().dot(g),
        }
    }

    /// Computes the full gradient of the operation with repsect to the given parameters.
    fn grad(
        &self,
        left_var: (&VarNode, &Matrix),
        right_var: (&VarNode, &Matrix),
        ans: &Matrix,
        g: &Matrix,
    ) -> [Matrix; 2] {
        let (left_node, left_val) = left_var;
        let (right_node, right_val) = right_var;

        let lg = {
            match left_node {
                VarNode::Constant(shape) => Matrix::zeros(*shape),
                _ => self.left_grad(left_val, right_val, ans, g),
            }
        };

        let rg = {
            match right_node {
                VarNode::Constant(shape) => Matrix::zeros(*shape),
                _ => self.right_grad(left_val, right_val, ans, g),
            }
        };

        [lg, rg]
    }
}

#[derive(Debug)]
/// Presents the operand in the graph.
enum VarNode {
    /// Represents a constant.
    Constant(Shape),
    /// Represents an input variable.
    Nullary(Shape),
    /// Represents the result of an unary operation.
    Unary {
        shape: Shape, // the lazily computed value
        op: UnaryOp,  // the operation resulting in the node
        dep: usize,   // the operand for this node's operation
    },
    /// Represents the result of an binary operation.
    Binary {
        shape: Shape,     // the lazily computed value
        op: BinaryOp,     // the operation resulting in the node
        deps: [usize; 2], // the operands for this node's operation
    },
}

impl VarNode {
    fn shape(&self) -> Shape {
        match self {
            VarNode::Constant(shape) | VarNode::Nullary(shape) => *shape,
            VarNode::Unary { shape, .. } | VarNode::Binary { shape, .. } => *shape,
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
        value: Option<Matrix>, // the lazily computed gradient
        dep: usize,            // the operands for this node's operation
        op: UnaryOp,           // the operation resulting in this node
    },
    /// Represents the gradients of a binary operation.
    Binary {
        values: Option<[Matrix; 2]>, // the lazily computed gradients
        deps: [usize; 2],            // the operands for this node's operation
        op: BinaryOp,                // the operation resulting in this node
    },
}

/// Represents a tape used to construct the adjoint graph.
pub struct Tape {
    grad_nodes: RefCell<Vec<GradNode>>,
    var_nodes: RefCell<Vec<VarNode>>,
    var_values: RefCell<Vec<Option<Matrix>>>,
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

    pub fn scalar_var<'t>(&'t self) -> ScalarVar<'t> {
        let index = self.push_nullary(None, (1, 1));
        ScalarVar { tape: self, index }
    }

    /// Creates a node representing a vector input variable.
    pub fn vector_var<'t>(&'t self, length: usize) -> VectorVar<'t> {
        let shape = (length, 1);
        let index = self.push_nullary(None, shape);
        VectorVar {
            tape: self,
            index,
            shape,
        }
    }

    /// Creates a node representing a matrix input variable.
    pub fn matrix_var<'t>(&'t self, nrow: usize, ncol: usize) -> MatrixVar<'t> {
        let shape = (nrow, ncol);
        let index = self.push_nullary(None, shape);
        MatrixVar {
            tape: self,
            index,
            shape,
        }
    }

    /// Creates a node representing a constant.
    fn scalar_const<'t>(&'t self, value: f64) -> ScalarVar<'t> {
        let value = Matrix::from_elem((1, 1), value);
        let index = self.push_constant(value);
        ScalarVar { tape: self, index }
    }

    fn vector_const<'t>(&'t self, value: Vec<f64>) -> VectorVar<'t> {
        let shape = (value.len(), 1);
        let value = Matrix::from_shape_vec(shape, value).unwrap();
        let index = self.push_constant(value);
        VectorVar {
            tape: self,
            shape,
            index,
        }
    }

    fn matrix_const<'t>(&'t self, value: Vec<f64>, nrow: usize, ncol: usize) -> MatrixVar<'t> {
        let shape = (nrow, ncol);
        let value = Matrix::from_shape_vec(shape, value).unwrap();
        let index = self.push_constant(value);
        MatrixVar {
            tape: self,
            shape,
            index,
        }
    }

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.grad_nodes.borrow().len()
    }

    fn push_constant(&self, value: Matrix) -> usize {
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
    fn push_nullary(&self, value: Option<Matrix>, shape: Shape) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Nullary(shape));

        let mut vals = self.var_values.borrow_mut();
        vals.push(value);

        let mut grads = self.grad_nodes.borrow_mut();
        grads.push(GradNode::Nullary);
        len
    }

    /// Pushes a node representing the result of an unary operator onto the graph.
    fn push_unary(&self, shape: Shape, dep: usize, op: UnaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Unary { shape, dep, op });

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
    fn push_binary(&self, shape: Shape, dep0: usize, dep1: usize, op: BinaryOp) -> usize {
        let mut vars = self.var_nodes.borrow_mut();
        let len = vars.len();
        vars.push(VarNode::Binary {
            shape,
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
pub struct MatrixVar<'t> {
    tape: &'t Tape,
    shape: Shape,
    index: usize,
}

#[derive(Clone, Copy)]
/// Represents a real-valued variable.
pub struct VectorVar<'t> {
    tape: &'t Tape,
    shape: Shape,
    index: usize,
}

#[derive(Clone, Copy)]
pub struct ScalarVar<'t> {
    tape: &'t Tape,
    index: usize,
}

impl<'t> VectorVar<'t> {
    fn unary_vector(&self, op: UnaryOp) -> VectorVar<'t> {
        let shape = op.eval_shape(self.shape);
        VectorVar {
            tape: self.tape,
            index: self.tape.push_unary(shape, self.index, op),
            shape,
        }
    }

    fn unary_scalar(&self, op: UnaryOp) -> ScalarVar<'t> {
        ScalarVar {
            tape: self.tape,
            index: self.tape.push_unary((1, 1), self.index, op),
        }
    }

    fn binary_vector(&self, other: &VectorVar<'t>, length: usize) -> VectorVar<'t> {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        VectorVar {
            tape: self.tape,
            index: self
                .tape
                .push_binary((length, 1), self.index, other.index, op),
            length,
        }
    }

    /// Sets the value of the variable.
    pub fn set(&self, value: Vec<f64>) {
        let value = Matrix::from_shape_vec((self.shape, 1), value).unwrap();
        let vars = self.tape.var_nodes.borrow();
        let mut vals = self.tape.var_values.borrow_mut();
        match &vars[self.index] {
            VarNode::Nullary(_) => vals[self.index] = Some(value),
            _ => panic!("Cannot set value for non-input variable."),
        }
        // invalidate the tape
        self.tape.is_evaluated.set(false);
    }
}

impl<'t> VectorVar<'t> {
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
    pub fn eval(&self) -> Matrix {
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

    /// Computes the gradients of the variable with respects to all of its parameters.
    pub fn grad(&self) -> Grad {
        if !self.tape.is_evaluated.get() {
            panic!("Graph has not been evaluated.");
        }

        let vars_order = self.topological_sort();
        let vars = self.tape.var_nodes.borrow();
        let vals = self.tape.var_values.borrow();
        let mut grads = self.tape.grad_nodes.borrow_mut();
        let mut derivs: Vec<Matrix> = vars.iter().map(|x| Matrix::zeros(x.shape())).collect();
        derivs[self.index] = derivs[self.index].ones_like();

        for &var_index in vars_order.iter().rev() {
            let node = &mut grads[var_index];
            println!("var = {}", var_index);
            match node {
                GradNode::Nullary => {}
                GradNode::Unary { value, dep, op } => {
                    println!("deriv = {}", derivs[var_index]);
                    println!("derive_dep = {}", derivs[*dep]);
                    let dep_grad = {
                        let var_val = vals[var_index].as_ref().unwrap();
                        let var_grad = &derivs[var_index];
                        let dep_node = &vars[*dep];
                        let dep_val = vals[*dep].as_ref().unwrap();
                        op.grad((dep_node, dep_val), var_val, var_grad)
                    };
                    println!("grad = {}", dep_grad);
                    derivs[*dep] = &derivs[*dep] + &dep_grad;
                    *value = Some(dep_grad);
                }
                GradNode::Binary { values, deps, op } => {
                    println!("g = {}", derivs[var_index]);
                    println!("deriv_dep = {}; {}", derivs[deps[0]], derivs[deps[1]]);
                    let dep_grads = {
                        let var_val = vals[var_index].as_ref().unwrap();
                        let var_grad = &derivs[var_index];
                        let left = (&vars[deps[0]], vals[deps[0]].as_ref().unwrap());
                        let right = (&vars[deps[1]], vals[deps[1]].as_ref().unwrap());
                        op.grad(left, right, var_val, var_grad)
                    };
                    println!("grad = {}; {}", dep_grads[0], dep_grads[1]);
                    derivs[deps[0]] = &derivs[deps[0]] + &dep_grads[0];
                    derivs[deps[1]] = &derivs[deps[1]] + &dep_grads[1];
                    *values = Some(dep_grads);
                }
            }
            println!("---")
        }

        Grad { derivs }
    }

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
        let const_var = self.tape.scalar_const(p, self.shape);
        self.binary_vector(&const_var, BinaryOp::Pow)
    }

    /// Takes this variable raised to a given variable power.
    pub fn pow(&self, other: VectorVar<'t>) -> Self {
        self.binary_vector(&other, BinaryOp::Pow)
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
        let const_var = self.tape.scalar_const(base, self.shape);
        self.binary_vector(&const_var, BinaryOp::Log)
    }

    /// Takes the log of this variable with a variable base.
    pub fn log(&self, other: VectorVar<'t>) -> Self {
        self.binary_vector(&other, BinaryOp::Log)
    }

    pub fn dot(&self, other: VectorVar<'t>) -> Self {
        self.binary_vector(&other, BinaryOp::Dot)
    }
}

impl<'t> Add<VectorVar<'t>> for VectorVar<'t> {
    type Output = Self;

    fn add(self, other: VectorVar<'t>) -> Self::Output {
        self.binary_vector(&other, BinaryOp::Add)
    }
}

impl<'t> Add<f64> for VectorVar<'t> {
    type Output = Self;

    fn add(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant, self.shape);
        self.binary_vector(&const_var, BinaryOp::Add)
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
        self.binary_vector(&other, BinaryOp::Mul)
    }
}

impl<'t> Mul<f64> for VectorVar<'t> {
    type Output = Self;

    fn mul(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant, self.shape);
        self.binary_vector(&const_var, BinaryOp::Mul)
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
        self.binary_vector(&other, BinaryOp::Sub)
    }
}

impl<'t> Sub<f64> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn sub(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant, self.shape);
        self.binary_vector(&const_var, BinaryOp::Sub)
    }
}

impl<'t> Sub<VectorVar<'t>> for f64 {
    type Output = VectorVar<'t>;

    fn sub(self, var: VectorVar<'t>) -> Self::Output {
        let const_var = var.tape.scalar_const(self, var.shape);
        const_var.binary(&var, BinaryOp::Sub)
    }
}

impl<'t> Div<VectorVar<'t>> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn div(self, other: VectorVar<'t>) -> Self::Output {
        self.binary_vector(&other, BinaryOp::Div)
    }
}

impl<'t> Div<f64> for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn div(self, constant: f64) -> Self::Output {
        let const_var = self.tape.scalar_const(constant, self.shape);
        self.binary_vector(&const_var, BinaryOp::Div)
    }
}

impl<'t> Div<VectorVar<'t>> for f64 {
    type Output = VectorVar<'t>;

    fn div(self, var: VectorVar<'t>) -> Self::Output {
        let const_var = var.tape.scalar_const(self, var.shape);
        const_var.binary(&var, BinaryOp::Div)
    }
}

impl<'t> Neg for VectorVar<'t> {
    type Output = VectorVar<'t>;

    fn neg(self) -> Self::Output {
        self.unary_vector(UnaryOp::Neg)
    }
}

pub struct Grad {
    derivs: Vec<Matrix>,
}

impl Grad {
    pub fn wrt<'t>(&self, var: VectorVar<'t>) -> &Matrix {
        &self.derivs[var.index]
    }
}
