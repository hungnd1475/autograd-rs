mod op;
mod var;

pub use ndarray::array;
use ndarray::{Array1, Array2};
use op::*;
use std::cell::{Cell, RefCell};
pub use var::*;

type FloatVector = Array1<f64>;
type FloatMatrix = Array2<f64>;
type Shape = (usize, usize);

pub trait ShapeExt {
    fn is_vector(&self) -> bool;
    fn is_row_vector(&self) -> bool;
    fn is_col_vector(&self) -> bool;
    fn is_scalar(&self) -> bool;
}

impl ShapeExt for Shape {
    fn is_vector(&self) -> bool {
        let (nrow, ncol) = *self;
        nrow == 1 || ncol == 1
    }

    fn is_row_vector(&self) -> bool {
        let (nrow, ncol) = *self;
        nrow == 1 && ncol != 1
    }

    fn is_col_vector(&self) -> bool {
        let (nrow, ncol) = *self;
        nrow != 1 && ncol == 1
    }

    fn is_scalar(&self) -> bool {
        let (nrow, ncol) = *self;
        nrow == 1 && ncol == 1
    }
}

pub trait MatrixExt {
    fn zeros_like(&self) -> FloatMatrix;
    fn ones_like(&self) -> FloatMatrix;

    fn sin(&self) -> FloatMatrix;
    fn cos(&self) -> FloatMatrix;
    fn tan(&self) -> FloatMatrix;
    fn ln(&self) -> FloatMatrix;
    fn exp(&self) -> FloatMatrix;

    fn pow(&self, power: &FloatMatrix) -> FloatMatrix;
    fn pow_scalar(&self, p: f64) -> FloatMatrix;
    fn log(&self, base: &FloatMatrix) -> FloatMatrix;
}

impl MatrixExt for FloatMatrix {
    fn zeros_like(&self) -> FloatMatrix {
        FloatMatrix::zeros(self.dim())
    }

    fn ones_like(&self) -> FloatMatrix {
        FloatMatrix::ones(self.dim())
    }

    fn sin(&self) -> FloatMatrix {
        self.mapv(|x| x.sin())
    }

    fn cos(&self) -> FloatMatrix {
        self.mapv(|x| x.cos())
    }

    fn tan(&self) -> FloatMatrix {
        self.mapv(|x| x.tan())
    }

    fn ln(&self) -> FloatMatrix {
        self.mapv(|x| x.ln())
    }

    fn exp(&self) -> FloatMatrix {
        self.mapv(|x| x.exp())
    }

    fn pow(&self, power: &FloatMatrix) -> FloatMatrix {
        let mut result = self.clone();
        result.zip_mut_with(power, |x, y| *x = x.powf(*y));
        result
    }

    fn pow_scalar(&self, p: f64) -> FloatMatrix {
        let power = FloatMatrix::from_elem(self.dim(), p);
        self.pow(&power)
    }

    fn log(&self, base: &FloatMatrix) -> FloatMatrix {
        let mut result = self.clone();
        result.zip_mut_with(base, |x, y| *x = x.log(*y));
        result
    }
}

#[derive(Debug)]
/// Presents the operand in the graph.
enum Node {
    /// Represents a constant.
    Constant(FloatMatrix),
    /// Represents an input variable.
    Nullary {
        shape: Shape,
        value: Option<FloatMatrix>,
    },
    /// Represents the result of an unary operation.
    Unary {
        shape: Shape,               // the shape
        value: Option<FloatMatrix>, // the lazily computed value
        // grad: Option<Matrix>,  // the computed gradients
        op: UnaryOp, // the operation resulting in the node
        dep: usize,  // the operand for this node's operation
    },
    /// Represents the result of an binary operation.
    Binary {
        shape: Shape,               // the shape
        value: Option<FloatMatrix>, // the lazily computed value
        // grads: Option<(Matrix, Matrix)>, // the computed gradients
        op: BinaryOp,     // the operation resulting in the node
        deps: [usize; 2], // the operands for this node's operation
    },
}

impl Node {
    fn shape(&self) -> Shape {
        match self {
            Node::Constant(value) => value.dim(),
            Node::Nullary { shape, .. }
            | Node::Unary { shape, .. }
            | Node::Binary { shape, .. } => *shape,
        }
    }

    fn value(&self) -> &FloatMatrix {
        match self {
            Node::Constant(value) => value,
            Node::Nullary { value, .. }
            | Node::Unary { value, .. }
            | Node::Binary { value, .. } => value.as_ref().unwrap(),
        }
    }
}

/// Represents a tape used to construct the adjoint graph.
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    is_evaluated: Cell<bool>,
}

impl Tape {
    /// Initializes a new tape.
    pub fn new() -> Self {
        Self {
            // grad_nodes: RefCell::new(Vec::new()),
            nodes: RefCell::new(Vec::new()),
            is_evaluated: Cell::new(false),
        }
    }

    pub fn scalar_var<'t>(&'t self) -> ScalarVar<'t> {
        let index = self.push_nullary(None, (1, 1));
        ScalarVar::new(self, index)
    }

    /// Creates a node representing a vector input variable.
    pub fn vector_var<'t>(&'t self, length: usize) -> VectorVar<'t> {
        let shape = (length, 1);
        let index = self.push_nullary(None, shape);
        VectorVar::new(self, shape, index)
    }

    /// Creates a node representing a matrix input variable.
    pub fn matrix_var<'t>(&'t self, nrow: usize, ncol: usize) -> MatrixVar<'t> {
        let shape = (nrow, ncol);
        let index = self.push_nullary(None, shape);
        MatrixVar::new(self, shape, index)
    }

    fn scalar_const<'t>(&'t self, value: f64) -> ScalarVar<'t> {
        let value = FloatMatrix::from_elem((1, 1), value);
        let index = self.push_constant(value);
        ScalarVar::new(self, index)
    }

    fn vector_const<'t>(&'t self, value: FloatVector) -> VectorVar<'t> {
        let shape = (value.len(), 1);
        let value = value.into_shape(shape).unwrap();
        let index = self.push_constant(value);
        VectorVar::new(self, shape, index)
    }

    fn matrix_const<'t>(&'t self, value: FloatMatrix) -> MatrixVar<'t> {
        let shape = value.dim();
        let index = self.push_constant(value);
        MatrixVar::new(self, shape, index)
    }

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    fn push_constant(&self, value: FloatMatrix) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Constant(value));
        len
    }

    /// Pushes a node representing an input variable onto the graph.
    fn push_nullary(&self, value: Option<FloatMatrix>, shape: Shape) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Nullary { shape, value });
        len
    }

    /// Pushes a node representing the result of an unary operator onto the graph.
    fn push_unary(&self, shape: Shape, dep: usize, op: UnaryOp) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Unary {
            shape,
            value: None,
            dep,
            op,
        });
        len
    }

    /// Pushes a node representing the result of a binary operator onto the graph.
    fn push_binary(&self, shape: Shape, dep0: usize, dep1: usize, op: BinaryOp) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Binary {
            shape,
            value: None,
            deps: [dep0, dep1],
            op,
        });
        len
    }
}
