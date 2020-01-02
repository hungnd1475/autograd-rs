use crate::alg::FloatMatrix;
use crate::op::{BinaryOp, UnaryOp};
use crate::var::{Matrix, Nullary, Scalar, Shape, Var, Vector};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Debug)]
/// Presents the operand in the graph.
pub(crate) enum Node {
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
        op: UnaryOp,                // the operation resulting in the node
        dep: usize,                 // the operand for this node's operation
    },
    /// Represents the result of an binary operation.
    Binary {
        shape: Shape,               // the shape
        value: Option<FloatMatrix>, // the lazily computed value
        op: BinaryOp,               // the operation resulting in the node
        deps: [usize; 2],           // the operands for this node's operation
    },
}

impl Node {
    /// Returns the shape of this node.
    pub(crate) fn shape(&self) -> Shape {
        match self {
            Node::Constant(value) => value.dim().into(),
            Node::Nullary { shape, .. }
            | Node::Unary { shape, .. }
            | Node::Binary { shape, .. } => *shape,
        }
    }

    /// Returns the value of this node.
    pub(crate) fn value(&self) -> &FloatMatrix {
        match self {
            Node::Constant(value) => value,
            Node::Nullary { value, .. }
            | Node::Unary { value, .. }
            | Node::Binary { value, .. } => value.as_ref().unwrap(),
        }
    }
}

#[derive(Debug)]
/// Represents a tape used to construct the adjoint graph.
pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
    pub(crate) is_evaluated: Cell<bool>,
}

impl Tape {
    /// Initializes a new tape.
    pub fn new() -> TapeContainer {
        TapeContainer {
            tape: Rc::new(Tape {
                nodes: RefCell::new(Vec::new()),
                is_evaluated: Cell::new(false),
            }),
        }
    }

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Pushes a node representing a constant onto the graph.
    pub(crate) fn push_constant(&self, value: FloatMatrix) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Constant(value));
        len
    }

    /// Pushes a node representing an input variable onto the graph.
    pub(crate) fn push_nullary(&self, value: Option<FloatMatrix>, shape: Shape) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Nullary { shape, value });
        len
    }

    /// Pushes a node representing the result of an unary operator onto the graph.
    pub(crate) fn push_unary(&self, shape: Shape, dep: usize, op: UnaryOp) -> usize {
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
    pub(crate) fn push_binary(
        &self,
        shape: Shape,
        dep0: usize,
        dep1: usize,
        op: BinaryOp,
    ) -> usize {
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

pub struct TapeContainer {
    tape: Rc<Tape>,
}

impl TapeContainer {
    /// Creates a new scalar free variable.
    pub fn scalar_var(&self) -> Var<Scalar, Nullary> {
        let index = self.tape.push_nullary(None, Shape(1, 1));
        Var::scalar(&self.tape, index)
    }

    /// Creates a new vector free variable.
    pub fn vector_var(&self, length: usize) -> Var<Vector, Nullary> {
        let shape = Shape(length, 1);
        let index = self.tape.push_nullary(None, shape);
        Var::vector(&self.tape, index, shape)
    }

    /// Creates a new matrix free variable.
    pub fn matrix_var(&self, nrow: usize, ncol: usize) -> Var<Matrix, Nullary> {
        let index = self.tape.push_nullary(None, Shape(nrow, ncol));
        Var::matrix(&self.tape, index, nrow, ncol)
    }
}
