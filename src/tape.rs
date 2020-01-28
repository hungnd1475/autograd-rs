use crate::op::{BinaryOp, UnaryOp};
use crate::var::{Matrix, Nullary, Scalar, Shape, Var, VarKind, Vector};
use crate::FloatMatrix;
use std::cell::RefCell;
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
            | Node::Binary { value, .. } => value.as_ref().expect("Node is not yet evaluated."),
        }
    }
}

/// Represents a tape used to construct the adjoint graph.
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    nodes_order: RefCell<Option<Vec<usize>>>,
}

impl Tape {
    /// Initializes a new tape.
    pub fn new() -> TapeContainer {
        TapeContainer {
            tape: Rc::new(Tape {
                nodes: RefCell::new(Vec::new()),
                nodes_order: RefCell::new(None),
            }),
        }
    }

    /// Sets the node's value at the given index.
    pub fn set_value(&self, index: usize, new_value: FloatMatrix) {
        let mut nodes = self.nodes.borrow_mut();
        match &mut nodes[index] {
            Node::Nullary { ref mut value, .. } => *value = Some(new_value),
            _ => panic!("Cannot set value for dependent variable."),
        }
        self.nodes_order.replace(None);
    }

    /// Sorts the nodes in topological order starting from the given index.
    fn topological_sort(&self, index: usize) -> Vec<usize> {
        let nodes = self.nodes.borrow();
        let mut visited = vec![false; nodes.len()]; // flag visited nodes
        let mut visit_stack = Vec::with_capacity(nodes.len()); // used to store the visited nodes
        let mut nodes_stack = Vec::with_capacity(nodes.len()); // used to store the traversal result
        let mut root = Some(index);

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

    pub(crate) fn eval(&self, index: usize) -> FloatMatrix {
        let nodes_order = self.topological_sort(index);
        let mut nodes = self.nodes.borrow_mut();

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

        self.nodes_order.replace(Some(nodes_order));
        nodes[index].value().clone()
    }

    /// Computes the gradients of the variable with respects to all of its parameters.
    pub(crate) fn grad(&self, index: usize) -> Grad {
        let nodes_order_ref = self.nodes_order.borrow();
        let nodes_order = nodes_order_ref
            .as_ref()
            .expect("Tape must be evaluated before gradient can be computed.");
        let nodes = self.nodes.borrow();
        let mut derivs: Vec<FloatMatrix> = nodes
            .iter()
            .map(|x| FloatMatrix::zeros(x.shape().dim()))
            .collect();
        derivs[index] = FloatMatrix::ones(derivs[index].dim());

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

    /// Gets the length of the tape.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Pushes a node representing a constant onto the graph.
    pub(crate) fn push_constant(&self, value: FloatMatrix) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Constant(value));
        self.nodes_order.replace(None);
        len
    }

    /// Pushes a node representing an input variable onto the graph.
    pub(crate) fn push_nullary(&self, value: Option<FloatMatrix>, shape: Shape) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::Nullary { shape, value });
        self.nodes_order.replace(None);
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
        self.nodes_order.replace(None);
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
        self.nodes_order.replace(None);
        len
    }
}

pub struct TapeContainer {
    tape: Rc<Tape>,
}

impl TapeContainer {
    /// Creates a new scalar free variable.
    pub fn scalar_var(&self) -> Var<Scalar, Nullary> {
        let index = self.tape.push_nullary(None, Shape([1, 1]));
        Var::scalar(&self.tape, index)
    }

    /// Creates a new vector free variable.
    pub fn vector_var(&self, length: usize) -> Var<Vector, Nullary> {
        let shape = Shape([length, 1]);
        let index = self.tape.push_nullary(None, shape);
        Var::vector(&self.tape, index, shape)
    }

    /// Creates a new matrix free variable.
    pub fn matrix_var(&self, nrow: usize, ncol: usize) -> Var<Matrix, Nullary> {
        let index = self.tape.push_nullary(None, Shape([nrow, ncol]));
        Var::matrix(&self.tape, index, nrow, ncol)
    }
}

pub struct Grad {
    derivs: Vec<FloatMatrix>,
}

impl Grad {
    pub fn wrt<K, S>(&self, var: &Var<K, S>) -> &FloatMatrix
    where
        K: VarKind,
    {
        &self.derivs[var.index]
    }
}
