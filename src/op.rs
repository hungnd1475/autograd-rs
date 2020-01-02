use crate::alg::{FloatMatrix, MatrixFunc};
use crate::tape::Node;
use crate::var::{Scalar, Shape};

#[derive(Clone, Copy, Debug, PartialEq)]
/// Represents the supported unary operations.
pub(crate) enum UnaryOp {
    T,
    Neg,
    Sin,
    Cos,
    Tan,
    Ln,
    Exp,
    Sum,
}

impl UnaryOp {
    pub(crate) fn eval_shape(&self, input_shape: Shape) -> Shape {
        match (self, input_shape) {
            (UnaryOp::T, Shape(nrow, ncol)) => Shape(ncol, nrow),
            (UnaryOp::Sum, _) => Scalar::shape(),
            _ => input_shape,
        }
    }

    /// Evaluates the operation with the given parameter.
    pub(crate) fn eval(&self, value: &FloatMatrix) -> FloatMatrix {
        match self {
            UnaryOp::T => value.t().to_owned(),
            UnaryOp::Neg => -value,
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Exp => value.exp(),
            UnaryOp::Sum => FloatMatrix::from_elem((1, 1), value.sum()),
        }
    }

    /// Computes the gradient of the operation with respect to the given parameter.
    pub(crate) fn grad(&self, node: &Node, ans: &FloatMatrix, g: &FloatMatrix) -> FloatMatrix {
        let val = node.value();
        match node {
            Node::Constant(value) => value.zeros_like(),
            _ => match self {
                UnaryOp::T => g.t().to_owned(),
                UnaryOp::Neg => -g,
                UnaryOp::Sin => val.cos() * g,
                UnaryOp::Cos => -val.sin() * g,
                UnaryOp::Tan => 2.0 * g / ((2.0 * val).cos() + 1.0),
                UnaryOp::Ln => g / val,
                UnaryOp::Exp => ans * g,
                UnaryOp::Sum => FloatMatrix::ones(node.shape().dim()),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Represents the supported binary operations.
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Dot,
}

impl BinaryOp {
    pub(crate) fn eval_shape(&self, left_shape: Shape, right_shape: Shape) -> Shape {
        match self {
            BinaryOp::Dot => Shape(left_shape.0, right_shape.1),
            _ => left_shape,
        }
    }

    /// Evaluates the operation with the given parameters.
    pub(crate) fn eval(&self, left: &FloatMatrix, right: &FloatMatrix) -> FloatMatrix {
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
    fn left_grad(
        &self,
        left: &FloatMatrix,
        right: &FloatMatrix,
        _ans: &FloatMatrix,
        g: &FloatMatrix,
    ) -> FloatMatrix {
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
    fn right_grad(
        &self,
        left: &FloatMatrix,
        right: &FloatMatrix,
        _ans: &FloatMatrix,
        g: &FloatMatrix,
    ) -> FloatMatrix {
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
    pub(crate) fn grad(
        &self,
        left_node: &Node,
        right_node: &Node,
        ans: &FloatMatrix,
        g: &FloatMatrix,
    ) -> [FloatMatrix; 2] {
        let left_val = left_node.value();
        let right_val = right_node.value();

        let lg = {
            match left_node {
                Node::Constant(value) => value.zeros_like(),
                _ => self.left_grad(left_val, right_val, ans, g),
            }
        };

        let rg = {
            match right_node {
                Node::Constant(value) => value.zeros_like(),
                _ => self.right_grad(left_val, right_val, ans, g),
            }
        };

        [lg, rg]
    }
}
