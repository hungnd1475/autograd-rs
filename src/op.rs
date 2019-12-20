use crate::{Matrix, MatrixExt, Shape, VarNode};

#[derive(Clone, Copy, Debug)]
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
            (UnaryOp::T, (nrow, ncol)) => (ncol, nrow),
            _ => input_shape,
        }
    }

    /// Evaluates the operation with the given parameter.
    pub(crate) fn eval(&self, value: &Matrix) -> Matrix {
        match self {
            UnaryOp::T => value.t().to_owned(),
            UnaryOp::Neg => -value,
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Exp => value.exp(),
            UnaryOp::Sum => Matrix::from_elem((1, 1), value.sum()),
        }
    }

    /// Computes the gradient of the operation with respect to the given parameter.
    pub(crate) fn grad(&self, var: &VarNode, ans: &Matrix, g: &Matrix) -> Matrix {
        let val = var.value();
        match var {
            VarNode::Constant(value) => value.zeros_like(),
            _ => match self {
                UnaryOp::T => g.t().to_owned(),
                UnaryOp::Neg => -g,
                UnaryOp::Sin => val.cos() * g,
                UnaryOp::Cos => -val.sin() * g,
                UnaryOp::Tan => 2.0 * g / ((2.0 * val).cos() + 1.0),
                UnaryOp::Ln => g / val,
                UnaryOp::Exp => ans * g,
                UnaryOp::Sum => Matrix::ones(var.shape()),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
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
            BinaryOp::Dot => (left_shape.0, right_shape.1),
            _ => left_shape,
        }
    }

    /// Evaluates the operation with the given parameters.
    pub(crate) fn eval(&self, left: &Matrix, right: &Matrix) -> Matrix {
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
    pub(crate) fn grad(
        &self,
        left_var: &VarNode,
        right_var: &VarNode,
        ans: &Matrix,
        g: &Matrix,
    ) -> [Matrix; 2] {
        let left_val = left_var.value();
        let right_val = right_var.value();

        let lg = {
            match left_var {
                VarNode::Constant(value) => value.zeros_like(),
                _ => self.left_grad(left_val, right_val, ans, g),
            }
        };

        let rg = {
            match right_var {
                VarNode::Constant(value) => value.zeros_like(),
                _ => self.right_grad(left_val, right_val, ans, g),
            }
        };

        [lg, rg]
    }
}
